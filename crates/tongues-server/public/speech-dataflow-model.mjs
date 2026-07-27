export const PIPELINE_SCHEMA_VERSION = 1;

const PORTS = Object.freeze({
  source:{in:null,out:"audio"}, cleanup:{in:"audio",out:"audio"}, vad:{in:"audio",out:"speech_audio"},
  segmentation:{in:"speech_audio",out:"segments"}, language_id:{in:"segments",out:"routed_segments"},
  diarization:{in:["segments","routed_segments"],out:"speaker_segments"},
  asr:{in:["audio","speech_audio","segments","routed_segments","speaker_segments"],out:"recognition_events"},
  normalization:{in:"recognition_events",out:"committed_text"}, parser:{in:"committed_text",out:"syntax"},
  interpretation:{in:"syntax",out:"meaning"}, response:{in:["committed_text","meaning"],out:"generated_text"},
  tts:{in:["generated_text","meaning"],out:"audio"}, output:{in:"audio",out:null},
});

export function buildCatalog(discovery) {
  const nodes = [];
  const add = (kind, id, label, capability_id, config = {}) =>
    nodes.push({kind,id:`${kind}:${id}`,label,capability_id,config,ports:PORTS[kind]});
  for (const kind of discovery.audio?.source_kinds ?? []) add("source",kind,`${kind} audio`,`audio-input/source/${kind}`);
  for (const stage of discovery.audio?.cleanup_stages ?? []) add("cleanup",stage.kind,stage.kind,`audio-cleanup/${stage.kind}`,stage);
  add("vad","registered","Voice activity detection","audio-input/vad");
  add("segmentation","registered","Utterance segmentation","audio-input/segmentation");
  for (const detector of discovery.language?.detectors ?? []) add("language_id",detector.detector_id,detector.detector_id,detector.detector_id,detector);
  add("diarization","anonymous","Anonymous diarization","diarization/anonymous");
  for (const provider of discovery.asr?.providers ?? []) add("asr",provider.provider_id,provider.provider_id,provider.provider_id,provider);
  const commands = flattenCommands(discovery.cli?.commands ?? []);
  for (const [kind, fragment] of [["normalization","normalize"],["parser","sentence-parser"],["interpretation","interpret"]]) {
    const command = commands.find(item => item.id?.includes(fragment));
    add(kind,command?.id ?? kind,kind,command?.id ?? kind);
  }
  for (const provider of discovery.live?.providers ?? []) add("response",provider.id,provider.label,provider.id,provider);
  const speechItems = [
    ...(discovery.speech?.compositions ?? []),
    ...(discovery.speech?.paths ?? []),
  ];
  for (const model of speechItems) add("tts",model.id,model.display_name ?? model.id,model.id,model);
  add("output","browser","Browser audio output","audio-output/browser");
  return nodes;
}

function flattenCommands(commands) {
  return commands.flatMap(command => [command,...flattenCommands(command.subcommands ?? [])]);
}

export function createPipeline(name = "Untitled pipeline") {
  return {schema_version:PIPELINE_SCHEMA_VERSION,id:`pipeline:${Date.now()}`,name,nodes:[],edges:[],revision:0};
}

export function addNode(pipeline, catalogNode, afterId = null) {
  const node = {
    instance_id:`node:${Date.now()}:${pipeline.nodes.length}`,catalog_id:catalogNode.id,
    kind:catalogNode.kind,label:catalogNode.label,capability_id:catalogNode.capability_id,
    config:structuredClone(catalogNode.config),bypassed:false,
  };
  const index = afterId ? pipeline.nodes.findIndex(item => item.instance_id === afterId) + 1 : pipeline.nodes.length;
  pipeline.nodes.splice(Math.max(0,index),0,node); pipeline.revision++; return node;
}

export function removeNode(pipeline, id) {
  pipeline.nodes = pipeline.nodes.filter(node => node.instance_id !== id);
  pipeline.edges = pipeline.edges.filter(edge => edge.from !== id && edge.to !== id); pipeline.revision++;
}

export function duplicateNode(pipeline, id) {
  const source = pipeline.nodes.find(node => node.instance_id === id);
  if (!source) return null;
  const copy = {...structuredClone(source),instance_id:`node:${Date.now()}:${pipeline.nodes.length}`,label:`${source.label} copy`};
  pipeline.nodes.splice(pipeline.nodes.indexOf(source)+1,0,copy); pipeline.revision++; return copy;
}

export function replaceNode(pipeline, id, catalogNode) {
  const node = pipeline.nodes.find(item => item.instance_id === id);
  if (!node || node.kind !== catalogNode.kind) throw new Error("Replacement must have the same typed stage kind.");
  Object.assign(node,{catalog_id:catalogNode.id,label:catalogNode.label,capability_id:catalogNode.capability_id,config:structuredClone(catalogNode.config)});
  pipeline.revision++;
}

export function moveNode(pipeline, id, delta) {
  const index = pipeline.nodes.findIndex(node => node.instance_id === id);
  const target = Math.max(0,Math.min(pipeline.nodes.length-1,index+delta));
  if (index < 0 || index === target) return;
  pipeline.nodes.splice(target,0,pipeline.nodes.splice(index,1)[0]); pipeline.revision++;
}

export function toggleBypass(pipeline, id) {
  const node = pipeline.nodes.find(item => item.instance_id === id);
  if (node) { node.bypassed = !node.bypassed; pipeline.revision++; }
}

export function connect(pipeline, from, to) {
  const result = connectionCompatibility(pipeline,from,to);
  if (!result.valid) throw new Error(result.reason);
  pipeline.edges = pipeline.edges.filter(edge => edge.to !== to);
  pipeline.edges.push({from,to}); pipeline.revision++;
}

export function connectionCompatibility(pipeline, from, to) {
  const source = pipeline.nodes.find(node => node.instance_id === from);
  const target = pipeline.nodes.find(node => node.instance_id === to);
  if (!source || !target) return {valid:false,reason:"Both connection endpoints must exist."};
  if (source.bypassed || target.bypassed) return {valid:false,reason:"Bypassed stages cannot be connected."};
  const output = PORTS[source.kind]?.out, input = PORTS[target.kind]?.in;
  const valid = Array.isArray(input) ? input.includes(output) : input === output;
  return valid ? {valid:true,reason:`${output} → ${Array.isArray(input)?input.join("|"):input}`}
    : {valid:false,reason:`${source.label} emits ${output ?? "nothing"}; ${target.label} requires ${Array.isArray(input)?input.join(" or "):input ?? "no input"}.`};
}

export function validatePipeline(pipeline) {
  if (pipeline.schema_version !== PIPELINE_SCHEMA_VERSION) return {valid:false,errors:[`Pipeline schema ${pipeline.schema_version} is unsupported; expected 1.`]};
  const errors = [];
  for (const edge of pipeline.edges) {
    const result = connectionCompatibility(pipeline,edge.from,edge.to);
    if (!result.valid) errors.push(result.reason);
  }
  const active = pipeline.nodes.filter(node => !node.bypassed);
  for (const [index,node] of active.entries()) {
    const ports = PORTS[node.kind];
    if (!ports) errors.push(`Unknown stage kind ${node.kind}.`);
    if (ports?.in && !pipeline.edges.some(edge => edge.to === node.instance_id)) errors.push(`${node.label} has no input.`);
    if (ports?.out && index < active.length-1 && !pipeline.edges.some(edge => edge.from === node.instance_id)) errors.push(`${node.label} has no output.`);
  }
  return {valid:errors.length===0,errors};
}

export function template(kind, catalog) {
  const definitions = {
    transcription:["source","vad","asr","normalization"],
    multilingual_transcription:["source","vad","segmentation","language_id","asr","normalization"],
    meeting_transcript:["source","vad","segmentation","diarization","asr","normalization"],
    spoken_interpretation:["source","vad","asr","normalization","parser","interpretation","tts","output"],
    full_conversation:["source","vad","asr","normalization","response","tts","output"],
  };
  const pipeline = createPipeline(kind.replaceAll("_"," "));
  for (const stage of definitions[kind] ?? []) {
    const candidates = catalog.filter(node => node.kind === stage);
    const selected = candidates.find(node => node.config?.installed !== false && node.config?.available !== false) ?? candidates[0];
    if (!selected) continue;
    const previous = pipeline.nodes.at(-1);
    const node = addNode(pipeline,selected);
    if (previous && connectionCompatibility(pipeline,previous.instance_id,node.instance_id).valid) connect(pipeline,previous.instance_id,node.instance_id);
  }
  return pipeline;
}
