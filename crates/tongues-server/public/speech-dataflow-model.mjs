export const PIPELINE_SCHEMA_VERSION = 2;
const LAYOUT_LABEL = "studio.layout.v1";

export function buildCatalog(discovery) {
  const kinds = discovery.node_kinds ?? {};
  const nodes = [];
  for (const kind of Object.values(kinds)) {
    if (!kind.requires_component) nodes.push(catalogEntry(kind));
  }
  for (const component of Object.values(discovery.components ?? {})) {
    const kind = kinds[component.node_kind];
    if (!kind) continue;
    nodes.push(catalogEntry(kind, component));
  }
  return nodes.sort((a,b)=>a.group.localeCompare(b.group)||a.label.localeCompare(b.label));
}

export function catalogEntryForNode(node,catalog) {
  const componentId=node?.component_id??null;
  return catalog.find(item=>item.kind===node?.kind&&(item.component_id??null)===componentId);
}

function catalogEntry(kind, component=null) {
  return {
    id:component?`component:${component.id}`:`kind:${kind.kind}`,
    kind:kind.kind,
    label:component?`${kind.label} · ${component.provider} / ${component.model}`:kind.label,
    component_id:component?.id??null,
    config:structuredClone(component?.default_config??kind.default_config??{}),
    schema:structuredClone(component?.configuration_schema??kind.configuration_schema??{}),
    ports:structuredClone(kind.ports??[]),
    readiness:component?.readiness??"ready",
    detail:component?.detail??"",
    adapter:kind.adapter??null,
    merge:kind.merge??null,
    group:capabilityGroup(kind),
  };
}

function capabilityGroup(kind) {
  const capabilities=[...(kind.required_capabilities??[])];
  const portTypes=(kind.ports??[]).map(port=>port.value_type);
  if (kind.adapter) return "Audio & linguistic processing";
  if (kind.merge) return "Inspection & control";
  if (["microphone","audio_file","text_source","text_file","text_url","control_source"].includes(kind.kind)) return "Sources";
  if (kind.kind==="asr") return "Recognition";
  if (kind.kind==="diarization") return "Language & speaker analysis";
  if (["linguistic","interpretation"].includes(kind.kind)) return "Linguistic processing";
  if (capabilities.includes("text_generation")) return "Response generation";
  if (kind.kind==="tts") return "Synthesis";
  if (kind.kind.endsWith("_sink")||kind.kind==="audio_output") return "Outputs";
  if (portTypes.some(type=>type.startsWith("audio"))) return "Audio processing";
  return "Inspection & control";
}

export function createPipeline(name="Untitled pipeline") {
  return {
    schema_version:PIPELINE_SCHEMA_VERSION,graph_id:`pipeline:${cryptoId()}`,
    revision:1,metadata:{name,description:"",allow_unsafe_execution:false,labels:{}},
    nodes:[],edges:[],selected_sinks:[],
  };
}

function cryptoId() {
  return globalThis.crypto?.randomUUID?.()??`${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function addNode(pipeline,catalogNode,afterId=null,position=null) {
  if (!catalogNode) throw new Error("Choose a backend-discovered node first.");
  const node={
    id:`node:${cryptoId()}`,kind:catalogNode.kind,
    component_id:catalogNode.component_id,config:structuredClone(catalogNode.config??{}),
    disabled:false,bypassed:false,
  };
  const index=afterId?pipeline.nodes.findIndex(item=>item.id===afterId)+1:pipeline.nodes.length;
  pipeline.nodes.splice(Math.max(0,index),0,node);
  if(position)setNodePosition(pipeline,node.id,position);
  touch(pipeline); return node;
}

export function removeNode(pipeline,id) {
  pipeline.nodes=pipeline.nodes.filter(node=>node.id!==id);
  pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==id&&edge.to.node_id!==id);
  pipeline.selected_sinks=pipeline.selected_sinks.filter(sink=>sink.node_id!==id);
  const layout=readLayout(pipeline);delete layout[id];writeLayout(pipeline,layout);touch(pipeline);
}

export function duplicateNode(pipeline,id,offset={x:36,y:36}) {
  const source=pipeline.nodes.find(node=>node.id===id);
  if(!source)return null;
  const copy={...structuredClone(source),id:`node:${cryptoId()}`};
  pipeline.nodes.splice(pipeline.nodes.indexOf(source)+1,0,copy);
  const position=nodePosition(pipeline,id);
  if(position)setNodePosition(pipeline,copy.id,{x:position.x+offset.x,y:position.y+offset.y});
  touch(pipeline);return copy;
}

export function applyNodeConfig(pipeline,id,values) {
  const node=pipeline.nodes.find(item=>item.id===id);
  if(!node)throw new Error("The selected node is no longer present.");
  Object.assign(node.config,structuredClone(values));
  touch(pipeline);return node;
}

export function replaceNode(pipeline,id,catalogNode) {
  const node=pipeline.nodes.find(item=>item.id===id);
  if(!node||node.kind!==catalogNode?.kind)throw new Error("Replacement must have the same backend node kind.");
  Object.assign(node,{component_id:catalogNode.component_id,config:structuredClone(catalogNode.config??{})});
  touch(pipeline);
}

export function insertSubgraph(pipeline,template,origin={x:80,y:80}) {
  const copy=structuredClone(template), idMap=new Map();
  copy.nodes.forEach((node,index)=>{
    const old=node.id;node.id=`node:${cryptoId()}`;idMap.set(old,node.id);
    pipeline.nodes.push(node);
    const row=Math.floor(index/3),column=row%2===0?index%3:2-(index%3);
    setNodePosition(pipeline,node.id,{x:origin.x+column*290,y:origin.y+row*220});
  });
  copy.edges.forEach(edge=>{
    edge.id=`edge:${cryptoId()}`;edge.from.node_id=idMap.get(edge.from.node_id);
    edge.to.node_id=idMap.get(edge.to.node_id);pipeline.edges.push(edge);
  });
  touch(pipeline);return copy.nodes.map(node=>node.id);
}

export function portsFor(node,direction,discovery) {
  return (discovery.node_kinds?.[node?.kind]?.ports??[]).filter(port=>port.direction===direction);
}

export function compatibleTargets(pipeline,fromNodeId,fromPortId,discovery) {
  const source=pipeline.nodes.find(node=>node.id===fromNodeId);
  const output=portsFor(source,"output",discovery).find(port=>port.id===fromPortId);
  if(!output)return[];
  return pipeline.nodes.flatMap(node=>portsFor(node,"input",discovery)
    .filter(port=>port.value_type===output.value_type)
    .map(port=>({node_id:node.id,port_id:port.id,value_type:port.value_type})));
}

export function adapterPaths(fromType,toType,discovery) {
  return Object.values(discovery.node_kinds??{})
    .filter(kind=>kind.adapter?.from===fromType&&kind.adapter?.to===toType)
    .map(kind=>({kind:kind.kind,label:kind.label}));
}

export function connectPorts(pipeline,fromNode,fromPort,toNode,toPort,discovery) {
  const source=pipeline.nodes.find(node=>node.id===fromNode),target=pipeline.nodes.find(node=>node.id===toNode);
  if(!source||!target)throw new Error("Both connection endpoints must exist.");
  const output=portsFor(source,"output",discovery).find(port=>port.id===fromPort);
  const input=portsFor(target,"input",discovery).find(port=>port.id===toPort);
  if(!output||!input)throw new Error("Choose an output port and an input port.");
  if(output.value_type!==input.value_type){
    const adapters=adapterPaths(output.value_type,input.value_type,discovery);
    const route=adapters.length?` Add ${adapters.map(item=>item.label).join(" or ")} between them.`:" No registered adapter path is available.";
    throw new Error(`${source.kind}.${output.id} emits ${output.value_type}; ${target.kind}.${input.id} requires ${input.value_type}.${route}`);
  }
  if(input.cardinality!=="many")pipeline.edges=pipeline.edges.filter(edge=>!(edge.to.node_id===toNode&&edge.to.port_id===toPort));
  const duplicate=pipeline.edges.some(edge=>edge.from.node_id===fromNode&&edge.from.port_id===fromPort&&edge.to.node_id===toNode&&edge.to.port_id===toPort);
  if(duplicate)return;
  pipeline.edges.push({id:`edge:${cryptoId()}`,from:{node_id:fromNode,port_id:fromPort},to:{node_id:toNode,port_id:toPort},capacity:16});
  touch(pipeline);
}

export function connect(pipeline,from,to,discovery) {
  const source=pipeline.nodes.find(node=>node.id===from),target=pipeline.nodes.find(node=>node.id===to);
  const pairs=portsFor(source,"output",discovery).flatMap(output=>
    portsFor(target,"input",discovery).filter(input=>input.value_type===output.value_type).map(input=>({output,input})));
  if(!pairs.length){
    const outputs=portsFor(source,"output",discovery).map(port=>port.value_type);
    const inputs=portsFor(target,"input",discovery).map(port=>port.value_type);
    const adapters=outputs.flatMap(fromType=>inputs.flatMap(toType=>adapterPaths(fromType,toType,discovery)));
    throw new Error(`${source?.kind??"source"} emits ${outputs.join(" or ")||"nothing"}; ${target?.kind??"target"} requires ${inputs.join(" or ")||"no input"}.${adapters.length?` Add ${adapters.map(item=>item.label).join(" or ")}.`:""}`);
  }
  connectPorts(pipeline,from,pairs[0].output.id,to,pairs[0].input.id,discovery);
}

export function removeEdge(pipeline,id){pipeline.edges=pipeline.edges.filter(edge=>edge.id!==id);touch(pipeline);}

export function bypassNode(pipeline,id,discovery) {
  const incoming=pipeline.edges.filter(edge=>edge.to.node_id===id);
  const outgoing=pipeline.edges.filter(edge=>edge.from.node_id===id);
  if(incoming.length!==1||outgoing.length<1)throw new Error("Bypass needs exactly one input edge and at least one output edge.");
  const upstream=incoming[0];
  const source=pipeline.nodes.find(node=>node.id===upstream.from.node_id);
  const sourcePort=portsFor(source,"output",discovery).find(port=>port.id===upstream.from.port_id);
  const replacements=outgoing.map(edge=>{
    const target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
    const input=portsFor(target,"input",discovery).find(port=>port.id===edge.to.port_id);
    if(!sourcePort||!input||sourcePort.value_type!==input.value_type)throw new Error(`Cannot bypass ${source?.kind}: ${sourcePort?.value_type??"unknown"} does not satisfy ${target?.kind}.${input?.id??"input"}.`);
    return{to:edge.to};
  });
  pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==id&&edge.to.node_id!==id);
  replacements.forEach(item=>connectPorts(pipeline,upstream.from.node_id,upstream.from.port_id,item.to.node_id,item.to.port_id,discovery));
  const node=pipeline.nodes.find(item=>item.id===id);if(node)node.bypassed=true;touch(pipeline);
}

export function nodeLabel(node,catalog) {
  return catalogEntryForNode(node,catalog)?.label??node?.kind??"Unknown node";
}

export function readLayout(pipeline) {
  try{return JSON.parse(pipeline.metadata?.labels?.[LAYOUT_LABEL]??"{}");}catch{return{};}
}
function writeLayout(pipeline,layout) {
  pipeline.metadata.labels??={};pipeline.metadata.labels[LAYOUT_LABEL]=JSON.stringify(layout);
}
export function nodePosition(pipeline,id){return readLayout(pipeline)[id]??null;}
export function setNodePosition(pipeline,id,position){const layout=readLayout(pipeline);layout[id]={x:Math.round(position.x),y:Math.round(position.y)};writeLayout(pipeline,layout);}

export function ensureLayout(pipeline) {
  const layout=readLayout(pipeline);
  pipeline.nodes.forEach((node,index)=>{
    const row=Math.floor(index/3),column=row%2===0?index%3:2-(index%3);
    layout[node.id]??={x:130+column*290,y:140+row*220};
  });
  writeLayout(pipeline,layout);return layout;
}

export function diagnosticsByTarget(report) {
  const result={graph:[],nodes:{},ports:{},edges:{}};
  for(const item of report?.diagnostics??[]){
    const target=item.target??item;
    if(target.edge_id)(result.edges[target.edge_id]??=[]).push(item);
    else if(target.node_id&&target.port_id)(result.ports[`${target.node_id}:${target.port_id}`]??=[]).push(item);
    else if(target.node_id)(result.nodes[target.node_id]??=[]).push(item);
    else result.graph.push(item);
  }
  return result;
}

export function requiredPortState(pipeline,node,port,report) {
  const connected=pipeline.edges.some(edge=>edge.to.node_id===node.id&&edge.to.port_id===port.id);
  const diagnostics=diagnosticsByTarget(report).ports[`${node.id}:${port.id}`]??[];
  return {connected,missing:!connected&&port.direction==="input"&&port.cardinality==="one",diagnostics};
}

export function touch(pipeline){pipeline.revision=Math.max(1,Number(pipeline.revision)||1)+1;}

export async function consumeNdjson(reader,onEvent) {
  const decoder=new TextDecoder();let buffer="";
  while(true){
    const{done,value}=await reader.read();
    if(done)break;
    buffer+=decoder.decode(value,{stream:true});
    const lines=buffer.split("\n");buffer=lines.pop();
    for(const line of lines)if(line.trim())onEvent(JSON.parse(line));
  }
  buffer+=decoder.decode();
  if(buffer.trim())onEvent(JSON.parse(buffer));
}
