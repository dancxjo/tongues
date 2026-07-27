import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  adapterPaths,addNode,alignGraphSelection,applyNodeConfig,attachReplacementValidation,buildCatalog,bypassNode,classifyReplacement,commitReplacement,compatibleTargets,connectPorts,consumeNdjson,copyGraphSelection,createEditHistory,createPipeline,
  catalogEntryForNode,connectionCompatibility,deleteGraphSelection,diagnosticsByTarget,distributeGraphSelection,duplicateNode,ensureLayout,insertSubgraph,migrateReplacementConfig,moveGraphSelection,nodeLabel,
  pasteGraphSelection,planNodeReplacement,reconnectEdge,recordEdit,redoEdit,replacementCandidates,requiredPortState,tidyGraphSelection,undoEdit,validateSchemaValue,
} from "./speech-dataflow-model.mjs";

const replacement=(family,extra={})=>({family,configuration_schema_id:`fixture.${family}.config`,configuration_schema_version:1,port_aliases:{},configuration_aliases:{},disconnect_ports:[],...extra});
const asrSchema={type:"object",properties:{
  language:{type:"string",enum:["en","fr"]},timestamps:{type:"boolean"},beam:{type:"integer",minimum:1,maximum:8},
  notes:{type:"string"},tags:{type:"array",items:{type:"string"}},options:{type:"object",properties:{punctuate:{type:"boolean"}},additionalProperties:false},
},required:["language"]};
const discovery={node_kinds:{
  text_source:{kind:"text_source",label:"Inline text",requires_component:false,default_config:{text:"Hello from Tongues."},configuration_schema:{
    type:"object",properties:{text:{type:"string",title:"Text",format:"multiline",minLength:1}},required:["text"]},ports:[
    {id:"out",direction:"output",value_type:"text",cardinality:"many",streaming:true}],replacement:replacement("text_source")},
  microphone:{kind:"microphone",label:"Microphone",requires_component:false,default_config:{},ports:[
    {id:"out",direction:"output",value_type:"audio_stream",cardinality:"many"}],replacement:replacement("microphone")},
  asr:{kind:"asr",label:"ASR",requires_component:true,required_capabilities:["asr"],configuration_schema:asrSchema,ports:[
    {id:"audio",direction:"input",value_type:"audio_stream",cardinality:"one"},
    {id:"committed",direction:"output",value_type:"transcript_committed",cardinality:"many"}],replacement:replacement("asr")},
  asr_equivalent:{kind:"asr_equivalent",label:"ASR equivalent",requires_component:true,required_capabilities:["asr"],ports:[
    {id:"samples",direction:"input",value_type:"audio_stream",cardinality:"one"},
    {id:"text",direction:"output",value_type:"transcript_committed",cardinality:"many"}],replacement:replacement("asr",{port_aliases:{audio:"samples",committed:"text"},configuration_aliases:{language:"locale"}})},
  asr_missing_output:{kind:"asr_missing_output",label:"ASR missing output",requires_component:true,required_capabilities:["asr"],ports:[
    {id:"audio",direction:"input",value_type:"audio_stream",cardinality:"one"}],replacement:replacement("asr")},
  to_text:{kind:"to_text",label:"Transcript to text",requires_component:false,adapter:{from:"transcript_committed",to:"text"},ports:[
    {id:"in",direction:"input",value_type:"transcript_committed",cardinality:"one"},
    {id:"out",direction:"output",value_type:"text",cardinality:"many"}],replacement:replacement("to_text")},
  transcript_sink:{kind:"transcript_sink",label:"Transcript",requires_component:false,ports:[
    {id:"in",direction:"input",value_type:"transcript_committed",cardinality:"one"}],replacement:replacement("transcript_sink")},
},components:{
  fixture:{id:"fixture",node_kind:"asr",provider:"fixture",model:"contract-v1",readiness:"ready",capabilities:["asr"],configuration_schema:asrSchema,default_config:{language:"en",timestamps:true,beam:4,tags:["speech"],options:{punctuate:true}},replacement:replacement("asr")},
  fixture_alt:{id:"fixture-alt",node_kind:"asr",provider:"fixture-two",model:"contract-v2",readiness:"ready",capabilities:["asr"],configuration_schema:asrSchema,default_config:{language:"fr",timestamps:false,beam:2,tags:[],options:{punctuate:false}},detail:"Ready alternate ASR provider",replacement:replacement("asr")},
  fixture_unavailable:{id:"fixture-unavailable",node_kind:"asr",provider:"fixture-two",model:"missing",readiness:"unavailable",capabilities:["asr"],configuration_schema:asrSchema,default_config:{language:"en"},detail:"Model files are absent",replacement:replacement("asr")},
  fixture_lookalike:{id:"fixture-lookalike",node_kind:"asr",provider:"lookalike",model:"not-semantic",readiness:"ready",capabilities:["asr"],configuration_schema:asrSchema,default_config:{language:"en"},replacement:replacement("not_asr")},
  fixture_cross:{id:"fixture-cross",node_kind:"asr_equivalent",provider:"fixture-three",model:"declared-equivalent",readiness:"ready",capabilities:["asr"],configuration_schema:{type:"object",properties:{locale:{type:"string",enum:["en","fr"]},timestamps:{type:"boolean"},required_text:{type:"string",minLength:1}},required:["locale","required_text"]},default_config:{},replacement:replacement("asr",{port_aliases:{audio:"samples",committed:"text"},configuration_aliases:{language:"locale"}})},
  fixture_missing:{id:"fixture-missing",node_kind:"asr_missing_output",provider:"fixture-three",model:"missing-port",readiness:"ready",capabilities:["asr"],configuration_schema:asrSchema,default_config:{language:"en"},replacement:replacement("asr")},
}};
const browserSource=fs.readFileSync(new URL("./speech-dataflow.js",import.meta.url),"utf8");
const browserHtml=fs.readFileSync(new URL("./speech-dataflow.html",import.meta.url),"utf8");

test("catalog groups inventory and ports derived from backend discovery",()=>{
  const catalog=buildCatalog(discovery);
  assert.equal(catalog.find(node=>node.id==="component:fixture").group,"Recognition");
  assert.equal(catalog.find(node=>node.id==="kind:microphone").ports[0].value_type,"audio_stream");
  assert.ok(!catalog.some(node=>node.label.includes("Whisper")));
});

test("componentless starter nodes resolve catalog configuration when component_id is omitted",()=>{
  const catalog=buildCatalog(discovery);
  const starterNode={id:"text",kind:"text_source",config:{}};
  const item=catalogEntryForNode(starterNode,catalog);
  assert.equal(item.readiness,"ready");
  assert.deepEqual(item.schema.properties.text,{
    type:"string",title:"Text",format:"multiline",minLength:1,
  });
  assert.equal(nodeLabel(starterNode,catalog),"Inline text");
});

test("explicit ports support fan-out but reject implicit replacement of a one-cardinality input",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const first=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const second=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  connectPorts(graph,first.id,"out",asr.id,"audio",discovery);
  assert.throws(()=>connectPorts(graph,second.id,"out",asr.id,"audio",discovery),/already occupied.*merge.*reconnect/);
  assert.equal(graph.edges[0].from.node_id,first.id);
  connectPorts(graph,asr.id,"committed",sink.id,"in",discovery);
  const sink2=duplicateNode(graph,sink.id);
  connectPorts(graph,asr.id,"committed",sink2.id,"in",discovery);
  assert.equal(graph.edges.filter(edge=>edge.to.node_id===asr.id).length,1);
  assert.equal(graph.edges.filter(edge=>edge.from.node_id===asr.id).length,2);
  assert.equal(compatibleTargets(graph,asr.id,"committed",discovery).length,0);
});

test("incompatible ports include an actionable adapter path",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  assert.throws(()=>connectPorts(graph,mic.id,"out",sink.id,"in",discovery),/audio_stream.*transcript_committed.*No registered adapter/);
  assert.deepEqual(adapterPaths("transcript_committed","text",discovery),[{kind:"to_text",label:"Transcript to text"}]);
});

test("cable reconnection preserves identity and capacity atomically",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const first=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const second=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const edge=connectPorts(graph,first.id,"out",asr.id,"audio",discovery);
  edge.capacity=41;
  const revision=graph.revision;
  reconnectEdge(graph,edge.id,"from",second.id,"out",discovery);
  assert.deepEqual(edge.from,{node_id:second.id,port_id:"out"});
  assert.equal(edge.id,graph.edges[0].id);
  assert.equal(edge.capacity,41);
  assert.equal(graph.revision,revision+1);
});

test("failed cable reconnection leaves serialized graph unchanged",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  const edge=connectPorts(graph,mic.id,"out",asr.id,"audio",discovery);
  const before=JSON.stringify(graph);
  assert.throws(()=>reconnectEdge(graph,edge.id,"to",sink.id,"in",discovery),/audio_stream.*transcript_committed/);
  assert.equal(JSON.stringify(graph),before);
});

test("compatibility explains occupied inputs and explicit adapter paths",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const first=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const second=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  connectPorts(graph,first.id,"out",asr.id,"audio",discovery);
  const occupied=connectionCompatibility(graph,second.id,"out",asr.id,"audio",discovery);
  const mismatch=connectionCompatibility(graph,asr.id,"committed",sink.id,"in",discovery);
  assert.equal(occupied.code,"connection.input_occupied");
  assert.match(occupied.reason,/explicit merge node/);
  assert.equal(mismatch.compatible,true);
  const adapter=connectionCompatibility(graph,asr.id,"committed",addNode(graph,catalog.find(node=>node.kind==="text_source")).id,"out",discovery);
  assert.equal(adapter.code,"connection.direction");
});

test("diagnostics attach to missing required ports and expose accessible detail",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const report={diagnostics:[{code:"missing_required_input",target:{node_id:asr.id,port_id:"audio"},message:"audio needs one audio stream"}]};
  assert.deepEqual(diagnosticsByTarget(report).ports[`${asr.id}:audio`][0].code,"missing_required_input");
  assert.equal(requiredPortState(graph,asr,{id:"audio",direction:"input",cardinality:"one"},report).missing,true);
});

test("layout and reusable subgraph insertion survive JSON persistence",()=>{
  const graph=createPipeline(),template=createPipeline("Template"),catalog=buildCatalog(discovery);
  addNode(template,catalog.find(node=>node.kind==="microphone"));
  addNode(template,catalog.find(node=>node.kind==="asr"));
  connectPorts(template,template.nodes[0].id,"out",template.nodes[1].id,"audio",discovery);
  const ids=insertSubgraph(graph,template,{x:140,y:90});
  assert.equal(ids.length,2);assert.equal(graph.edges.length,1);
  assert.deepEqual(Object.keys(ensureLayout(JSON.parse(JSON.stringify(graph)))).sort(),ids.sort());
});

test("copy and paste mint fresh identities and retain only selected internal topology",()=>{
  const graph=createPipeline(),catalog=buildCatalog(discovery);
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"),null,{x:100,y:100});
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"),null,{x:400,y:100});
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"),null,{x:700,y:100});
  connectPorts(graph,mic.id,"out",asr.id,"audio",discovery);
  connectPorts(graph,asr.id,"committed",sink.id,"in",discovery);
  const copied=copyGraphSelection(graph,[mic.id,asr.id]);
  assert.deepEqual(copied.nodes.map(node=>node.id),[mic.id,asr.id]);
  assert.equal(copied.edges.length,1);assert.equal(copied.edges[0].to.node_id,asr.id);
  const pasted=pasteGraphSelection(graph,copied,{x:50,y:75});
  assert.equal(pasted.length,2);assert.ok(pasted.every(id=>![mic.id,asr.id].includes(id)));
  const pastedEdge=graph.edges.at(-1);
  assert.ok(pasted.includes(pastedEdge.from.node_id));assert.ok(pasted.includes(pastedEdge.to.node_id));
  assert.notEqual(pastedEdge.id,copied.edges[0].id);
  assert.equal(graph.edges.filter(edge=>edge.to.node_id===sink.id).length,1,"external topology is not fabricated");
  const layout=ensureLayout(graph);
  assert.deepEqual(layout[pasted[0]],{x:150,y:175});assert.deepEqual(layout[pasted[1]],{x:450,y:175});
});

test("multi-object delete and presentation arrangement are atomic and semantics-safe",()=>{
  const graph=createPipeline(),catalog=buildCatalog(discovery);
  const nodes=[
    addNode(graph,catalog.find(node=>node.kind==="microphone"),null,{x:100,y:100}),
    addNode(graph,catalog.find(node=>node.kind==="asr"),null,{x:340,y:180}),
    addNode(graph,catalog.find(node=>node.kind==="transcript_sink"),null,{x:760,y:360}),
  ];
  connectPorts(graph,nodes[0].id,"out",nodes[1].id,"audio",discovery);
  connectPorts(graph,nodes[1].id,"committed",nodes[2].id,"in",discovery);
  const semantic=()=>graph.edges.map(edge=>structuredClone({id:edge.id,from:edge.from,to:edge.to,capacity:edge.capacity}));
  const before=semantic(),revision=graph.revision;
  assert.equal(moveGraphSelection(graph,nodes.map(node=>node.id),{x:13,y:29},{snap:24}),true);
  assert.equal(alignGraphSelection(graph,nodes.map(node=>node.id),"y","start"),true);
  assert.equal(distributeGraphSelection(graph,nodes.map(node=>node.id),"x"),true);
  assert.equal(tidyGraphSelection(graph,nodes.map(node=>node.id),{columns:2}),true);
  assert.deepEqual(semantic(),before);assert.equal(graph.revision,revision+4);
  const edgeId=graph.edges[0].id;
  assert.equal(deleteGraphSelection(graph,[nodes[1].id],[edgeId]),true);
  assert.deepEqual(graph.nodes.map(node=>node.id),[nodes[0].id,nodes[2].id]);
  assert.equal(graph.edges.length,0);
});

test("bypass is explicit structural rewiring and preserves compatible fan-out",()=>{
  const graph=createPipeline(),catalog=buildCatalog(discovery);
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const adapter=addNode(graph,catalog.find(node=>node.kind==="to_text"));
  connectPorts(graph,mic.id,"out",asr.id,"audio",discovery);
  connectPorts(graph,asr.id,"committed",adapter.id,"in",discovery);
  assert.throws(()=>bypassNode(graph,asr.id,discovery),/does not satisfy/);
});

test("persisted reload preserves graph meaning and backend-owned identities",()=>{
  const graph=createPipeline("Saved"),catalog=buildCatalog(discovery);
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  connectPorts(graph,mic.id,"out",asr.id,"audio",discovery);
  const reopened=JSON.parse(JSON.stringify(graph));
  assert.deepEqual(reopened,graph);
  assert.equal(reopened.nodes[1].component_id,"fixture");
});

test("text source configuration survives duplication and JSON persistence",()=>{
  const graph=createPipeline("Text"),catalog=buildCatalog(discovery);
  const source=addNode(graph,catalog.find(node=>node.kind==="text_source"));
  source.config.text="First line.\nSecond line.";
  graph.edges.push({id:"edge:keep",from:{node_id:source.id,port_id:"out"},to:{node_id:"downstream",port_id:"in"},capacity:16});
  const copy=duplicateNode(graph,source.id);
  const reopened=JSON.parse(JSON.stringify(graph));
  assert.equal(copy.config.text,"First line.\nSecond line.");
  assert.equal(reopened.nodes[0].config.text,"First line.\nSecond line.");
  assert.equal(reopened.edges[0].id,"edge:keep");
  assert.notEqual(copy.config,source.config);
});

test("applying text source configuration preserves graph connections",()=>{
  const graph=createPipeline("Text"),catalog=buildCatalog(discovery);
  const source=addNode(graph,catalog.find(node=>node.kind==="text_source"));
  graph.edges.push({id:"edge:keep",from:{node_id:source.id,port_id:"out"},to:{node_id:"downstream",port_id:"in"},capacity:16});
  applyNodeConfig(graph,source.id,{text:"Updated\nmultiline text"});
  assert.equal(source.config.text,"Updated\nmultiline text");
  assert.deepEqual(graph.edges.map(edge=>edge.id),["edge:keep"]);
});

function replacementGraph(){
  const catalog=buildCatalog(discovery),graph=createPipeline("Replacement fixture");
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.id==="component:fixture"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  asr.config={language:"en",timestamps:true,beam:7,notes:"keep me",tags:["speech"],options:{punctuate:true}};
  connectPorts(graph,mic.id,"out",asr.id,"audio",discovery);connectPorts(graph,asr.id,"committed",sink.id,"in",discovery);
  graph.selected_sinks=[{node_id:asr.id,port_id:"committed"}];ensureLayout(graph);
  return{catalog,graph,mic,asr,sink};
}

test("replacement candidates are backend-family classified, readiness-aware, and deterministically ranked",()=>{
  const{catalog,graph,asr}=replacementGraph(),candidates=replacementCandidates(graph,asr.id,catalog);
  assert.equal(candidates[0].id,"component:fixture-alt");
  assert.equal(candidates[0].compatibility,"exact_drop_in");
  assert.equal(candidates.find(item=>item.id==="component:fixture-unavailable").code,"replacement.not_ready");
  assert.equal(candidates.find(item=>item.id==="component:fixture-cross").compatibility,"migration");
  assert.equal(candidates.find(item=>item.id==="component:fixture-lookalike").code,"replacement.family_mismatch");
  assert.equal(candidates.find(item=>item.id==="component:fixture-missing").code,"replacement.connected_port_missing");
  assert.ok(candidates.every(item=>item.reason&&item.provider&&item.model));
});

test("configuration migration validates scalar, enum, array, object, alias, default, removal, and required-input states",()=>{
  assert.deepEqual(validateSchemaValue(9,{type:"integer",maximum:8}),["$ must be at most 8."]);
  assert.equal(validateSchemaValue(["a"],{type:"array",items:{type:"string"}}).length,0);
  assert.equal(validateSchemaValue({extra:true},{type:"object",additionalProperties:false})[0],"$.extra is not allowed.");
  const{catalog,graph,asr}=replacementGraph(),current=catalogEntryForNode(asr,catalog);
  const exact=catalog.find(item=>item.id==="component:fixture-alt");
  const preserved=migrateReplacementConfig(asr.config,current,exact);
  assert.deepEqual(preserved.config,asr.config);
  assert.ok(preserved.changes.every(change=>change.state==="preserved"));
  const defaults=migrateReplacementConfig(asr.config,current,exact,{useDefaults:true});
  assert.equal(defaults.config.language,"fr");assert.ok(defaults.changes.some(change=>change.state==="defaulted"));
  const cross=catalog.find(item=>item.id==="component:fixture-cross");
  const mapped=migrateReplacementConfig(asr.config,current,cross);
  assert.equal(mapped.config.locale,"en");assert.equal(mapped.config.timestamps,true);
  assert.ok(mapped.changes.some(change=>change.field==="locale"&&change.state==="mapped"));
  assert.ok(mapped.changes.some(change=>change.field==="beam"&&change.state==="removed"));
  assert.ok(mapped.blocking.some(item=>item.field==="required_text"));
});

test("exact replacement plans are preview-only, preserve graph identity and wiring, and apply as one undoable edit",()=>{
  const{catalog,graph,asr}=replacementGraph(),before=JSON.stringify(graph),position=ensureLayout(graph)[asr.id];
  const candidate=catalog.find(item=>item.id==="component:fixture-alt");
  let plan=planNodeReplacement(graph,asr.id,candidate,catalog,{catalogRevision:"fixture-revision"});
  assert.equal(JSON.stringify(graph),before,"planning and cancel must not mutate the graph");
  assert.ok(plan.edge_changes.every(change=>change.state==="preserved"));
  assert.ok(plan.sink_changes.every(change=>change.state==="preserved"));
  plan=attachReplacementValidation(plan,{valid:true,diagnostics:[]});assert.equal(plan.applyable,true);
  const history=createEditHistory(),applied=commitReplacement(graph,plan,"fixture-revision",history,asr.id);
  const replaced=applied.nodes.find(node=>node.id===asr.id);
  assert.equal(applied.revision,graph.revision+1);assert.equal(replaced.component_id,"fixture-alt");
  assert.deepEqual(replaced.config,asr.config);assert.deepEqual(ensureLayout(applied)[asr.id],position);
  assert.deepEqual(applied.edges,graph.edges);assert.deepEqual(applied.selected_sinks,graph.selected_sinks);
  const undone=undoEdit(history);assert.deepEqual(undone.pipeline,graph);assert.equal(undone.selection,asr.id);
  const redone=redoEdit(history);assert.deepEqual(redone.pipeline,applied);assert.equal(redone.selection,asr.id);
});

test("declared cross-kind plans name port/config remaps and reject stale or failed previews atomically",()=>{
  const{catalog,graph,asr}=replacementGraph(),candidate=catalog.find(item=>item.id==="component:fixture-cross");
  let plan=planNodeReplacement(graph,asr.id,candidate,catalog,{catalogRevision:"fixture-revision",overrides:{required_text:"ready"}});
  assert.ok(plan.edge_changes.some(change=>change.from==="audio"&&change.to==="samples"));
  assert.ok(plan.edge_changes.some(change=>change.from==="committed"&&change.to==="text"));
  assert.ok(plan.config_changes.some(change=>change.field==="locale"&&change.state==="mapped"));
  const failed=attachReplacementValidation(plan,{valid:false,diagnostics:[{message:"fixture backend rejection"}]});
  assert.equal(failed.applyable,false);assert.throws(()=>commitReplacement(graph,failed,"fixture-revision",createEditHistory(),asr.id),/not applyable/);
  plan=attachReplacementValidation(plan,{valid:true,diagnostics:[]});
  const changed=structuredClone(graph);changed.metadata.description="concurrent edit";
  assert.throws(()=>commitReplacement(changed,plan,"fixture-revision",createEditHistory(),asr.id),/graph changed/i);
  assert.throws(()=>commitReplacement(graph,plan,"new-catalog-revision",createEditHistory(),asr.id),/discovery changed/i);
  const applied=commitReplacement(graph,plan,"fixture-revision",createEditHistory(),asr.id);
  assert.equal(applied.nodes.find(node=>node.id===asr.id).kind,"asr_equivalent");
  assert.deepEqual(applied.selected_sinks,[{node_id:asr.id,port_id:"text"}]);
});

test("general edit history restores graph, selection, focus, and only clears the invalid redo branch",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline(),history=createEditHistory();
  const before=structuredClone(graph),mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  assert.equal(recordEdit(history,before,graph,{
    label:"Add node",selectionBefore:{node_id:null,edge_id:null},selectionAfter:{node_id:mic.id,edge_id:null},
    focusBefore:{id:"canvas"},focusAfter:{id:"delete"},
  }),true);
  const afterAdd=structuredClone(graph),asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  recordEdit(history,afterAdd,graph,{label:"Add ASR",selectionAfter:{node_id:asr.id,edge_id:null}});
  const undone=undoEdit(history);
  assert.deepEqual(undone.pipeline,afterAdd);assert.equal(undone.label,"Add ASR");
  assert.deepEqual(undone.selection,null);assert.equal(history.redo.length,1);
  const branchBefore=structuredClone(undone.pipeline);
  const branch=addNode(undone.pipeline,catalog.find(node=>node.kind==="transcript_sink"));
  recordEdit(history,branchBefore,undone.pipeline,{label:"Branch edit",selectionAfter:{node_id:branch.id,edge_id:null}});
  assert.equal(history.redo.length,0);
  assert.deepEqual(undoEdit(history).pipeline,branchBefore);
  const firstUndo=undoEdit(history);
  assert.deepEqual(firstUndo.pipeline,before);assert.deepEqual(firstUndo.selection,{node_id:null,edge_id:null});
  assert.deepEqual(firstUndo.focus,{id:"canvas"});
});

test("recording a byte-identical graph is ignored",()=>{
  const graph=createPipeline(),history=createEditHistory();
  assert.equal(recordEdit(history,graph,structuredClone(graph),{label:"No-op"}),false);
  assert.deepEqual(history,{undo:[],redo:[]});
});

test("large replacement catalogs classify within a generous interaction budget",()=>{
  const{catalog,graph,asr}=replacementGraph(),template=catalog.find(item=>item.id==="component:fixture-alt");
  for(let index=0;index<2000;index++)catalog.push({...structuredClone(template),id:`component:bulk-${index}`,component_id:`bulk-${index}`,model:`model-${index}`});
  const started=performance.now(),candidates=replacementCandidates(graph,asr.id,catalog),elapsed=performance.now()-started;
  assert.ok(candidates.length>=2000);assert.ok(elapsed<1500,`classification took ${elapsed.toFixed(1)} ms`);
});

test("streamed run lifecycle survives arbitrary response chunk boundaries",async()=>{
  const encoder=new TextEncoder(),chunks=[
    encoder.encode('{"node_id":"mic","kind":"started"}\n{"node'),
    encoder.encode('_id":"mic","kind":"output"}\n'),
    encoder.encode('{"node_id":"mic","kind":"completed"}'),
  ],events=[];
  const reader={async read(){return chunks.length?{done:false,value:chunks.shift()}:{done:true};}};
  await consumeNdjson(reader,event=>events.push(event));
  assert.deepEqual(events.map(event=>event.kind),["started","output","completed"]);
});

test("browser workflow wires persistence, streamed execution, cancellation, and accessibility",()=>{
  assert.match(browserHtml,/cytoscape@3\.33\.1/);
  assert.match(browserHtml,/role="application"/);
  assert.match(browserHtml,/id="graph-outline".*aria-label="Keyboard graph outline"/);
  assert.match(browserSource,/\/api\/pipeline\/graphs\/\$\{encodeURIComponent/);
  assert.match(browserSource,/fetch\("\/api\/pipeline\/run"/);
  assert.match(browserSource,/new AbortController\(\)/);
  assert.match(browserSource,/runController\?\.abort\(\)/);
  assert.match(browserSource,/item\.suggestions/);
  assert.match(browserSource,/spec\.format==="multiline".*"textarea"/);
  assert.match(browserSource,/input\.checkValidity\(\)/);
  assert.match(browserSource,/event\.output.*event\.output\.port_id/);
  assert.match(browserHtml,/id="replacement-dialog"/);
  assert.match(browserHtml,/role="listbox"/);
  assert.match(browserHtml,/id="replacement-results".*aria-live="polite"/);
  assert.match(browserSource,/replacementCandidates\(pipeline,node\.id,catalog\)/);
  assert.match(browserSource,/\/api\/pipeline\/validate/);
  assert.match(browserSource,/commitReplacement\(/);
  assert.match(browserSource,/replacementRenderLimit=100/);
  assert.doesNotMatch(browserSource,/\bprompt\s*\(|\bconfirm\s*\(|\balert\s*\(/);
  assert.doesNotMatch(browserSource,/Whisper|FastPitch|OpenAI|Anthropic/);
});

test("canvas nodes use readable cards and humanized port types",()=>{
  assert.match(browserSource,/"width":228,"height":126/);
  assert.match(browserSource,/"font-size":14/);
  assert.match(browserSource,/"text-justification":"left"/);
  assert.match(browserSource,/transcript_committed:"committed transcript"/);
  assert.match(browserSource,/\["error","cancellation"\]/);
  assert.match(browserSource,/NODE_THEMES\[item\?\.group\]/);
});
