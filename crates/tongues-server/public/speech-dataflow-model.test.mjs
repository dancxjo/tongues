import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  adapterPaths,addNode,buildCatalog,bypassNode,compatibleTargets,connectPorts,consumeNdjson,createPipeline,
  diagnosticsByTarget,duplicateNode,ensureLayout,insertSubgraph,requiredPortState,
} from "./speech-dataflow-model.mjs";

const discovery={node_kinds:{
  microphone:{kind:"microphone",label:"Microphone",requires_component:false,default_config:{},ports:[
    {id:"out",direction:"output",value_type:"audio_stream",cardinality:"many"}]},
  asr:{kind:"asr",label:"ASR",requires_component:true,required_capabilities:["asr"],ports:[
    {id:"audio",direction:"input",value_type:"audio_stream",cardinality:"one"},
    {id:"committed",direction:"output",value_type:"transcript_committed",cardinality:"many"}]},
  to_text:{kind:"to_text",label:"Transcript to text",requires_component:false,adapter:{from:"transcript_committed",to:"text"},ports:[
    {id:"in",direction:"input",value_type:"transcript_committed",cardinality:"one"},
    {id:"out",direction:"output",value_type:"text",cardinality:"many"}]},
  transcript_sink:{kind:"transcript_sink",label:"Transcript",requires_component:false,ports:[
    {id:"in",direction:"input",value_type:"transcript_committed",cardinality:"one"}]},
},components:{fixture:{id:"fixture",node_kind:"asr",provider:"fixture",model:"contract-v1",readiness:"ready",default_config:{}}}};
const browserSource=fs.readFileSync(new URL("./speech-dataflow.js",import.meta.url),"utf8");
const browserHtml=fs.readFileSync(new URL("./speech-dataflow.html",import.meta.url),"utf8");

test("catalog groups inventory and ports derived from backend discovery",()=>{
  const catalog=buildCatalog(discovery);
  assert.equal(catalog.find(node=>node.id==="component:fixture").group,"Recognition");
  assert.equal(catalog.find(node=>node.id==="kind:microphone").ports[0].value_type,"audio_stream");
  assert.ok(!catalog.some(node=>node.label.includes("Whisper")));
});

test("explicit ports support fan-out but replace a one-cardinality input",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const first=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const second=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  connectPorts(graph,first.id,"out",asr.id,"audio",discovery);
  connectPorts(graph,second.id,"out",asr.id,"audio",discovery);
  connectPorts(graph,asr.id,"committed",sink.id,"in",discovery);
  const sink2=duplicateNode(graph,sink.id);
  connectPorts(graph,asr.id,"committed",sink2.id,"in",discovery);
  assert.equal(graph.edges.filter(edge=>edge.to.node_id===asr.id).length,1);
  assert.equal(graph.edges.filter(edge=>edge.from.node_id===asr.id).length,2);
  assert.equal(compatibleTargets(graph,asr.id,"committed",discovery).length,2);
});

test("incompatible ports include an actionable adapter path",()=>{
  const catalog=buildCatalog(discovery),graph=createPipeline();
  const mic=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  assert.throws(()=>connectPorts(graph,mic.id,"out",sink.id,"in",discovery),/audio_stream.*transcript_committed.*No registered adapter/);
  assert.deepEqual(adapterPaths("transcript_committed","text",discovery),[{kind:"to_text",label:"Transcript to text"}]);
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
  assert.doesNotMatch(browserSource,/Whisper|FastPitch|OpenAI|Anthropic/);
});
