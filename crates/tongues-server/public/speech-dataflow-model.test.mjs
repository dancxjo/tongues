import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  adapterPaths,addNode,applyNodeConfig,buildCatalog,bypassNode,compatibleTargets,connectPorts,consumeNdjson,createPipeline,
  catalogEntryForNode,diagnosticsByTarget,duplicateNode,ensureLayout,insertSubgraph,nodeLabel,requiredPortState,
} from "./speech-dataflow-model.mjs";

const discovery={node_kinds:{
  text_source:{kind:"text_source",label:"Inline text",requires_component:false,default_config:{text:"Hello from Tongues."},configuration_schema:{
    type:"object",properties:{text:{type:"string",title:"Text",format:"multiline",minLength:1}},required:["text"]},ports:[
    {id:"out",direction:"output",value_type:"text",cardinality:"many",streaming:true}]},
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
