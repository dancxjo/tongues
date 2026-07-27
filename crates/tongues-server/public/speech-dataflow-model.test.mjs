import test from "node:test";
import assert from "node:assert/strict";
import {addNode,buildCatalog,connect,createPipeline,duplicateNode} from "./speech-dataflow-model.mjs";

const discovery = {
  node_kinds:{
    microphone:{kind:"microphone",label:"Microphone",requires_component:false,default_config:{},ports:[
      {id:"out",direction:"output",value_type:"audio_stream",cardinality:"many"},
    ]},
    asr:{kind:"asr",label:"ASR",requires_component:true,ports:[
      {id:"audio",direction:"input",value_type:"audio_stream",cardinality:"one"},
      {id:"committed",direction:"output",value_type:"transcript_committed",cardinality:"many"},
    ]},
    transcript_sink:{kind:"transcript_sink",label:"Transcript",requires_component:false,ports:[
      {id:"in",direction:"input",value_type:"transcript_committed",cardinality:"one"},
    ]},
  },
  components:{
    fixture:{id:"fixture",node_kind:"asr",provider:"fixture",model:"contract-v1",readiness:"ready",default_config:{}},
  },
};

test("catalog inventory and ports come entirely from backend discovery", () => {
  const catalog=buildCatalog(discovery);
  assert.ok(catalog.some(node=>node.id==="component:fixture"));
  assert.ok(catalog.some(node=>node.id==="kind:microphone"));
  assert.ok(!catalog.some(node=>node.label.includes("Whisper")));
});

test("graph mutations preserve the backend document and endpoint contract", () => {
  const catalog=buildCatalog(discovery), graph=createPipeline();
  const source=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const asr=addNode(graph,catalog.find(node=>node.kind==="asr"));
  connect(graph,source.id,asr.id,discovery);
  assert.deepEqual(graph.edges[0].from,{node_id:source.id,port_id:"out"});
  assert.deepEqual(graph.edges[0].to,{node_id:asr.id,port_id:"audio"});
  const copy=duplicateNode(graph,source.id);
  assert.notEqual(copy.id,source.id);
  assert.doesNotThrow(()=>JSON.parse(JSON.stringify(graph)));
});

test("incompatible connections report discovered types", () => {
  const catalog=buildCatalog(discovery), graph=createPipeline();
  const source=addNode(graph,catalog.find(node=>node.kind==="microphone"));
  const sink=addNode(graph,catalog.find(node=>node.kind==="transcript_sink"));
  assert.throws(()=>connect(graph,source.id,sink.id,discovery),/audio_stream.*transcript_committed/);
});
