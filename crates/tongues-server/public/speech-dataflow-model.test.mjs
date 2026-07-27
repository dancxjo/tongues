import test from "node:test";
import assert from "node:assert/strict";
import {addNode,buildCatalog,connect,createPipeline,duplicateNode,template,toggleBypass,validatePipeline} from "./speech-dataflow-model.mjs";

const discovery = {
  audio:{source_kinds:["microphone","file"],cleanup_stages:[{kind:"noise_gate"}]},
  asr:{providers:[{provider_id:"fixture",installed:true}]},
  language:{detectors:[{detector_id:"lid",installed:true}]},
  live:{providers:[{id:"deterministic",label:"Deterministic",available:true}]},
  speech:{compositions:[{id:"tts:fixture",display_name:"Fixture TTS"}]},
  cli:{commands:[{id:"sentence-parser/parse",subcommands:[]},{id:"interpretation/interpret",subcommands:[]},{id:"normalize",subcommands:[]}]},
};

test("catalog inventory is derived from backend payloads", () => {
  const catalog = buildCatalog(discovery);
  assert.ok(catalog.some(node => node.catalog_id === undefined && node.id === "asr:fixture"));
  assert.ok(catalog.some(node => node.capability_id === "sentence-parser/parse"));
  assert.ok(!catalog.some(node => node.label.includes("Whisper")));
});

test("typed invalid connections are explained before execution", () => {
  const catalog = buildCatalog(discovery), pipeline = createPipeline();
  const source = addNode(pipeline,catalog.find(node => node.kind === "source"));
  const parser = addNode(pipeline,catalog.find(node => node.kind === "parser"));
  assert.throws(() => connect(pipeline,source.instance_id,parser.instance_id),/emits audio.*requires committed_text/);
});

test("starter templates validate and mutations remain serializable", () => {
  const catalog = buildCatalog(discovery);
  for (const name of ["transcription","multilingual_transcription","meeting_transcript","spoken_interpretation","full_conversation"]) {
    const pipeline = template(name,catalog);
    assert.equal(validatePipeline(pipeline).valid,true,`${name}: ${validatePipeline(pipeline).errors}`);
    const copy = duplicateNode(pipeline,pipeline.nodes[0].instance_id);
    toggleBypass(pipeline,copy.instance_id);
    assert.doesNotThrow(() => JSON.parse(JSON.stringify(pipeline)));
  }
});
