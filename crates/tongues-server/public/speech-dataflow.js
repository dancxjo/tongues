import {
  addNode,buildCatalog,connect,createPipeline,duplicateNode,moveNode,removeNode,replaceNode,
  template,toggleBypass,validatePipeline,
} from "./speech-dataflow-model.mjs";

const byId = id => document.getElementById(id);
let catalog = [], pipeline = createPipeline(), selected = null, runEvents = [];
const STORAGE_KEY = "tongues.speech.dataflow.v1";

function announce(text, error=false) { byId("status").textContent=text; byId("status").classList.toggle("error",error); }

async function discover() {
  const get = async path => { const response=await fetch(path); if(!response.ok) throw new Error(`${path}: ${await response.text()}`); return response.json(); };
  const [audio,asr,language,live,speech,cli] = await Promise.all([
    get("/api/audio-input/capabilities"),get("/api/asr/capabilities"),get("/api/language-routing/capabilities"),
    get("/api/live/providers"),get("/api/speech/models"),get("/api/cli/schema"),
  ]);
  catalog=buildCatalog({audio,asr,language,live,speech,cli});
  renderCatalog(); loadTemplate("transcription");
  announce(`Loaded ${catalog.length} backend-derived nodes.`);
}

function renderCatalog() {
  const select=byId("catalog"); select.replaceChildren(...catalog.map(node=>new Option(`${node.kind} · ${node.label}`,node.id)));
  const replacement=byId("replacement"); replacement.replaceChildren(...catalog.map(node=>new Option(`${node.kind} · ${node.label}`,node.id)));
}

function render() {
  byId("pipeline-name").value=pipeline.name;
  const validation=validatePipeline(pipeline);
  byId("validation").textContent=validation.valid?"Ready to execute":validation.errors.join(" ");
  byId("validation").dataset.state=validation.valid?"valid":"invalid";
  byId("nodes").replaceChildren(...pipeline.nodes.map((node,index)=>{
    const item=document.createElement("li"); item.tabIndex=0; item.dataset.nodeId=node.instance_id;
    item.className=`node${node.bypassed?" bypassed":""}${selected===node.instance_id?" selected":""}`;
    const incoming=pipeline.edges.find(edge=>edge.to===node.instance_id);
    item.innerHTML=`<span class="order">${index+1}</span><div><strong>${escapeHtml(node.label)}</strong>
      <small>${escapeHtml(node.kind)} · ${escapeHtml(node.capability_id)}</small>
      ${incoming?`<small>← ${escapeHtml(pipeline.nodes.find(n=>n.instance_id===incoming.from)?.label??"missing")}</small>`:""}</div>
      <span class="badge">${node.bypassed?"bypassed":"active"}</span>`;
    item.onclick=()=>{selected=node.instance_id;render();announce(`Selected ${node.label}.`);};
    item.onkeydown=event=>{if(event.key==="ArrowUp"){event.preventDefault();moveNode(pipeline,node.instance_id,-1);autoConnect();render();}
      if(event.key==="ArrowDown"){event.preventDefault();moveNode(pipeline,node.instance_id,1);autoConnect();render();}};
    return item;
  }));
  for (const id of ["connect-from","connect-to"]) {
    const select=byId(id), previous=select.value;
    select.replaceChildren(...pipeline.nodes.map(node=>new Option(node.label,node.instance_id)));
    if ([...select.options].some(option=>option.value===previous)) select.value=previous;
  }
  const selectedNode=pipeline.nodes.find(node=>node.instance_id===selected);
  byId("node-config").value=selectedNode?JSON.stringify(selectedNode.config,null,2):"";
  byId("open-timeline").href=`/wavedeck.html#pipeline=${encodeURIComponent(pipeline.id)}`;
}

function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function catalogNode(id){return catalog.find(node=>node.id===id);}

function autoConnect() {
  pipeline.edges=[];
  const active=pipeline.nodes.filter(node=>!node.bypassed);
  for(let index=1;index<active.length;index++){try{connect(pipeline,active[index-1].instance_id,active[index].instance_id);}catch(_){}}
}

function loadTemplate(name){pipeline=template(name,catalog);selected=pipeline.nodes[0]?.instance_id??null;render();}
function mutate(operation){if(!selected)return announce("Select a node first.",true);try{operation();autoConnect();render();}catch(error){announce(error.message,true);}}

function save() { pipeline.name=byId("pipeline-name").value.trim()||"Untitled pipeline";localStorage.setItem(STORAGE_KEY,JSON.stringify(pipeline));announce("Saved versioned pipeline locally; capabilities resolve again when opened.");}
function restore() {try{const value=JSON.parse(localStorage.getItem(STORAGE_KEY));if(value.schema_version!==1)throw new Error(`schema ${value.schema_version} needs migration to 1`);pipeline=value;selected=pipeline.nodes[0]?.instance_id;render();announce("Saved pipeline restored against current backend catalog.");}catch(error){announce(`Cannot restore: ${error.message}`,true);}}
function share(){const blob=new Blob([JSON.stringify(pipeline,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`${pipeline.id.replaceAll(":","-")}.json`;link.click();URL.revokeObjectURL(link.href);}

async function runFixture() {
  const validation=validatePipeline(pipeline);if(!validation.valid)return announce(`Cannot run: ${validation.errors.join(" ")}`,true);
  runEvents=[];renderRun();const started=performance.now();byId("level").value=.18;
  setTelemetry(["speech started","pending","pending","pending","audio→ASR pending","none"]);
  const wav=new ArrayBuffer(44+32000);const view=new DataView(wav);writeWavHeader(view,16000);
  const response=await fetch("/api/asr/transcriptions?provider=fixture&language=en",{method:"POST",headers:{"Content-Type":"audio/wav"},body:wav});
  if(!response.ok)return announce(await response.text(),true);
  const result=await response.json();runEvents=result.events;renderRun();
  const first=runEvents.find(event=>event.type==="partial_hypothesis");
  const committed=runEvents.find(event=>event.type==="committed_segment");
  setTelemetry(["speech ended",first?.data?.text??"—",committed?.data?.text??"—",
    `${committed?.data?.language?.language??"unknown"} / ${committed?.data?.speaker_id??"unknown"}`,
    `ASR ${(performance.now()-started).toFixed(1)} ms`,"none"]);
  byId("level").value=0;announce("Deterministic pipeline run completed. Open its session in Timeline for corrections.");
}

function writeWavHeader(v,rate){const text=(o,s)=>[...s].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));text(0,"RIFF");v.setUint32(4,v.byteLength-8,true);text(8,"WAVEfmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,"data");v.setUint32(40,v.byteLength-44,true);}
function setTelemetry(values){[...byId("telemetry").querySelectorAll("dd")].forEach((node,index)=>node.textContent=values[index]??"—");}
function renderRun(){byId("run-events").replaceChildren(...runEvents.map((event,index)=>{const item=document.createElement("li");item.className=event.type==="committed_segment"?"committed":"partial";const confidence=event.data?.confidence?.value;item.textContent=`#${index} ${event.type}: ${event.data?.text??event.data?.reason??""}${confidence==null?"":` · confidence ${confidence}`}`;return item;}));}

byId("add").onclick=()=>{const node=addNode(pipeline,catalogNode(byId("catalog").value),selected);selected=node.instance_id;autoConnect();render();};
byId("remove").onclick=()=>mutate(()=>{removeNode(pipeline,selected);selected=null;});
byId("duplicate").onclick=()=>mutate(()=>{selected=duplicateNode(pipeline,selected).instance_id;});
byId("bypass").onclick=()=>mutate(()=>toggleBypass(pipeline,selected));
byId("replace").onclick=()=>mutate(()=>replaceNode(pipeline,selected,catalogNode(byId("replacement").value)));
byId("configure").onclick=()=>mutate(()=>{pipeline.nodes.find(node=>node.instance_id===selected).config=JSON.parse(byId("node-config").value);pipeline.revision++;});
byId("connect").onclick=()=>{try{connect(pipeline,byId("connect-from").value,byId("connect-to").value);render();announce("Typed connection accepted.");}catch(error){announce(error.message,true);}};
byId("up").onclick=()=>mutate(()=>moveNode(pipeline,selected,-1));byId("down").onclick=()=>mutate(()=>moveNode(pipeline,selected,1));
byId("template").onchange=event=>loadTemplate(event.target.value);byId("save").onclick=save;byId("restore").onclick=restore;byId("share").onclick=share;byId("run").onclick=()=>runFixture().catch(error=>announce(error.message,true));
discover().catch(error=>announce(`Discovery failed: ${error.message}`,true));
