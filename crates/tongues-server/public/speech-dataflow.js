import {
  addNode,buildCatalog,connect,duplicateNode,moveNode,nodeLabel,removeNode,replaceNode,
} from "./speech-dataflow-model.mjs";

const byId = id => document.getElementById(id);
let discovery = null, catalog = [], starters = [], pipeline = null, selected = null, runEvents = [];
let validationGeneration = 0;
const STORAGE_KEY = "tongues.speech.dataflow.v2";

function announce(text, error=false) {
  byId("status").textContent=text;
  byId("status").classList.toggle("error",error);
}

async function request(path, options={}) {
  const response=await fetch(path,options);
  const value=await response.json().catch(async()=>({error:await response.text()}));
  if(!response.ok) throw new Error(value.error??value.validation?.diagnostics?.map(item=>item.message).join(" ")??`${path}: ${response.status}`);
  return value;
}

async function discover() {
  [discovery,{graphs:starters}]=await Promise.all([
    request("/api/pipeline/catalog"),request("/api/pipeline/starters"),
  ]);
  catalog=buildCatalog(discovery);
  renderCatalog();
  const selector=byId("template");
  selector.replaceChildren(...starters.map(graph=>new Option(graph.metadata.name,graph.graph_id)));
  loadTemplate(starters[0]?.graph_id);
  announce(`Loaded ${catalog.length} backend-owned node/component choices and ${starters.length} executable starters.`);
}

function renderCatalog() {
  const options=()=>catalog.map(node=>new Option(`${node.kind} · ${node.label}${node.readiness&&node.readiness!=="ready"?` · ${node.readiness}`:""}`,node.id));
  byId("catalog").replaceChildren(...options());
  byId("replacement").replaceChildren(...options());
}

function render() {
  if(!pipeline)return;
  byId("pipeline-name").value=pipeline.metadata.name;
  byId("nodes").replaceChildren(...pipeline.nodes.map((node,index)=>{
    const item=document.createElement("li"); item.tabIndex=0; item.dataset.nodeId=node.id;
    item.className=`node${selected===node.id?" selected":""}`;
    const incoming=pipeline.edges.filter(edge=>edge.to.node_id===node.id);
    const label=nodeLabel(node,catalog);
    item.innerHTML=`<span class="order">${index+1}</span><div><strong>${escapeHtml(label)}</strong>
      <small>${escapeHtml(node.kind)}${node.component_id?` · ${escapeHtml(node.component_id)}`:""}</small>
      ${incoming.map(edge=>`<small>← ${escapeHtml(nodeLabel(pipeline.nodes.find(n=>n.id===edge.from.node_id),catalog))}.${escapeHtml(edge.from.port_id)}</small>`).join("")}</div>
      <span class="badge">${incoming.length} in</span>`;
    item.onclick=()=>{selected=node.id;render();announce(`Selected ${label}.`);};
    item.onkeydown=event=>{
      if(event.key==="ArrowUp"){event.preventDefault();moveNode(pipeline,node.id,-1);render();}
      if(event.key==="ArrowDown"){event.preventDefault();moveNode(pipeline,node.id,1);render();}
    };
    return item;
  }));
  for (const id of ["connect-from","connect-to"]) {
    const select=byId(id), previous=select.value;
    select.replaceChildren(...pipeline.nodes.map(node=>new Option(nodeLabel(node,catalog),node.id)));
    if ([...select.options].some(option=>option.value===previous)) select.value=previous;
  }
  const selectedNode=pipeline.nodes.find(node=>node.id===selected);
  byId("node-config").value=selectedNode?JSON.stringify(selectedNode.config,null,2):"";
  byId("open-timeline").href=`/wavedeck.html#pipeline=${encodeURIComponent(pipeline.graph_id)}`;
  validateRemote();
}

async function validateRemote() {
  const generation=++validationGeneration;
  byId("validation").textContent="Validating against current backend catalog…";
  try {
    const report=await request("/api/pipeline/validate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(pipeline),
    });
    if(generation!==validationGeneration)return;
    byId("validation").textContent=report.valid?"Ready to compile and execute":report.diagnostics.map(item=>item.message).join(" ");
    byId("validation").dataset.state=report.valid?"valid":"invalid";
  } catch(error) {
    if(generation===validationGeneration) {
      byId("validation").textContent=error.message;
      byId("validation").dataset.state="invalid";
    }
  }
}

function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function catalogNode(id){return catalog.find(node=>node.id===id);}

function autoConnect() {
  pipeline.edges=[];
  for(let index=1;index<pipeline.nodes.length;index++){
    try{connect(pipeline,pipeline.nodes[index-1].id,pipeline.nodes[index].id,discovery);}catch(_){}
  }
}

function loadTemplate(id) {
  const starter=starters.find(graph=>graph.graph_id===id);
  if(!starter)return announce("No executable starter is available for the current registry.",true);
  pipeline=structuredClone(starter);selected=pipeline.nodes[0]?.id??null;render();
}

function mutate(operation, reconnect=false) {
  if(!selected)return announce("Select a node first.",true);
  try{operation();if(reconnect)autoConnect();render();}catch(error){announce(error.message,true);}
}

function save() {
  pipeline.metadata.name=byId("pipeline-name").value.trim()||"Untitled pipeline";
  localStorage.setItem(STORAGE_KEY,JSON.stringify(pipeline));
  announce("Saved schema-versioned graph locally; runtime components resolve again when opened.");
}

async function restore() {
  try {
    const stored=localStorage.getItem(STORAGE_KEY)??localStorage.getItem("tongues.speech.dataflow.v1");
    if(!stored)throw new Error("no saved graph");
    const report=await request("/api/pipeline/migrate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:stored,
    });
    pipeline=report.document;selected=pipeline.nodes[0]?.id??null;render();
    announce(report.steps.length?`Graph restored after migration: ${report.steps.join("; ")}.`:"Graph restored against the current backend catalog.");
  } catch(error){announce(`Cannot restore: ${error.message}`,true);}
}

function share(){
  const blob=new Blob([JSON.stringify(pipeline,null,2)],{type:"application/json"});
  const link=document.createElement("a");link.href=URL.createObjectURL(blob);
  link.download=`${pipeline.graph_id.replaceAll(":","-")}.json`;link.click();URL.revokeObjectURL(link.href);
}

async function runFixture() {
  const plan=await request("/api/pipeline/compile",{
    method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(pipeline),
  });
  runEvents=[];renderRun();const started=performance.now();byId("level").value=.18;
  setTelemetry(["plan started","pending","pending","pending",`${plan.steps.length} planned stages`,"none"]);
  const wav=new ArrayBuffer(44+32000);const view=new DataView(wav);writeWavHeader(view,16000);
  const response=await fetch("/api/asr/transcriptions?provider=fixture&language=en",{method:"POST",headers:{"Content-Type":"audio/wav"},body:wav});
  if(!response.ok)throw new Error(await response.text());
  const result=await response.json();runEvents=result.events;renderRun();
  const first=runEvents.find(event=>event.type==="partial_hypothesis");
  const committed=runEvents.find(event=>event.type==="committed_segment");
  setTelemetry(["fixture evidence ended",first?.data?.text??"—",committed?.data?.text??"—",
    `${committed?.data?.language?.language??"unknown"} / ${committed?.data?.speaker_id??"unknown"}`,
    `${plan.plan_id} · ${(performance.now()-started).toFixed(1)} ms`,"none"]);
  byId("level").value=0;
  announce("Backend plan compiled; deterministic recognition evidence completed.");
}

function writeWavHeader(v,rate){const text=(o,s)=>[...s].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));text(0,"RIFF");v.setUint32(4,v.byteLength-8,true);text(8,"WAVEfmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,"data");v.setUint32(40,v.byteLength-44,true);}
function setTelemetry(values){[...byId("telemetry").querySelectorAll("dd")].forEach((node,index)=>node.textContent=values[index]??"—");}
function renderRun(){byId("run-events").replaceChildren(...runEvents.map((event,index)=>{const item=document.createElement("li");item.className=event.type==="committed_segment"?"committed":"partial";const confidence=event.data?.confidence?.value;item.textContent=`#${index} ${event.type}: ${event.data?.text??event.data?.reason??""}${confidence==null?"":` · confidence ${confidence}`}`;return item;}));}

byId("add").onclick=()=>{const node=addNode(pipeline,catalogNode(byId("catalog").value),selected);selected=node.id;render();};
byId("remove").onclick=()=>mutate(()=>{removeNode(pipeline,selected);selected=null;});
byId("duplicate").onclick=()=>mutate(()=>{selected=duplicateNode(pipeline,selected).id;});
byId("replace").onclick=()=>mutate(()=>replaceNode(pipeline,selected,catalogNode(byId("replacement").value)));
byId("configure").onclick=()=>mutate(()=>{pipeline.nodes.find(node=>node.id===selected).config=JSON.parse(byId("node-config").value);pipeline.revision++;});
byId("connect").onclick=()=>{try{connect(pipeline,byId("connect-from").value,byId("connect-to").value,discovery);render();announce("Backend-typed connection added; server validation refreshed.");}catch(error){announce(error.message,true);}};
byId("up").onclick=()=>mutate(()=>moveNode(pipeline,selected,-1));byId("down").onclick=()=>mutate(()=>moveNode(pipeline,selected,1));
byId("template").onchange=event=>loadTemplate(event.target.value);byId("save").onclick=save;byId("restore").onclick=()=>restore();byId("share").onclick=share;byId("run").onclick=()=>runFixture().catch(error=>announce(error.message,true));
discover().catch(error=>announce(`Discovery failed: ${error.message}`,true));
