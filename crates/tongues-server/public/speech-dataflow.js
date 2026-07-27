import {
  addNode,addNodeAtConnectionIntent,alignGraphSelection,applyNodeConfig,attachReplacementValidation,bypassNode,buildCatalog,commitReplacement,compatibleTargets,isNodeFaceplateCollapsed,nodeLabel,nodePosition,
  catalogEntryForNode,copyGraphSelection,createEditHistory,createPipeline,deleteGraphSelection,diagnosticsByTarget,distributeGraphSelection,ensureLayout,insertNodeOnEdge,insertionCandidates,insertSubgraph,moveGraphSelection,
  pasteGraphSelection,planNodeReplacement,portsFor,recordEdit,redoEdit,readNodeFaceplateGeometry,removeEdge,replacementCandidates,setNodeFaceplateGeometry,setNodePosition,
  connectionIntentCandidates,tidyGraphSelection,touch,undoEdit,validateSchemaValue,consumeNdjson,
  setNodeFaceplateCollapsed,
  catalogItemIcon,nodeLabelWithIcon,
  addNote,createEmbeddedSubpatch,createFrame,migratePipelineOrganization,moveFrame,setCablePresentation,setSubpatchCollapsed,subpatchBoundaryPorts,subpatchUrl,
} from "./speech-dataflow-model.mjs";
import {createPatchCanvas} from "./speech-patch-canvas.mjs";

const byId=id=>document.getElementById(id);
let discovery=null,catalog=[],starters=[],pipeline=null,cy=null,patchCanvas=null;
let selectedNode=null,selectedEdge=null,connecting=null,validation={valid:false,diagnostics:[]};
let selectedNodes=new Set(),selectedEdges=new Set(),graphClipboard=null,pasteGeneration=0,snapToGrid=false;
let validationGeneration=0,validationTimer=null;
const PIPELINE_RUN_ACTIVE_STATUSES=new Set(["preparing","loading","running","stopping","monitoring"]);
const RUN_EVENT_LIMIT=200;
const PIPELINE_RUN_STATUS_LABELS={
  idle:"Idle",
  preparing:"Preparing",
  loading:"Loading",
  running:"Running",
  monitoring:"Monitoring",
  stopping:"Stopping",
  completed:"Completed",
  failed:"Failed",
  cancelled:"Cancelled",
};
let runState={status:"idle",runId:null,startedAt:0,elapsedMs:0};
let runArtifacts=new Map();
let runStatusTimer=null;
let runTransportRefreshInFlight=false;
let nodeRuntimeState={};
let edgeRuntimeState={};
let runtimeRenderRequested=false;
let editHistory=createEditHistory(),replacementOptions=[],replacementSelected=null,replacementPlan=null;
let replacementPreviewGeneration=0,replacementRenderLimit=100,replacementReturnFocus=null,replacementOverrides={};
let quickAddContext=null,quickAddOptions=[],quickAddReturnFocus=null;
let activeSubpatchId=null,organizationMode=null,organizationBoundary=[];
let browserAudioOutputs=[{deviceId:"default",label:"Browser default"}];

const NODE_THEMES={
  "Sources":{accent:"#75bfff",surface:"#192f44"},
  "Audio processing":{accent:"#5ed7d2",surface:"#183238"},
  "Audio & linguistic processing":{accent:"#69d5bb",surface:"#19342f"},
  "Recognition":{accent:"#a99bff",surface:"#292744"},
  "Language & speaker analysis":{accent:"#d999f2",surface:"#342641"},
  "Linguistic processing":{accent:"#c5a4ff",surface:"#2e2945"},
  "Response generation":{accent:"#f29ac2",surface:"#3a2638"},
  "Synthesis":{accent:"#f3b36f",surface:"#382d23"},
  "Outputs":{accent:"#68dca6",surface:"#1c332c"},
  "Inspection & control":{accent:"#e2ca78",surface:"#343025"},
};
const FALLBACK_NODE_THEME={accent:"#8ba5bf",surface:"#202c3b"};

function announce(text,error=false){byId("status").textContent=text;byId("status").classList.toggle("error",error);}

function linguisticCoverageLabel(item){
  const languages=item?.linguistic_coverage?.languages??[];
  const varieties=item?.linguistic_coverage?.varieties??[];
  return [
    languages.length?`languages ${languages.join(", ")}`:null,
    varieties.length?`varieties ${varieties.join(", ")}`:null,
  ].filter(Boolean).join(" · ");
}

function isRunActive(status=runState.status) {
  return PIPELINE_RUN_ACTIVE_STATUSES.has(status);
}

function isRunLocked() {
  return isRunActive();
}

function renderRunArtifacts() {
  const container=byId("run-artifacts");
  container.replaceChildren();
  if(!runArtifacts.size){
    container.hidden=true;
    return;
  }
  const heading=document.createElement("strong");
  heading.textContent="Generated files";
  container.append(heading);
  for(const artifact of runArtifacts.values()){
    const link=document.createElement("a");
    const filename=artifact.path?.split("/").filter(Boolean).at(-1)??"audio.wav";
    link.className="ui-button";
    link.href=artifact.download_url;
    link.download=filename;
    link.textContent=`Download ${filename}`;
    link.setAttribute("aria-label",`Download generated WAV ${filename}`);
    container.append(link);
  }
  container.hidden=false;
}

function setRunArtifacts(artifacts=[]) {
  runArtifacts=new Map(artifacts
    .filter(artifact=>artifact?.download_url&&artifact?.path)
    .map(artifact=>[artifact.download_url,artifact]));
  renderRunArtifacts();
}

function addRunArtifact(artifact) {
  if(!artifact?.download_url||!artifact?.path)return;
  runArtifacts.set(artifact.download_url,artifact);
  renderRunArtifacts();
}

function setRunState(nextState) {
  runState = {...runState, ...nextState};
  byId("run-state").textContent = PIPELINE_RUN_STATUS_LABELS[runState.status] ?? runState.status;
  byId("run-id").textContent = runState.runId ? `run ${runState.runId}` : "no run yet";
  byId("run-elapsed").textContent = runState.startedAt
    ? ` · ${(runState.elapsedMs / 1000).toFixed(1)}s`
    : "";
  const active = isRunActive(runState.status);
  byId("run").disabled = active;
  byId("stop").disabled = !active || !runState.runId;
  byId("panic").disabled = !active || !runState.runId;
  byId("run-context").hidden = runState.status === "idle" && !runState.runId;
  if (runState.runId) byId("run-tracks-link").href = `/runs/${encodeURIComponent(runState.runId)}/tracks`;
}

function refreshRunTransportClock() {
  if (runState.status !== "idle" && runState.startedAt) {
    runState.elapsedMs = Math.max(0, Date.now() - runState.startedAt);
  }
}

async function refreshRunStateFromServer(runId = runState.runId) {
  if (!runId) return;
  if (runTransportRefreshInFlight) return;
  runTransportRefreshInFlight = true;
  try {
    const run = await request(`/api/pipeline/runs/${encodeURIComponent(runId)}`);
    const next = {runId};
    if (run.started_at_ms) next.startedAt = run.started_at_ms;
    if (run.status) next.status = run.status;
    if (Array.isArray(run.artifacts)) setRunArtifacts(run.artifacts);
    setRunState(next);
  } finally {
    runTransportRefreshInFlight = false;
  }
}

function startRunTransportClock() {
  stopRunTransportClock();
  runStatusTimer = setInterval(() => {
    refreshRunTransportClock();
    if (runState.status !== "idle") byId("run-elapsed").textContent = ` · ${(runState.elapsedMs / 1000).toFixed(1)}s`;
    refreshRunStateFromServer().catch(() => {});
  }, 500);
}

function stopRunTransportClock() {
  if (runStatusTimer) clearInterval(runStatusTimer);
  runStatusTimer = null;
}

function resetRunState() {
  setRunArtifacts();
  setRunState({
    status: "idle",
    runId: null,
    startedAt: 0,
    elapsedMs: 0,
  });
  stopRunTransportClock();
}

function clearRuntimeActivity() {
  nodeRuntimeState = {};
  edgeRuntimeState = {};
  patchCanvas?.render();
}

function updateEditControls(){byId("undo").disabled=!editHistory.undo.length;byId("redo").disabled=!editHistory.redo.length;}
function selectionState(){return{node_id:selectedNode,edge_id:selectedEdge,node_ids:[...selectedNodes],edge_ids:[...selectedEdges]};}
function replaceNodeSelection(id){
  selectedNode=id??null;selectedEdge=null;selectedNodes=new Set(id?[id]:[]);selectedEdges=new Set();
}
function replaceEdgeSelection(id){
  selectedEdge=id??null;selectedNode=null;selectedEdges=new Set(id?[id]:[]);selectedNodes=new Set();
}
function clearSelectionState(){selectedNode=null;selectedEdge=null;selectedNodes=new Set();selectedEdges=new Set();}
function applySelectionState(selection){
  if(typeof selection==="string"){replaceNodeSelection(selection);return;}
  selectedNode=selection?.node_id??null;selectedEdge=selection?.edge_id??null;
  selectedNodes=new Set(selection?.node_ids??(selectedNode?[selectedNode]:[]));
  selectedEdges=new Set(selection?.edge_ids??(selectedEdge?[selectedEdge]:[]));
  selectedNodes.forEach(id=>{if(!pipeline.nodes.some(node=>node.id===id))selectedNodes.delete(id);});
  selectedEdges.forEach(id=>{if(!pipeline.edges.some(edge=>edge.id===id))selectedEdges.delete(id);});
  if(selectedNode&&!pipeline.nodes.some(node=>node.id===selectedNode))selectedNode=null;
  if(selectedEdge&&!pipeline.edges.some(edge=>edge.id===selectedEdge))selectedEdge=null;
  if(selectedNode&&!selectedNodes.has(selectedNode))selectedNodes.add(selectedNode);
  if(selectedEdge&&!selectedEdges.has(selectedEdge))selectedEdges.add(selectedEdge);
}
function focusState(){
  const element=document.activeElement;
  return element?.id?{id:element.id}:null;
}
function restoreFocus(focus){
  if(!focus?.id)return;
  requestAnimationFrame(()=>byId(focus.id)?.focus());
}
function performGraphEdit(label,mutate,after=()=>{}){
  if (isRunLocked()) {
    announce("Stop transport before editing the graph structure.", true);
    return;
  }
  const before=structuredClone(pipeline),selectionBefore=selectionState(),focusBefore=focusState();
  const result=mutate();after(result);
  recordEdit(editHistory,before,pipeline,{
    label,selectionBefore,selectionAfter:selectionState(),focusBefore,focusAfter:focusState(),
  });
  updateEditControls();return result;
}
function recordCompletedGraphEdit(edit){
  const selectionBefore=selectionState(),focusBefore=focusState();
  if(edit.kind==="edge.delete"){
    selectedEdges.delete(edit.edge_id);if(selectedEdge===edit.edge_id)selectedEdge=[...selectedEdges].at(-1)??null;
  }else if(["edge.connect","edge.reconnect"].includes(edit.kind))replaceEdgeSelection(edit.edge_id);
  const labels={"edge.connect":"Connect cable","edge.reconnect":"Reconnect cable","edge.delete":"Delete cable"};
  recordEdit(editHistory,edit.before,pipeline,{
    label:labels[edit.kind]??"Edit cable",selectionBefore,selectionAfter:selectionState(),
    focusBefore,focusAfter:focusState(),
  });
  updateEditControls();renderGraph();scheduleValidation();
}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function readableType(value){
  const labels={
    audio_stream:"audio stream",audio_buffer:"audio buffer",transcript_partial:"partial transcript",
    transcript_revised:"revised transcript",transcript_committed:"committed transcript",
    speaker_assignment:"speaker labels",utterance_plan:"utterance plan",
  };
  return labels[value]??String(value??"unknown").replaceAll("_"," ");
}
function nodePortSummary(node,ports,direction){
  const connected=port=>pipeline.edges.some(edge=>{
    const endpoint=direction==="input"?edge.to:edge.from;
    return endpoint.node_id===node.id&&endpoint.port_id===port.id;
  });
  const primary=ports.filter(port=>!["error","cancellation"].includes(port.value_type))
    .map((port,index)=>({port,index,connected:connected(port)}))
    .sort((a,b)=>Number(b.connected)-Number(a.connected)||Number(Boolean(b.port.must_consume))-Number(Boolean(a.port.must_consume))||a.index-b.index)
    .map(item=>item.port);
  const visible=primary.slice(0,2).map(port=>{
    const name=String(port.label??port.id).replaceAll("_"," ");
    const type=readableType(port.value_type);
    return name==="in"||name==="out"||type.includes(name)?type:`${name}: ${type}`;
  });
  return `${visible.join("  ·  ")}${primary.length>visible.length?`  +${primary.length-visible.length}`:""}`;
}

function nodeTitle(node) {
  return nodeLabelWithIcon(node, catalog);
}

function catalogTitle(item) {
  return `${catalogItemIcon(item)} ${item?.label ?? "Unknown module"}`;
}

async function request(path,options={}){
  const response=await fetch(path,options),text=await response.text();
  let value={};try{value=text?JSON.parse(text):{};}catch{value={error:text};}
  if(!response.ok)throw new Error(value.error??value.validation?.diagnostics?.map(item=>item.message).join(" ")??`${path}: ${response.status}`);
  return value;
}
function jsonOptions(method,value){return{method,headers:{"Content-Type":"application/json"},body:JSON.stringify(value)};}
function graphIdFromRoute(pathname=location.pathname){
  const match=pathname.match(/^\/studio\/graphs\/([^/]+)\/?$/);
  if(!match||match[1]==="new")return null;
  try{return decodeURIComponent(match[1]);}catch{return null;}
}
function graphRoute(graphId,nodeId=""){
  const route=`/studio/graphs/${encodeURIComponent(graphId)}`;
  return nodeId?`${route}?node=${encodeURIComponent(nodeId)}`:route;
}
function showRouteRecovery(message){
  const target=byId("route-recovery");target.hidden=false;
  target.innerHTML=`${escapeHtml(message)} <a href="/studio/graphs/new">Start a new graph</a> or <a href="/runs">open recent runs</a>.`;
}

function findNode(nodeId) {
  return pipeline?.nodes.find(node => node.id === nodeId) ?? null;
}

function signalFamily(valueType) {
  const value=String(valueType ?? "").toLowerCase();
  if (value.includes("audio")) return "audio";
  if (value.includes("transcript") || value === "text") return "text";
  if (["utterance_plan", "control", "cancellation"].includes(value)) return "control";
  if (value === "error") return "error";
  return "other";
}

function canBypassNode(nodeId) {
  const node = findNode(nodeId);
  if (!node || node.bypassed || node.disabled) return false;
  const incoming = pipeline.edges.filter(edge => edge.to.node_id === nodeId);
  const outgoing = pipeline.edges.filter(edge => edge.from.node_id === nodeId);
  if (incoming.length !== 1 || outgoing.length < 1) return false;
  const upstream = pipeline.nodes.find(item => item.id === incoming[0].from.node_id);
  const upstreamPort = portsFor(upstream, "output", discovery).find(port => port.id === incoming[0].from.port_id);
  if (!upstreamPort) return false;
  for (const edge of outgoing) {
    const target = pipeline.nodes.find(item => item.id === edge.to.node_id);
    const targetInput = portsFor(target, "input", discovery).find(port => port.id === edge.to.port_id);
    if (!targetInput || targetInput.value_type !== upstreamPort.value_type) return false;
  }
  return true;
}

function updateInlineNodeConfig({nodeId, field, value}) {
  const node = findNode(nodeId);
  if (!node) throw new Error("Node selection changed; choose this node again.");
  const item = catalogEntryForNode(node, catalog);
  const spec = item?.schema?.properties?.[field];
  if (!spec) throw new Error(`Unknown configuration field: ${field}.`);
  const errors = validateSchemaValue(value, spec);
  if (errors.length) throw new Error(errors.join(" "));
  performGraphEdit("Update node control", () => applyNodeConfig(pipeline, nodeId, {[field]: value}));
  renderGraph();scheduleValidation();
}

function updateNodeDisabledState(nodeId, disabled) {
  const node = findNode(nodeId);
  if (!node) return;
  performGraphEdit(disabled ? "Disable node" : "Enable node", () => {
    if (disabled) {
      pipeline.edges = pipeline.edges.filter(edge => edge.from.node_id !== nodeId && edge.to.node_id !== nodeId);
      pipeline.selected_sinks = pipeline.selected_sinks.filter(sink => sink.node_id !== nodeId);
      node.disabled = true;
    } else node.disabled = false;
    touch(pipeline);
  });
  announce(disabled
    ? "Node disabled and removed from execution; its connections were removed explicitly."
    : "Node enabled; reconnect any required relationships.");
  renderGraph();scheduleValidation();
}

function toggleNodeBypass(nodeId) {
  const node = findNode(nodeId);
  if (!node) throw new Error("Node no longer exists.");
  if (node.bypassed) throw new Error("Undoing bypass is not yet implemented.");
  if (!canBypassNode(nodeId)) throw new Error("This node cannot be bypassed in its current wiring.");
  performGraphEdit("Bypass node", () => bypassNode(pipeline, nodeId, discovery));
  renderGraph();scheduleValidation();
  announce("Node bypassed by explicit compatible rewiring.");
}

function toggleNodeFaceplateCollapsed(nodeId, collapsed) {
  if (!findNode(nodeId)) return;
  performGraphEdit(collapsed ? "Collapse node faceplate" : "Expand node faceplate", () => {
    setNodeFaceplateCollapsed(pipeline, nodeId, collapsed);
  });
  patchCanvas?.renderNodeCards();
}

function updateNodeRuntimeState(event) {
  if (!event?.node_id) return;
  const node = pipeline.nodes.find(item => item.id === event.node_id);
  if (!node) return;
  const entry = nodeRuntimeState[event.node_id] ??= {
    status: "ready",active:0,lastActivityAt:0,lastElapsedMs:0,updatedAt:0,outputCount:0,
  };
  entry.updatedAt = performance.now();
  entry.lastActivityAt = performance.now();
  if (typeof event.elapsed_ms === "number") entry.lastElapsedMs = event.elapsed_ms;
  if (event.kind === "started") {
    entry.status = "loading";
    entry.active += 1;
  } else if (event.kind === "failed" || event.kind === "cancelled") {
    entry.status = "failed";
    entry.error = event.detail ?? "node reported a failure";
  } else if (event.kind === "completed") {
    entry.status = "ready";
    entry.error = null;
  }
  if (event.output) {
    const outputPort = (discovery.node_kinds?.[node.kind]?.ports ?? []).find(port => port.id === event.output?.port_id && port.direction === "output");
    const kind = signalFamily(outputPort?.value_type);
    const preview = computeRuntimePreview(event.output.value, kind);
    entry.preview = preview.text;
    entry.kind = kind;
    if (kind === "audio") entry.meter = preview.meter;
    if (kind === "control") entry.pulse = (entry.pulse ?? 0) + 1;
    entry.status = "active";
    entry.lastEventKind = event.kind;
    entry.outputCount = (entry.outputCount ?? 0) + 1;
  }
  patchCanvas?.renderNodeCards();
}

function updateEdgeRuntimeState(event) {
  if (!event?.node_id || !event.kind) return;
  const node = pipeline.nodes.find(item => item.id === event.node_id);
  if (!node) return;
  const portId = event.output?.port_id;

  const apply = (edgeId, status, details={}) => {
    const entry = edgeRuntimeState[edgeId] ??= {status:"ready"};
    entry.status = status;
    entry.updatedAt = performance.now();
    if (event.elapsed_ms != null) entry.lastElapsedMs = event.elapsed_ms;
    Object.assign(entry, details);
  };

  const outgoing = pipeline.edges.filter(edge => edge.from.node_id === event.node_id);
  const incident = pipeline.edges.filter(edge => edge.from.node_id === event.node_id || edge.to.node_id === event.node_id);

  if (event.kind === "started") {
    outgoing.forEach(edge => apply(edge.id, "loading"));
  } else if (event.kind === "output") {
    if (portId) {
      outgoing
        .filter(edge => edge.from.port_id === portId)
        .forEach(edge => apply(edge.id, "active", {lastPortId: portId}));
    }
  } else if (event.kind === "completed") {
    outgoing.forEach(edge => apply(edge.id, "ready"));
  } else if (event.kind === "failed" || event.kind === "cancelled") {
    incident.forEach(edge => apply(edge.id, "failed", {detail: event.detail}));
  }
}
function computeRuntimePreview(value, kind) {
  const textValue = (raw) => {
    const rendered = typeof raw === "string" ? raw : JSON.stringify(raw);
    return rendered.length > 140 ? `${rendered.slice(0, 137)}…` : rendered;
  };
  if (kind === "audio") {
    const number = Array.isArray(value) ? value[0] : typeof value === "number" ? value : null;
    const meter = number == null ? 0 : Math.max(0, Math.min(1, Math.abs(number)));
    const text = number == null ? "audio activity" : `${meter.toFixed(2)} peak`;
    return {kind:"audio",meter,text};
  }
  if (kind === "control") return {kind:"control",text:textValue(value)};
  if (kind === "text") return {kind:"text",text:textValue(value)};
  if (kind === "error") return {kind:"error",text:textValue(value)};
  return {kind:"other",text:textValue(value)};
}

async function discover(){
  [discovery,{graphs:starters},browserAudioOutputs]=await Promise.all([
    request("/api/pipeline/catalog"),
    request("/api/pipeline/starters"),
    discoverBrowserAudioOutputs(),
  ]);
  const browserOutputSchema=discovery.node_kinds?.audio_output?.configuration_schema?.properties?.browser_device_id;
  if(browserOutputSchema){
    browserOutputSchema.enum=browserAudioOutputs.map(device=>device.deviceId);
    browserOutputSchema["x-enum-labels"]=browserAudioOutputs.map(device=>device.label);
  }
  catalog=buildCatalog(discovery);renderPalette();renderTemplates();
  byId("template").replaceChildren(...starters.map(graph=>new Option(graph.metadata.name,graph.graph_id)));
  const params=new URLSearchParams(location.search),routeGraphId=graphIdFromRoute();
  const requestedStarter=params.get("starter");
  const starterId=routeGraphId?.startsWith("starter:")?routeGraphId:requestedStarter;
  const selectedStarter=starters.find(graph=>graph.graph_id===`starter:${starterId}`)
    ??starters.find(graph=>graph.graph_id===starterId);
  initCanvas();
  let graph=selectedStarter;
  if(routeGraphId&&!selectedStarter){
    try{const value=await request(`/api/pipeline/graphs/${encodeURIComponent(routeGraphId)}`);graph=value.document??value;}
    catch(error){showRouteRecovery(`Graph ${routeGraphId} could not be restored: ${error.message}`);}
  }
  loadGraph(graph??starters[0]??createPipeline());
  const requestedNode=params.get("node");
  if(requestedNode&&pipeline.nodes.some(node=>node.id===requestedNode))selectNode(requestedNode);
  else if(requestedNode)showRouteRecovery(`Node ${requestedNode} is not present in graph ${pipeline.graph_id}.`);
  announce(`Loaded ${pipeline.metadata.name} as an editable graph configuration.`);
}

async function discoverBrowserAudioOutputs(){
  if(!navigator.mediaDevices?.enumerateDevices)return[{deviceId:"default",label:"Browser default"}];
  try{
    const outputs=(await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==="audiooutput");
    const seen=new Set(),result=[{deviceId:"default",label:"Browser default"}];
    outputs.forEach((device,index)=>{
      if(!device.deviceId||device.deviceId==="default"||seen.has(device.deviceId))return;
      seen.add(device.deviceId);
      result.push({deviceId:device.deviceId,label:device.label||`Browser audio output ${index+1}`});
    });
    return result;
  }catch{
    return[{deviceId:"default",label:"Browser default (device discovery unavailable)"}];
  }
}

function initCanvas(){
  if(!globalThis.cytoscape)throw new Error("The patch-canvas library did not load. Check network access to cdn.jsdelivr.net.");
  cy=globalThis.cytoscape({
    container:byId("canvas"),elements:[],wheelSensitivity:.18,boxSelectionEnabled:true,selectionType:"additive",
    style:[
      {selector:"node",style:{
        "shape":"round-rectangle","width":"data(width)","height":"data(height)",
        "background-color":"data(surface)","border-width":3,"border-color":"data(accent)",
        "label":"data(label)","color":"#f7fbff","font-family":"system-ui, sans-serif","font-size":14,
        "font-weight":600,"line-height":1.35,"text-wrap":"wrap","text-max-width":194,
        "text-valign":"center","text-halign":"center","text-justification":"left",
        "shadow-blur":16,"shadow-color":"#000000","shadow-opacity":.36,"shadow-offset-x":0,"shadow-offset-y":6,
      }},
      {selector:"node:selected",style:{"border-color":"#f7fffd","border-width":5,"underlay-color":"#76e2ce","underlay-opacity":.2,"underlay-padding":9}},
      {selector:"node.unavailable",style:{"border-color":"#ff8c91","border-style":"dashed","background-color":"#3a252d"}},
      {selector:"node.invalid",style:{"border-color":"#ffc86b","background-color":"#3b3024"}},
      {selector:"node.inactive",style:{"opacity":.68,"border-style":"dashed"}},
      {selector:"node.compatible",style:{"overlay-color":"#76e2ce","overlay-opacity":.22,"overlay-padding":12}},
      {selector:"edge",style:{
        "curve-style":"bezier","target-arrow-shape":"triangle","arrow-scale":1.2,
        "line-color":"#839bb4","target-arrow-color":"#839bb4","width":4,
        "opacity":0,
        "label":"data(type)","font-family":"system-ui, sans-serif","font-size":11,"font-weight":600,
        "color":"#e5eff9","text-background-color":"#101923","text-background-opacity":.96,
        "text-background-padding":5,"text-background-shape":"roundrectangle",
      }},
      {selector:"edge:selected",style:{"line-color":"#76e2ce","target-arrow-color":"#76e2ce","width":6,"z-index":10}},
      {selector:"edge.invalid",style:{"line-color":"#ffc86b","target-arrow-color":"#ffc86b","line-style":"dashed"}},
    ],
  });
  cy.on("tap","node",event=>{
    const original=event.originalEvent,additive=Boolean(original?.shiftKey||original?.ctrlKey||original?.metaKey);
    selectNode(event.target.id(),{additive,toggle:additive});
  });
  cy.on("tap","edge",event=>{
    const original=event.originalEvent,additive=Boolean(original?.shiftKey||original?.ctrlKey||original?.metaKey);
    selectEdge(event.target.id(),{additive,toggle:additive});
  });
  cy.on("tap",event=>{if(event.target===cy)clearSelection();});
  cy.on("boxselect","node",event=>selectNode(event.target.id(),{additive:true}));
  cy.on("boxselect","edge",event=>selectEdge(event.target.id(),{additive:true}));
  cy.on("dragfree","node",event=>{
    const targetId=event.target.id(),ids=selectedNodes.has(targetId)?[...selectedNodes]:[targetId];
    performGraphEdit("Move selection",()=>{
      ids.forEach(id=>{
        const position=cy.getElementById(id).position(),rounded=snapToGrid?{x:Math.round(position.x/24)*24,y:Math.round(position.y/24)*24}:position;
        setNodePosition(pipeline,id,rounded);
      });
      touch(pipeline);
    });
    scheduleValidation();renderOutline();
  });
  byId("canvas").addEventListener("dblclick",event=>{
    if(event.target.closest?.("[data-patch-jack]"))return;
    openQuickAdd({kind:"empty",position:canvasPoint(event.clientX,event.clientY)});
  });
  byId("canvas").addEventListener("contextmenu",event=>{
    event.preventDefault();openQuickAdd({kind:"empty",position:canvasPoint(event.clientX,event.clientY)});
  });
}

export const graphStudioTestHooks={
  renderedNodeBounds(nodeId){
    const node=cy?.getElementById(nodeId);
    if(!node?.length)return null;
    const bounds=node.renderedBoundingBox({includeLabels:false,includeOverlays:false});
    return{x:bounds.x1,y:bounds.y1,width:bounds.w,height:bounds.h,right:bounds.x2,bottom:bounds.y2};
  },
  viewportBounds(){
    const bounds=byId("canvas").getBoundingClientRect();
    return{x:0,y:0,width:bounds.width,height:bounds.height,right:bounds.width,bottom:bounds.height};
  },
  panBy(offset){cy?.panBy(offset);},
  zoom(level){cy?.zoom({level,renderedPosition:{x:byId("canvas").clientWidth/2,y:byId("canvas").clientHeight/2}});},
  teardownAndReinitialize(){
    patchCanvas?.destroy();
    patchCanvas=null;
    cy?.destroy();
    cy=null;
    initCanvas();
    renderGraph();
    ensurePatchCanvas();
    patchCanvas.render();
  },
};

function renderPalette(){
  const query=byId("palette-search").value.trim().toLowerCase(),groups=new Map();
  catalog.filter(item=>`${item.label} ${item.kind} ${item.detail} ${item.group} ${linguisticCoverageLabel(item)}`.toLowerCase().includes(query))
    .forEach(item=>{if(!groups.has(item.group))groups.set(item.group,[]);groups.get(item.group).push(item);});
  byId("palette").replaceChildren(...[...groups].map(([group,items])=>{
    const details=document.createElement("details");details.className="palette-group";details.open=true;
    const summary=document.createElement("summary");summary.textContent=`${group} (${items.length})`;details.append(summary);
    const list=document.createElement("div");list.className="palette-list";
    items.forEach(item=>{const button=document.createElement("button");button.className="palette-node";button.dataset.readiness=item.readiness;
      button.innerHTML=`
        <span class="palette-node-title">
          <span class="palette-node-icon" aria-hidden="true">${escapeHtml(catalogItemIcon(item))}</span>
          <span>${escapeHtml(item.label)}</span>
        </span>
        <small>${escapeHtml([item.kind,item.readiness,linguisticCoverageLabel(item)].filter(Boolean).join(" · "))}</small>
      `;
      button.title=[item.detail,linguisticCoverageLabel(item)].filter(Boolean).join(" · ");button.draggable=true;
      button.ondragstart=event=>event.dataTransfer.setData("application/x-tongues-catalog-id",item.id);
      button.onclick=()=>addCatalogNode(item);list.append(button);});
    details.append(list);return details;
  }));
}

function renderTemplates(){
  byId("subgraphs").replaceChildren(...starters.map(graph=>{
    const button=document.createElement("button");button.textContent=`Insert ${graph.metadata.name}`;
    button.onclick=()=>{
      const ids=performGraphEdit(`Insert ${graph.metadata.name}`,()=>insertSubgraph(pipeline,graph,{x:cy.extent().x1+80,y:cy.extent().y1+80}),result=>{selectedNodes=new Set(result);selectedNode=result[0]??null;selectedEdges=new Set();selectedEdge=null;});
      renderGraph();if(ids[0])selectNode(ids[0]);announce(`Inserted ${graph.metadata.name} as a reusable subgraph.`);
    };return button;
  }));
}

function addCatalogNode(item){
  const center=cy.extent(),afterId=selectedNode;
  const node=performGraphEdit(`Add ${item.label}`,()=>addNode(pipeline,item,afterId,{x:(center.x1+center.x2)/2,y:(center.y1+center.y2)/2}),result=>replaceNodeSelection(result.id));
  if(!node)return;
  renderGraph();selectNode(node.id);announce(`Added ${item.label}.`);
}

function canvasPoint(clientX,clientY){
  const bounds=byId("canvas").getBoundingClientRect(),pan=cy.pan?.()??{x:0,y:0},zoom=cy.zoom?.()??1;
  return{x:Math.round((clientX-bounds.left-pan.x)/zoom),y:Math.round((clientY-bounds.top-pan.y)/zoom)};
}
function canvasCenterPoint(){
  const bounds=byId("canvas").getBoundingClientRect();
  return canvasPoint(bounds.left+bounds.width/2,bounds.top+bounds.height/2);
}
function openQuickAdd(context){
  quickAddContext=context;quickAddReturnFocus=document.activeElement;
  if(context.kind==="insert_edge")quickAddOptions=insertionCandidates(pipeline,context.edge_id,catalog,discovery);
  else if(["from_output","to_input"].includes(context.kind)){
    const node=pipeline.nodes.find(item=>item.id===context.anchor.node_id);
    const direction=context.kind==="from_output"?"output":"input";
    const port=portsFor(node,direction,discovery).find(item=>item.id===context.anchor.port_id);
    quickAddOptions=connectionIntentCandidates(catalog,context.kind,port?.value_type);
  }else quickAddOptions=catalog.map(candidate=>({candidate,compatible:(candidate.readiness??"ready")==="ready",ambiguous:false,reason:candidate.detail||"Backend-discovered module."}));
  const labels={
    empty:"Add a backend-discovered module at this canvas position.",
    from_output:"Choose a module with one compatible input; it will be added and connected atomically.",
    to_input:"Choose a module with one compatible output; it will be added and connected atomically.",
    insert_edge:"Choose an unambiguous typed processor to insert on the selected cable.",
  };
  byId("quick-add-context").textContent=labels[context.kind];byId("quick-add-search").value="";
  renderQuickAdd();byId("quick-add-dialog").showModal();byId("quick-add-search").focus();
}
function renderQuickAdd(){
  const query=byId("quick-add-search").value.trim().toLowerCase();
  const options=quickAddOptions.filter(option=>`${option.candidate.label} ${option.candidate.provider} ${option.candidate.model} ${option.candidate.kind} ${option.reason} ${linguisticCoverageLabel(option.candidate)}`.toLowerCase().includes(query));
  byId("quick-add-results").replaceChildren(...options.slice(0,200).map(option=>{
    const button=document.createElement("button");button.type="button";button.className="quick-add-option";button.setAttribute("role","option");
    button.setAttribute("aria-disabled",String(!option.compatible));button.disabled=!option.compatible;
    button.innerHTML=`
      <span class="quick-add-title">
        <span class="quick-add-icon" aria-hidden="true">${escapeHtml(catalogItemIcon(option.candidate))}</span>
        <span>
          <strong>${escapeHtml(option.candidate.label)}</strong>
          <small>${escapeHtml([option.candidate.provider,option.candidate.model,linguisticCoverageLabel(option.candidate),option.reason].filter(Boolean).join(" · "))}</small>
        </span>
      </span>
    `;
    button.onclick=()=>applyQuickAdd(option);return button;
  }));
  if(!options.length)byId("quick-add-results").innerHTML='<p class="muted">No backend-discovered modules match this typed intent.</p>';
}
function closeQuickAdd(message="Quick-add cancelled; the graph was not changed."){
  if(byId("quick-add-dialog").open)byId("quick-add-dialog").close();
  quickAddReturnFocus?.focus?.();quickAddContext=null;announce(message);
}
function applyQuickAdd(option){
  try{
    const context=quickAddContext;
    const result=performGraphEdit(context.kind==="insert_edge"?"Insert module on cable":"Quick-add module",()=>{
      if(context.kind==="insert_edge")return insertNodeOnEdge(pipeline,context.edge_id,option.candidate,option.mappings[0],discovery,context.position);
      if(["from_output","to_input"].includes(context.kind))return addNodeAtConnectionIntent(pipeline,option.candidate,context.anchor,context.kind,discovery,context.position);
      return{node:addNode(pipeline,option.candidate,null,context.position)};
    },value=>replaceNodeSelection(value.node.id));
    byId("quick-add-dialog").close();quickAddContext=null;renderGraph();scheduleValidation();selectNode(result.node.id);
    announce(`${context.kind==="insert_edge"?"Inserted":"Added"} ${option.candidate.label} as one undoable edit.`);
  }catch(error){announce(error.message,true);}
}
function dropCatalogOnEdge(intent){
  const item=catalog.find(candidate=>candidate.id===intent.catalog_id);if(!item)return;
  const option=insertionCandidates(pipeline,intent.edge_id,[item],discovery)[0];
  if(!option?.compatible)return announce(option?.reason??"That module cannot be inserted on this cable.",true);
  try{
    const result=performGraphEdit("Insert module on cable",()=>insertNodeOnEdge(pipeline,intent.edge_id,item,option.mappings[0],discovery,canvasPoint(intent.clientX,intent.clientY)),value=>replaceNodeSelection(value.node.id));
    renderGraph();scheduleValidation();selectNode(result.node.id);announce(`Inserted ${item.label} on the cable.`);
  }catch(error){announce(error.message,true);}
}
function dropCatalogOnJack(intent){
  const item=catalog.find(candidate=>candidate.id===intent.catalog_id);if(!item)return;
  const kind=intent.direction==="output"?"from_output":"to_input",anchor={node_id:intent.node_id,port_id:intent.port_id};
  const option=connectionIntentCandidates([item],kind,portsFor(pipeline.nodes.find(node=>node.id===intent.node_id),intent.direction,discovery).find(port=>port.id===intent.port_id)?.value_type)[0];
  if(!option?.compatible)return announce(option?.reason??"That module has no unambiguous compatible port.",true);
  try{
    const result=performGraphEdit("Add module at jack",()=>addNodeAtConnectionIntent(pipeline,item,anchor,kind,discovery,canvasPoint(intent.clientX,intent.clientY)),value=>replaceNodeSelection(value.node.id));
    renderGraph();scheduleValidation();selectNode(result.node.id);announce(`Added ${item.label} at the ${intent.direction} jack.`);
  }catch(error){announce(error.message,true);}
}

function loadGraph(graph,{preserveHistory=false}={}){
  const previousSelection=selectionState();
  pipeline=migratePipelineOrganization(structuredClone(graph));pipeline.metadata.labels??={};ensureLayout(pipeline);
  const requestedSubpatch=new URLSearchParams(location.search).get("subpatch");
  activeSubpatchId=pipeline.subpatches.some(item=>item.id===requestedSubpatch)?requestedSubpatch:null;
  nodeRuntimeState={};
  edgeRuntimeState={};
  if(!preserveHistory)editHistory=createEditHistory();
  if(preserveHistory)applySelectionState(previousSelection);
  else replaceNodeSelection(pipeline.nodes[0]?.id??null);
  connecting=null;
  byId("pipeline-name").value=pipeline.metadata.name;
  byId("cable-opacity").value=String(pipeline.presentation.global_cable_opacity);byId("focus-path").setAttribute("aria-pressed",String(pipeline.presentation.selected_path_focus));
  byId("graph-identity").textContent=pipeline.graph_id.startsWith("starter:")
    ?"Editing a configuration draft seeded from a backend template"
    :`Editing saved graph ${pipeline.graph_id}, revision ${pipeline.revision}`;
  document.title=`${pipeline.metadata.name} · Graph Studio · Tongues`;
  renderGraph();ensurePatchCanvas();patchCanvas.render();updateEditControls();renderOrganization();scheduleValidation(0);
}

function ensurePatchCanvas(){
  if(patchCanvas)return;
  patchCanvas=createPatchCanvas({
    container:byId("canvas"),cy,
    getPipeline:()=>pipeline,getDiscovery:()=>discovery,getCatalog:()=>catalog,
    nodeLabel,getSelectedEdgeId:()=>selectedEdge,
    nodeIcon:nodeLabelWithIcon,
    isEdgeSelected:id=>selectedEdges.has(id),
    diagnosticsByEdge:()=>diagnosticsByTarget(validation).edges,
    onSelectNode:selectNode,onSelectEdge:selectEdge,
    onGraphEdit:recordCompletedGraphEdit,
    onDropEmpty:intent=>openQuickAdd({...intent,position:canvasPoint(intent.clientX,intent.clientY)}),
    onDropCatalogOnEdge:dropCatalogOnEdge,onDropCatalogOnJack:dropCatalogOnJack,
    getNodeRuntimeState:nodeId=>nodeRuntimeState[nodeId],
    getEdgeRuntimeState:edgeId=>edgeRuntimeState[edgeId],
    getNodeControlState:nodeId=>(diagnosticsByTarget(validation).nodes[nodeId] ?? {}),
    getNodeFaceplateGeometry:nodeId=>readNodeFaceplateGeometry(pipeline,nodeId),
    onSetNodeFaceplateGeometry:(nodeId,geometry)=>setNodeFaceplateGeometry(pipeline,nodeId,geometry),
    isNodeCollapsed:nodeId=>isNodeFaceplateCollapsed(pipeline,nodeId),
    onSetNodeCollapsed:toggleNodeFaceplateCollapsed,
    onNodeConfigChange:updateInlineNodeConfig,
    canBypassNode,
    onBypassNode:toggleNodeBypass,
    onDisableNode:updateNodeDisabledState,
    isRunLocked,
    getVisibleNodeIds:()=>activeSubpatchId?new Set(pipeline.subpatches.find(item=>item.id===activeSubpatchId)?.node_ids??[]):null,
    onAnnounce:announce,
  });
}

function graphElements(){
  const grouped=diagnosticsByTarget(validation);
  const visibleIds=activeSubpatchId?new Set(pipeline.subpatches.find(item=>item.id===activeSubpatchId)?.node_ids??[]):null;
  const nodes=pipeline.nodes.filter(node=>!visibleIds||visibleIds.has(node.id)).map(node=>{
    const item=catalogEntryForNode(node,catalog);
    const ports=discovery.node_kinds?.[node.kind]?.ports??[];
    const kind=discovery.node_kinds?.[node.kind],theme=NODE_THEMES[item?.group]??FALLBACK_NODE_THEME;
    const label=nodeTitle(node);
    const geometry=readNodeFaceplateGeometry(pipeline,node.id);
    const collapsed=isNodeFaceplateCollapsed(pipeline,node.id);
    const width=Math.max(1,Math.round(geometry.width));
    const height=Math.max(1,Math.round(collapsed?geometry.collapsed_height:geometry.height));
    const classes=[item?.readiness&&item.readiness!=="ready"?"unavailable":"",grouped.nodes[node.id]?.length?"invalid":"",node.disabled||node.bypassed?"inactive":""].filter(Boolean).join(" ");
    return{group:"nodes",data:{id:node.id,label,accent:theme.accent,surface:theme.surface,width,height},position:nodePosition(pipeline,node.id),classes};
  });
  const edges=pipeline.edges.filter(edge=>!visibleIds||(visibleIds.has(edge.from.node_id)&&visibleIds.has(edge.to.node_id))).map(edge=>{
    const source=pipeline.nodes.find(node=>node.id===edge.from.node_id);
    const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
    return{group:"edges",data:{id:edge.id,source:edge.from.node_id,target:edge.to.node_id,type:readableType(port?.value_type)},classes:grouped.edges[edge.id]?.length?"invalid":""};
  });
  return[...nodes,...edges];
}

function renderGraph(){
  cy.elements().remove();cy.add(graphElements());
  [...selectedNodes,...selectedEdges].forEach(id=>cy.getElementById(id).select());
  renderOutline();renderInspector();renderOrganization();patchCanvas?.render();byId("pipeline-name").value=pipeline.metadata.name;
}

function renderOrganization(){
  if(!pipeline)return;
  const crumbs=[{id:null,title:pipeline.metadata.name},...subpatchAncestors(activeSubpatchId)];
  byId("subpatch-breadcrumbs").replaceChildren(...crumbs.flatMap((crumb,index)=>{
    const button=document.createElement("button");button.type="button";button.textContent=crumb.title;button.disabled=crumb.id===activeSubpatchId;
    button.onclick=()=>navigateSubpatch(crumb.id);
    return index?[document.createTextNode("›"),button]:[button];
  }));
  byId("organization-list").replaceChildren(
    ...pipeline.presentation.frames.map(frame=>{
      const item=organizationItem(`Frame: ${frame.title}`,`${frame.node_ids.length} nodes · presentation only`);
      const move=document.createElement("button");move.type="button";move.textContent="Move frame and contents right";move.onclick=()=>{performGraphEdit("Move frame and contents",()=>moveFrame(pipeline,frame.id,{x:24,y:0},{moveContents:true}));renderGraph();scheduleValidation();};
      item.append(move);return item;
    }),
    ...pipeline.presentation.notes.map(note=>organizationItem(`Note: ${note.text}`,"Presentation only")),
    ...pipeline.subpatches.map(subpatch=>{
      const members=new Set(subpatch.node_ids);
      const diagnosticCount=(validation.diagnostics??[]).filter(item=>members.has(item.target?.node_id)).length;
      const unavailable=subpatch.node_ids.filter(id=>catalogEntryForNode(pipeline.nodes.find(node=>node.id===id),catalog)?.readiness!=="ready").length;
      const activity=subpatch.node_ids.map(id=>nodeRuntimeState[id]?.status).filter(status=>["active","failed"].includes(status));
      const item=organizationItem(subpatch.title,`${subpatch.node_ids.length} internal nodes · ${subpatch.exposed_ports.length} reviewed ports · ${diagnosticCount} diagnostics · ${unavailable} unavailable · ${activity.filter(status=>status==="active").length} active · ${activity.filter(status=>status==="failed").length} failed · embedded definition v${subpatch.definition_version}`);
      const row=document.createElement("div");row.className="row";
      const open=document.createElement("button");open.type="button";open.textContent="Open";open.onclick=()=>navigateSubpatch(subpatch.id);
      const collapsed=(pipeline.presentation.collapsed_subpatches??[]).includes(subpatch.id);
      const toggle=document.createElement("button");toggle.type="button";toggle.textContent=collapsed?"Expand":"Collapse";toggle.setAttribute("aria-pressed",String(collapsed));
      toggle.onclick=()=>{performGraphEdit(`${collapsed?"Expand":"Collapse"} subpatch`,()=>setSubpatchCollapsed(pipeline,subpatch.id,!collapsed));renderGraph();scheduleValidation();};
      row.append(open,toggle);item.append(row);return item;
    }),
  );
}
function organizationItem(title,detail){
  const item=document.createElement("section");item.className="organization-item";
  const strong=document.createElement("strong");strong.textContent=title;
  const paragraph=document.createElement("p");paragraph.textContent=detail;item.append(strong,paragraph);return item;
}
function subpatchAncestors(id){
  const result=[],seen=new Set();let current=pipeline.subpatches.find(item=>item.id===id);
  while(current&&!seen.has(current.id)&&result.length<8){seen.add(current.id);result.unshift({id:current.id,title:current.title});current=pipeline.subpatches.find(item=>item.id===current.parent_subpatch_id);}
  return result;
}
function navigateSubpatch(id,{replace=false}={}){
  activeSubpatchId=id;
  const url=id?subpatchUrl(pipeline.graph_id,id):graphRoute(pipeline.graph_id);
  history[replace?"replaceState":"pushState"]({subpatch:id},"",url);
  const members=id?new Set(pipeline.subpatches.find(item=>item.id===id)?.node_ids??[]):null;
  if(members){selectedNodes=new Set([...selectedNodes].filter(nodeId=>members.has(nodeId)));selectedNode=[...selectedNodes].at(-1)??null;}
  renderGraph();announce(id?`Opened subpatch ${pipeline.subpatches.find(item=>item.id===id)?.title}.`:"Returned to the root graph.");
}

function renderOutline(){
  const visibleIds=activeSubpatchId?new Set(pipeline.subpatches.find(item=>item.id===activeSubpatchId)?.node_ids??[]):null;
  byId("graph-outline").replaceChildren(...pipeline.nodes.filter(node=>!visibleIds||visibleIds.has(node.id)).map(node=>{
    const item=document.createElement("li"),button=document.createElement("button");
    button.textContent=nodeTitle(node);
    button.setAttribute("aria-pressed",String(selectedNodes.has(node.id)));
    button.onclick=event=>{const additive=Boolean(event.shiftKey||event.ctrlKey||event.metaKey);selectNode(node.id,{additive,toggle:additive});};button.onkeydown=event=>{
      const deltas={ArrowLeft:[-20,0],ArrowRight:[20,0],ArrowUp:[0,-20],ArrowDown:[0,20]};
      if(deltas[event.key]){
        event.preventDefault();const [x,y]=deltas[event.key],ids=selectedNodes.has(node.id)?[...selectedNodes]:[node.id];
        performGraphEdit("Nudge selection",()=>moveGraphSelection(pipeline,ids,{x,y},{snap:snapToGrid?24:0}));
        renderGraph();cy.getElementById(node.id).select();
      }
      if(event.key==="Delete"){event.preventDefault();deleteSelectedObjects();}
    };item.append(button);return item;
  }));
}

function selectNode(id,{additive=false,toggle=false}={}){
  if(!additive)replaceNodeSelection(id);
  else{
    if(toggle&&selectedNodes.has(id))selectedNodes.delete(id);else selectedNodes.add(id);
    selectedNode=selectedNodes.has(id)?id:[...selectedNodes].at(-1)??null;selectedEdge=null;
  }
  cy.elements().unselect();[...selectedNodes,...selectedEdges].forEach(selected=>cy.getElementById(selected).select());
  renderOutline();renderInspector();patchCanvas?.render();
  const selected=pipeline.nodes.find(node=>node.id===id);
  announce(`${selectedNodes.size+selectedEdges.size} object${selectedNodes.size+selectedEdges.size===1?"":"s"} selected. ${selected?nodeTitle(selected):"Module"} is ${selectedNodes.has(id)?"included":"not included"}.`);
}
function selectEdge(id,{additive=false,toggle=false}={}){
  if(!additive)replaceEdgeSelection(id);
  else{
    if(toggle&&selectedEdges.has(id))selectedEdges.delete(id);else selectedEdges.add(id);
    selectedEdge=selectedEdges.has(id)?id:[...selectedEdges].at(-1)??null;selectedNode=null;
  }
  cy.elements().unselect();[...selectedNodes,...selectedEdges].forEach(selected=>cy.getElementById(selected).select());
  renderOutline();renderInspector();patchCanvas?.render();const edge=pipeline.edges.find(item=>item.id===id);
  announce(`${selectedNodes.size+selectedEdges.size} object${selectedNodes.size+selectedEdges.size===1?"":"s"} selected. Connection from ${edge?.from.node_id}.${edge?.from.port_id} to ${edge?.to.node_id}.${edge?.to.port_id} is ${selectedEdges.has(id)?"included":"not included"}.`);
}
function clearSelection(){clearSelectionState();connecting=null;cy.elements().unselect();renderOutline();renderInspector();patchCanvas?.render();}

function renderInspector(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode),edge=pipeline.edges.find(item=>item.id===selectedEdge);
  byId("empty-inspector").hidden=Boolean(node||edge);byId("node-inspector").hidden=!node;byId("edge-inspector").hidden=!edge;
  if(node)renderNodeInspector(node);if(edge)renderEdgeInspector(edge);
}

function renderNodeInspector(node){
  const item=catalogEntryForNode(node,catalog);
  byId("node-title").textContent=nodeTitle(node);byId("node-detail").textContent=[item?.detail??node.kind,linguisticCoverageLabel(item)].filter(Boolean).join(" · ");
  const commandByKind={tts:"speak",interpretation:"interpretation/stream"};
  const commandId=commandByKind[node.kind];
  byId("node-docs").href=commandId?`/commands/${commandId}`:`/commands?capability=${encodeURIComponent(node.kind)}`;
  byId("disable").textContent=node.disabled?"Enable":"Disable";
  byId("node-readiness").textContent=`Readiness: ${item?.readiness??"unknown"}`;byId("node-readiness").className=item?.readiness==="ready"?"ready":"error";
  const grouped=diagnosticsByTarget(validation);
  byId("input-ports").replaceChildren(...portsFor(node,"input",discovery).map(port=>portButton(node,port,grouped)));
  byId("output-ports").replaceChildren(...portsFor(node,"output",discovery).map(port=>portButton(node,port,grouped)));
  renderConfig(node,item?.schema??{});renderDiagnostics(byId("node-diagnostics"),[...(grouped.nodes[node.id]??[]),...Object.entries(grouped.ports).filter(([key])=>key.startsWith(`${node.id}:`)).flatMap(([,items])=>items)]);
  byId("connection-panel").hidden=!connecting;byId("canvas").dataset.connecting=String(Boolean(connecting));
  if(connecting){
    const source=pipeline.nodes.find(n=>n.id===connecting.node_id);
    byId("connection-source").textContent=`Connecting ${source?nodeTitle(source):"this module"}.${connecting.port_id} (${connecting.value_type})`;
  }
}

function portButton(node,port,grouped){
  const button=document.createElement("button"),connected=pipeline.edges.some(edge=>(port.direction==="input"?edge.to:edge.from).node_id===node.id&&(port.direction==="input"?edge.to:edge.from).port_id===port.id);
  const portDiagnostics=grouped.ports[`${node.id}:${port.id}`]??[];
  const missing=port.direction==="input"&&!connected&&(port.cardinality==="one"||portDiagnostics.some(item=>item.code.includes("input")));
  const compatible=connecting&&port.direction==="input"&&compatibleTargets(pipeline,connecting.node_id,connecting.port_id,discovery).some(target=>target.node_id===node.id&&target.port_id===port.id);
  button.className=`port${missing?" missing":""}${compatible?" compatible":""}`;button.dataset.type=port.value_type;
  const delivery=port.streaming?"stream":port.cardinality;
  button.innerHTML=`<span class="port-dot"></span><span>${escapeHtml(port.label)}<small>${escapeHtml(port.value_type)} · ${escapeHtml(delivery)}</small></span><span>${connected?"●":missing?"○":"·"}</span>`;
  const diagnostics=portDiagnostics;
  button.setAttribute("aria-label",`${port.direction} ${port.label}, type ${port.value_type}, ${connected?"connected":missing?"required and missing":"not connected"}${diagnostics.length?`. ${diagnostics.flatMap(item=>[item.message,...(item.suggestions??[])]).join(". ")}`:""}`);
  button.onclick=()=>{
    if(port.direction==="output"){connecting={node_id:node.id,port_id:port.id,value_type:port.value_type};highlightCompatible();renderInspector();announce(`Choose an input compatible with ${port.value_type}.`);}
    else if(connecting){try{
      performGraphEdit("Connect cable",()=>connectPorts(pipeline,connecting.node_id,connecting.port_id,node.id,port.id,discovery));
      connecting=null;renderGraph();scheduleValidation();announce("Typed connection added.");
    }catch(error){announce(error.message,true);}}
  };
  return button;
}

function highlightCompatible(){
  cy.nodes().removeClass("compatible");
  if(!connecting)return;
  const targets=new Set(compatibleTargets(pipeline,connecting.node_id,connecting.port_id,discovery).map(item=>item.node_id));
  targets.forEach(id=>cy.getElementById(id).addClass("compatible"));
}

function renderConfig(node,schema){
  const properties=schema.properties??{},required=new Set(schema.required??[]);
  const entries=Object.entries(properties).filter(([,spec])=>{
    const condition=spec["x-ui-visible-when"];
    return !condition||Object.entries(condition).every(([field,value])=>(node.config?.[field]??properties[field]?.default)===value);
  });
  if(!entries.length){byId("config-fields").innerHTML='<span class="muted">This node has no configurable fields.</span>';return;}
  byId("config-fields").replaceChildren(...entries.map(([name,spec])=>{
    const label=document.createElement("label");label.textContent=`${spec.title??name}${required.has(name)?" *":""}`;
    let input;
    if(spec["x-ui-source"]==="browser_audio_outputs"){
      input=document.createElement("select");
      input.replaceChildren(...browserAudioOutputs.map(device=>new Option(device.label,device.deviceId)));
    }
    else if(spec.enum){
      input=document.createElement("select");
      const labels=spec["x-enum-labels"]??[];
      input.replaceChildren(...spec.enum.map((value,index)=>new Option(String(labels[index]??value),String(value))));
    }
    else{
      input=document.createElement(spec.type==="string"&&spec.format==="multiline"?"textarea":"input");
      if(input.tagName==="INPUT")input.type=spec.type==="number"||spec.type==="integer"?"number":spec.type==="boolean"?"checkbox":spec.format==="uri"?"url":"text";
      if(spec.minimum!=null)input.min=spec.minimum;if(spec.maximum!=null)input.max=spec.maximum;
    }
    input.dataset.config=name;const value=node.config?.[name]??spec.default;
    if(input.type==="checkbox")input.checked=Boolean(value);else if(value!=null)input.value=typeof value==="object"?JSON.stringify(value):String(value);
    if(spec.description){input.title=spec.description;if(input.tagName==="TEXTAREA"||input.type==="text")input.placeholder=spec.description;}
    if(required.has(name))input.required=true;
    label.append(input);return label;
  }));
}

function applyConfig(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;
  try{
    const fields=[...byId("config-fields").querySelectorAll("[data-config]")];
    const invalid=fields.find(input=>!input.checkValidity());
    if(invalid){invalid.reportValidity();return announce(`Enter a valid value for ${invalid.dataset.config}.`,true);}
    const values=Object.fromEntries(fields.map(input=>{let value=input.type==="checkbox"?input.checked:input.value;if(input.type==="number")value=Number(value);return[input.dataset.config,value];}));
    performGraphEdit("Edit node configuration",()=>applyNodeConfig(pipeline,node.id,values));
    renderGraph();renderInspector();scheduleValidation();announce("Configuration applied.");
  }catch(error){announce(error.message,true);}
}

function renderEdgeInspector(edge){
  const source=pipeline.nodes.find(node=>node.id===edge.from.node_id),target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
  const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
  byId("edge-title").textContent=`${source?nodeTitle(source):"Unknown module"} → ${target?nodeTitle(target):"Unknown module"}`;
  byId("edge-type").textContent=`${edge.from.port_id} → ${edge.to.port_id} · ${port?.value_type??"unknown"}`;byId("edge-capacity").value=edge.capacity;
  const presentation=pipeline.presentation.cables[edge.id]??{};
  byId("cable-routing").value=presentation.routing??"curved";byId("cable-emphasized").checked=Boolean(presentation.emphasized);
  renderDiagnostics(byId("edge-diagnostics"),diagnosticsByTarget(validation).edges[edge.id]??[]);
}
function renderDiagnostics(container,items){container.replaceChildren(...items.map(item=>{const p=document.createElement("p");p.className="diagnostic";p.textContent=[item.message,...(item.suggestions??[])].join(" ");return p;}));}

function replacementCompatibilityLabel(option){
  return option.compatibility==="exact_drop_in"?"Exact drop-in":option.compatibility==="migration"?"Migration":"Incompatible";
}

function openReplacementPicker(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;
  replacementOptions=replacementCandidates(pipeline,node.id,catalog);
  replacementSelected=null;replacementPlan=null;replacementOverrides={};replacementRenderLimit=100;
  replacementReturnFocus=document.activeElement;
  byId("replacement-context").textContent=`Replacing ${nodeTitle(node)} (${node.component_id??node.kind}). The graph is unchanged until Apply.`;
  byId("replacement-search").value="";byId("replacement-provider").replaceChildren(new Option("All providers",""),...derivedReplacementOptions("provider"));
  byId("replacement-readiness").replaceChildren(new Option("All readiness states",""),...derivedReplacementOptions("readiness"));
  showReplacementPickerStep();renderReplacementCandidates();
  byId("replacement-dialog").showModal();byId("replacement-search").focus();
}

function derivedReplacementOptions(field){
  return [...new Set(replacementOptions.map(option=>option[field]).filter(Boolean))].sort((a,b)=>a.localeCompare(b)).map(value=>new Option(value,value));
}

function showReplacementPickerStep(){
  byId("replacement-picker-controls").hidden=false;byId("replacement-list").hidden=false;byId("replacement-review").hidden=true;
  byId("replacement-back").hidden=true;byId("replacement-continue").hidden=false;byId("replacement-apply").hidden=true;
  byId("replacement-continue").disabled=!replacementSelected?.applyable;
}

function filteredReplacementOptions(){
  const query=byId("replacement-search").value.trim().toLowerCase(),provider=byId("replacement-provider").value,readiness=byId("replacement-readiness").value;
  return replacementOptions.filter(option=>(!provider||option.provider===provider)&&(!readiness||option.readiness===readiness)
    &&(!query||`${option.label} ${option.provider} ${option.model} ${option.component_id} ${option.detail} ${option.reason} ${linguisticCoverageLabel(option)}`.toLowerCase().includes(query)));
}

function renderReplacementCandidates(){
  const filtered=filteredReplacementOptions(),visible=filtered.slice(0,replacementRenderLimit),fragment=document.createDocumentFragment();
  for(const option of visible){
    const button=document.createElement("button");button.type="button";button.className="replacement-candidate";button.setAttribute("role","option");
    button.setAttribute("aria-selected",String(replacementSelected?.id===option.id));button.dataset.applicable=String(option.applyable);
    button.setAttribute("aria-description",option.applyable?option.reason:`Cannot apply: ${option.reason}`);
    button.dataset.compatible=String(option.compatibility!=="incompatible");
    button.innerHTML=`
      <span class="replacement-candidate-title">
        <span class="replacement-candidate-icon" aria-hidden="true">${escapeHtml(catalogItemIcon(option))}</span>
        <span>
          <strong>${escapeHtml(option.label)}</strong>
          <small>${escapeHtml(option.provider)} · ${escapeHtml(option.model)} · ${escapeHtml(option.component_id??option.kind)}</small>
          ${option.detail?`<small>${escapeHtml(option.detail)}</small>`:""}
          <small>${escapeHtml(option.reason)}</small>
        </span>
      </span>
      <span class="replacement-badge ${escapeHtml(option.readiness)}">${escapeHtml(replacementCompatibilityLabel(option))} · ${escapeHtml(option.readiness)}</span>
    `;
    button.onclick=()=>{replacementSelected=option;renderReplacementCandidates();byId("replacement-continue").disabled=!option.applyable;announce(`${option.label} selected. ${option.reason}`,!option.applyable);};
    button.onkeydown=event=>{
      if(!["ArrowDown","ArrowUp","Home","End"].includes(event.key))return;
      event.preventDefault();const buttons=[...byId("replacement-list").querySelectorAll(".replacement-candidate")],index=buttons.indexOf(button);
      const next=event.key==="Home"?0:event.key==="End"?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+(event.key==="ArrowDown"?1:-1)));
      buttons[next]?.focus();
    };
    fragment.append(button);
  }
  if(filtered.length>visible.length){
    const more=document.createElement("button");more.type="button";more.textContent=`Show ${Math.min(100,filtered.length-visible.length)} more`;
    more.onclick=()=>{replacementRenderLimit+=100;renderReplacementCandidates();};fragment.append(more);
  }
  byId("replacement-list").replaceChildren(fragment);
  const suffix=filtered.length>visible.length?`; ${visible.length} rendered at once for responsiveness`:"";
  byId("replacement-results").textContent=`${filtered.length} candidate${filtered.length===1?"":"s"} from backend discovery${suffix}.`;
  if(!filtered.length)byId("replacement-list").innerHTML='<p class="muted">No candidates match these backend-derived filters.</p>';
}

function closeReplacementPicker(message="Replacement cancelled; the graph was not changed."){
  replacementPreviewGeneration++;replacementPlan=null;
  if(byId("replacement-dialog").open)byId("replacement-dialog").close();
  replacementReturnFocus?.focus?.();announce(message);
}

function replacementInput(name,spec){
  const label=document.createElement("label");label.textContent=`${spec.title??name} *`;
  let input;if(spec.enum){input=document.createElement("select");input.replaceChildren(...spec.enum.map(value=>new Option(String(value),JSON.stringify(value))));}
  else{input=document.createElement(spec.type==="string"&&spec.format==="multiline"?"textarea":"input");
    if(input.tagName==="INPUT")input.type=["number","integer"].includes(spec.type)?"number":spec.type==="boolean"?"checkbox":"text";
  }
  input.dataset.replacementConfig=name;input.dataset.schemaType=spec.type??"string";
  if(spec.minimum!=null)input.min=spec.minimum;if(spec.maximum!=null)input.max=spec.maximum;
  input.oninput=()=>{readReplacementOverrides();refreshReplacementPlan();};input.onchange=input.oninput;
  label.append(input);return label;
}

function readReplacementOverrides(){
  const next={};
  for(const input of byId("replacement-required-config").querySelectorAll("[data-replacement-config]")){
    if(input.type==="checkbox")next[input.dataset.replacementConfig]=input.checked;
    else if(!input.value.trim())continue;
    else if(["number","integer"].includes(input.dataset.schemaType))next[input.dataset.replacementConfig]=Number(input.value);
    else if(["array","object"].includes(input.dataset.schemaType)){try{next[input.dataset.replacementConfig]=JSON.parse(input.value);}catch{continue;}}
    else if(input.tagName==="SELECT"){try{next[input.dataset.replacementConfig]=JSON.parse(input.value);}catch{next[input.dataset.replacementConfig]=input.value;}}
    else next[input.dataset.replacementConfig]=input.value;
  }
  replacementOverrides=next;
}

function renderReplacementImpact(){
  const plan=replacementPlan,candidate=replacementSelected;if(!plan||!candidate)return;
  byId("replacement-review-summary").textContent=`${replacementCompatibilityLabel(candidate)}: ${candidate.reason}`;
  const wiring=plan.edge_changes.map(change=>`${change.edge_id}: ${change.state}${change.to?` (${change.from} → ${change.to})`:` (${change.from})`}`);
  wiring.push(...plan.sink_changes.map(change=>`Selected sink: ${change.state}${change.to?` (${change.from} → ${change.to})`:` (${change.from})`}`));
  if(!wiring.length)wiring.push("No connected edges or selected sinks are affected.");
  byId("replacement-wiring-impact").replaceChildren(...wiring.map(text=>{const item=document.createElement("li");item.textContent=text;return item;}));
  const config=plan.config_changes.map(change=>`${change.field}: ${change.state}. ${change.reason}`);
  if(!config.length)config.push("No configuration fields change.");
  byId("replacement-config-impact").replaceChildren(...config.map(text=>{const item=document.createElement("li");item.textContent=text;return item;}));
  const needs=plan.config_changes.filter(change=>["requires_input","invalid"].includes(change.state));
  const existing=new Set([...byId("replacement-required-config").querySelectorAll("[data-replacement-config]")].map(input=>input.dataset.replacementConfig));
  for(const change of needs)if(change.field&&!existing.has(change.field)){
    const spec=candidate.schema?.properties?.[change.field]??{};byId("replacement-required-config").append(replacementInput(change.field,spec));
  }
  byId("replacement-required-config").hidden=!needs.length;
  byId("replacement-lossy-row").hidden=!plan.lossy;byId("replacement-lossy-ack").checked=plan.lossless||byId("replacement-lossy-ack").checked;
  const diagnostics=plan.introduced_diagnostics??plan.validation?.diagnostics??[];
  renderDiagnostics(byId("replacement-diagnostics"),diagnostics);
  if(!diagnostics.length&&!plan.validation?.valid&&(plan.validation?.diagnostics?.length??0)>0){
    const note=document.createElement("p");note.className="muted";
    note.textContent=`The draft still has ${plan.validation.diagnostics.length} existing diagnostic${plan.validation.diagnostics.length===1?"":"s"}; this replacement adds none.`;
    byId("replacement-diagnostics").append(note);
  }
  const error=plan.blocking.map(item=>item.message).filter(Boolean).join(" ");
  byId("replacement-error").hidden=!error;byId("replacement-error").textContent=error;
  updateReplacementApply();
}

function updateReplacementApply(){
  const acknowledged=!replacementPlan?.lossy||byId("replacement-lossy-ack").checked;
  byId("replacement-apply").disabled=!replacementPlan?.applyable||!acknowledged;
}

async function refreshReplacementPlan(){
  if(!replacementSelected)return;
  const generation=++replacementPreviewGeneration;
  replacementPlan=planNodeReplacement(pipeline,selectedNode,replacementSelected,catalog,{
    useDefaults:byId("replacement-use-defaults").checked,overrides:replacementOverrides,catalogRevision:discovery.revision,
  });
  renderReplacementImpact();
  if(replacementPlan.blocking.some(item=>item.code==="replacement.incompatible"||item.code==="replacement.edge_unmapped"||item.code==="replacement.sink_unmapped"))return;
  byId("replacement-error").hidden=false;byId("replacement-error").textContent="Validating the replacement preview with the backend…";
  try{
    const [baseline,report]=await Promise.all([
      request("/api/pipeline/validate",jsonOptions("POST",pipeline)),
      request("/api/pipeline/validate",jsonOptions("POST",replacementPlan.preview_graph)),
    ]);
    if(generation!==replacementPreviewGeneration)return;
    replacementPlan=attachReplacementValidation(replacementPlan,report,baseline);renderReplacementImpact();
  }catch(error){
    if(generation!==replacementPreviewGeneration)return;
    replacementPlan=attachReplacementValidation(replacementPlan,{valid:false,diagnostics:[{message:`Preview validation could not complete: ${error.message}`} ]});renderReplacementImpact();
  }
}

function showReplacementReview(){
  if(!replacementSelected?.applyable)return;
  byId("replacement-picker-controls").hidden=true;byId("replacement-list").hidden=true;byId("replacement-review").hidden=false;
  byId("replacement-back").hidden=false;byId("replacement-continue").hidden=true;byId("replacement-apply").hidden=false;
  byId("replacement-required-config").replaceChildren();byId("replacement-use-defaults").checked=false;byId("replacement-lossy-ack").checked=false;
  refreshReplacementPlan();byId("replacement-review-title").focus?.();
}

function applyReplacement(){
  try{
    pipeline=commitReplacement(pipeline,replacementPlan,discovery.revision,editHistory,selectionState());
    const label=replacementSelected.label;replaceNodeSelection(replacementPlan.node_id);
    replacementPreviewGeneration++;byId("replacement-dialog").close();renderGraph();updateEditControls();scheduleValidation();
    announce(`Replaced with ${label}. Undo is available.`);
  }catch(error){byId("replacement-error").hidden=false;byId("replacement-error").textContent=error.message;announce(error.message,true);}
}

function undoGraphEdit(){
  const result=undoEdit(editHistory);if(!result)return;
  pipeline=result.pipeline;applySelectionState(result.selection);renderGraph();restoreFocus(result.focus);updateEditControls();scheduleValidation();announce(`${result.label} undone.`);
}
function redoGraphEdit(){
  const result=redoEdit(editHistory);if(!result)return;
  pipeline=result.pipeline;applySelectionState(result.selection);renderGraph();restoreFocus(result.focus);updateEditControls();scheduleValidation();announce(`${result.label} redone.`);
}

function scheduleValidation(delay=180){
  clearTimeout(validationTimer);validationTimer=setTimeout(validateRemote,delay);
}
async function validateRemote(){
  const generation=++validationGeneration;
  try{const report=await request("/api/pipeline/validate",jsonOptions("POST",pipeline));if(generation!==validationGeneration)return;validation=report;
    byId("validation").textContent=report.valid?"Ready to compile and execute":`${report.diagnostics.length} graph diagnostic${report.diagnostics.length===1?"":"s"}: ${report.diagnostics[0]?.message??""}`;
    byId("validation").dataset.state=report.valid?"valid":"invalid";renderGraph();
  }catch(error){if(generation===validationGeneration){byId("validation").textContent=error.message;byId("validation").dataset.state="invalid";}}
}

function syncName(){
  const name=byId("pipeline-name").value.trim()||"Untitled pipeline";
  if(name!==pipeline.metadata.name)performGraphEdit("Rename graph",()=>{pipeline.metadata.name=name;touch(pipeline);});
}
async function saveGraph(){
  syncName();
  if(pipeline.graph_id.startsWith("starter:")){
    pipeline.graph_id=`pipeline:${globalThis.crypto?.randomUUID?.()??Date.now()}`;
    pipeline.revision=1;
  }
  const saved=await request(`/api/pipeline/graphs/${encodeURIComponent(pipeline.graph_id)}`,jsonOptions("PUT",pipeline));pipeline=saved.document??saved;
  history.replaceState({graph_id:pipeline.graph_id},"",graphRoute(pipeline.graph_id));
  loadGraph(pipeline,{preserveHistory:true});announce(`Saved ${pipeline.metadata.name} revision ${pipeline.revision} through the backend.`);
}
async function showOpen(){
  const {graphs}=await request("/api/pipeline/graphs");byId("saved-graphs").replaceChildren(...graphs.map(summary=>{const button=document.createElement("button");button.textContent=`${summary.name} · revision ${summary.revision}`;button.onclick=async()=>{const value=await request(`/api/pipeline/graphs/${encodeURIComponent(summary.graph_id)}`);loadGraph(value.document??value);byId("open-dialog").close();announce(`Opened ${summary.name}.`);};return button;}));byId("open-dialog").showModal();
}
async function shareGraph(){
  await saveGraph();const url=new URL(graphRoute(pipeline.graph_id),location.href).href;
  byId("share-url").value=url;byId("share-json").value=JSON.stringify(pipeline,null,2);byId("share-dialog").showModal();
}

function updateRuntimeStateFromEvent(event) {
  if(event?.artifact)addRunArtifact(event.artifact);
  if (event?.status) setRunState({status: event.status});
  if (event?.run_id) {
    if (runState.runId !== event.run_id) setRunState({runId: event.run_id, startedAt: Date.now(), status: event.status ?? runState.status, elapsedMs: 0});
    byId("run-context").hidden = false;
    byId("run-tracks-link").href = `/runs/${encodeURIComponent(event.run_id)}/tracks`;
  }
  updateNodeRuntimeState(event);
  updateEdgeRuntimeState(event);
  if(!runtimeRenderRequested){
    runtimeRenderRequested=true;
    requestAnimationFrame(()=>{
      runtimeRenderRequested=false;
      patchCanvas?.render();
    });
  }
}

async function runGraph() {
  if (isRunActive()) {
    announce("A transport is already active. Stop it before starting again.", true);
    return;
  }
  syncName();
  byId("run-events").replaceChildren();
  setRunArtifacts();
  clearRuntimeActivity();
  setRunState({
    status: "preparing",
    runId: null,
    startedAt: Date.now(),
    elapsedMs: 0,
  });
  startRunTransportClock();
  try {
    const response = await fetch("/api/pipeline/run", jsonOptions("POST", pipeline));
    if (!response.ok) {
      let value = {};
      try { value = await response.json(); } catch {
        const raw = await response.text();
        if (raw) value = {error: raw};
      }
      throw new Error(value.validation?.diagnostics?.map(item => item.message).join(" ") ?? value.error ?? "Run rejected");
    }
    const reader = response.body?.getReader();
    if (!reader) throw new Error("Run stream was not returned by the server.");
    await consumeNdjson(reader, event => {
      updateRuntimeStateFromEvent(event);
      renderRunEvent(event);
    });
    if (runState.runId) {
      try {
        await refreshRunStateFromServer(runState.runId);
      } catch {}
      if (runState.status === "completed") announce("Graph run completed with streamed lifecycle evidence.");
      else announce("Run finished; monitor the track record for final status.");
    } else {
      setRunState({status: "completed"});
      announce("Graph run completed.");
    }
  } catch (error) {
    announce(error.message, true);
    if (runState.runId) await refreshRunStateFromServer().catch(() => {});
    setRunState({status: "failed"});
  } finally {
    stopRunTransportClock();
  }
}

async function stopRunGraph() {
  if (!isRunActive() || !runState.runId) {
    announce("No active transport to stop.", true);
    return;
  }
  setRunState({status: "stopping"});
  try {
    await request(`/api/pipeline/runs/${encodeURIComponent(runState.runId)}/stop`, {method: "POST"});
    announce("Stop requested; runtime is cancelling.");
    refreshRunStateFromServer().catch(() => {});
  } catch (error) {
    announce(`Stop request failed: ${error.message}`, true);
    await refreshRunStateFromServer();
  }
}

async function panicRunGraph() {
  if (!isRunActive() || !runState.runId) {
    announce("No active transport to panic.", true);
    return;
  }
  setRunState({status: "stopping"});
  try {
    await request(`/api/pipeline/runs/${encodeURIComponent(runState.runId)}/panic`, {method: "POST"});
    announce("Panic requested; runtime is aborting immediately.");
    refreshRunStateFromServer().catch(() => {});
  } catch (error) {
    announce(`Panic request failed: ${error.message}`, true);
    await refreshRunStateFromServer();
  }
}

function renderRunEvent(event){
  if (event.kind === "cancelled" && event.status === "stopping") {
    setRunState({status: "stopping", runId: runState.runId ?? event.run_id ?? null});
  }
  const item=document.createElement("li");
  item.className=event.kind;
  const output=event.output?` · ${event.output.port_id}=${JSON.stringify(event.output.value)}`:"";
  item.textContent=`${event.node_id} · ${event.kind}${event.elapsed_ms==null?"":` · ${event.elapsed_ms} ms`}${output}${event.detail?` · ${event.detail}`:""}`;
  byId("run-events").append(item);
  while(byId("run-events").children.length>RUN_EVENT_LIMIT)byId("run-events").firstElementChild.remove();
  item.scrollIntoView({block:"nearest"});
  if(event.node_id&&pipeline.nodes.some(node=>node.id===event.node_id)){cy.nodes().removeClass("compatible");cy.getElementById(event.node_id).addClass("compatible");}
}

function selectedNodeIds(){return selectedNodes.size?[...selectedNodes]:(selectedNode?[selectedNode]:[]);}
function selectedEdgeIds(){return selectedEdges.size?[...selectedEdges]:(selectedEdge?[selectedEdge]:[]);}
async function copySelectedObjects(){
  const ids=selectedNodeIds();if(!ids.length)return announce("Select at least one module to copy.",true);
  graphClipboard=copyGraphSelection(pipeline,ids);pasteGeneration=0;
  try{await navigator.clipboard?.writeText(JSON.stringify(graphClipboard));}catch{}
  announce(`Copied ${graphClipboard.nodes.length} module${graphClipboard.nodes.length===1?"":"s"} and ${graphClipboard.edges.length} internal cable${graphClipboard.edges.length===1?"":"s"}.`);
  return graphClipboard;
}
async function cutSelectedObjects(){
  const copied=await copySelectedObjects();if(!copied)return;
  deleteSelectedObjects("Cut selection");
}
function pasteSelectedObjects(){
  if(!graphClipboard?.nodes?.length)return announce("Copy modules before pasting.",true);
  pasteGeneration+=1;
  const ids=performGraphEdit("Paste selection",()=>pasteGraphSelection(pipeline,graphClipboard,{x:36*pasteGeneration,y:36*pasteGeneration}),result=>{
    selectedNodes=new Set(result);selectedNode=result.at(-1)??null;selectedEdges=new Set();selectedEdge=null;
  });
  renderGraph();scheduleValidation();announce(`Pasted ${ids.length} module${ids.length===1?"":"s"} with fresh identities.`);
}
function duplicateSelectedObjects(){
  const ids=selectedNodeIds();if(!ids.length)return announce("Select at least one module to duplicate.",true);
  graphClipboard=copyGraphSelection(pipeline,ids);pasteGeneration=1;
  const pasted=performGraphEdit("Duplicate selection",()=>pasteGraphSelection(pipeline,graphClipboard,{x:36,y:36}),result=>{
    selectedNodes=new Set(result);selectedNode=result.at(-1)??null;selectedEdges=new Set();selectedEdge=null;
  });
  renderGraph();scheduleValidation();announce(`Duplicated ${pasted.length} module${pasted.length===1?"":"s"}.`);
}
function deleteSelectedObjects(label="Delete selection"){
  const nodes=selectedNodeIds(),edges=selectedEdgeIds();
  if(!nodes.length&&!edges.length)return;
  performGraphEdit(label,()=>deleteGraphSelection(pipeline,nodes,edges),clearSelectionState);
  renderGraph();scheduleValidation();announce(`Deleted ${nodes.length} module${nodes.length===1?"":"s"} and ${edges.length} selected cable${edges.length===1?"":"s"}.`);
}
function arrangeSelection(label,operation,minimum=2){
  const ids=selectedNodeIds();if(ids.length<minimum)return announce(`Select at least ${minimum} modules to ${label.toLowerCase()}.`,true);
  performGraphEdit(label,()=>operation(ids));renderGraph();scheduleValidation();announce(`${label} applied to ${ids.length} modules.`);
}
function fitSelectedObjects(){
  const ids=[...selectedNodes,...selectedEdges];if(!ids.length)return announce("Select objects to fit.",true);
  const collection=cy.collection(ids.map(id=>cy.getElementById(id)));cy.fit(collection,48);
  announce(`Fit ${ids.length} selected object${ids.length===1?"":"s"} in view.`);
}
function openOrganizationDialog(mode){
  organizationMode=mode;organizationBoundary=mode==="subpatch"?subpatchBoundaryPorts(pipeline,[...selectedNodes],discovery):[];
  const labels={frame:"Frame selected nodes without changing execution.",note:"Place a freeform presentation note at the canvas center.",subpatch:"Review every external port before creating an embedded semantic subpatch."};
  byId("organization-title").textContent={frame:"Create frame",note:"Add note",subpatch:"Create subpatch"}[mode];
  byId("organization-context").textContent=labels[mode];byId("organization-text").value={frame:"Section",note:"Note",subpatch:"Subpatch"}[mode];
  byId("organization-color-row").hidden=mode==="subpatch";byId("organization-error").hidden=true;
  byId("organization-port-review").replaceChildren(...organizationBoundary.map((port,index)=>{
    const label=document.createElement("label"),checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.checked=true;checkbox.dataset.portIndex=String(index);
    label.append(checkbox,document.createTextNode(`${port.direction} ${port.label} · ${readableType(port.value_type)} → ${port.internal.node_id}.${port.internal.port_id}`));return label;
  }));
  if(mode==="subpatch"&&!selectedNodes.size){byId("organization-error").textContent="Select at least one runtime node.";byId("organization-error").hidden=false;}
  byId("organization-dialog").showModal();byId("organization-text").focus();
}
function applyOrganization(){
  try{
    const title=byId("organization-text").value.trim();
    if(!title)throw new Error("Enter a title or note.");
    performGraphEdit({frame:"Create frame",note:"Add note",subpatch:"Create subpatch"}[organizationMode],()=>{
      if(organizationMode==="frame")return createFrame(pipeline,[...selectedNodes],{title,color:byId("organization-color").value});
      if(organizationMode==="note")return addNote(pipeline,{text:title,position:canvasCenterPoint(),color:byId("organization-color").value});
      const reviewed=[...byId("organization-port-review").querySelectorAll("[data-port-index]")].filter(input=>input.checked).map(input=>organizationBoundary[Number(input.dataset.portIndex)]);
      if(reviewed.length!==organizationBoundary.length)throw new Error("Every current external boundary must be explicitly reviewed and exposed.");
      return createEmbeddedSubpatch(pipeline,[...selectedNodes],{title,exposed_ports:reviewed,parent_subpatch_id:activeSubpatchId},discovery);
    });
    byId("organization-dialog").close();renderGraph();scheduleValidation();announce(`${title} created as ${organizationMode==="subpatch"?"an embedded semantic subpatch":"presentation organization"}.`);
  }catch(error){byId("organization-error").textContent=error.message;byId("organization-error").hidden=false;}
}

function deleteSelectedNode(){deleteSelectedObjects();}
function disableSelectedNode(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;
  try{
    updateNodeDisabledState(node.id,!node.disabled);
  }catch(error){
    announce(error.message,true);
  }
}

byId("palette-search").oninput=renderPalette;
byId("template").onchange=event=>{const starter=starters.find(graph=>graph.graph_id===event.target.value);if(starter)loadGraph(starter);};
byId("new").onclick=()=>loadGraph(createPipeline());byId("save").onclick=()=>saveGraph().catch(error=>announce(error.message,true));byId("open").onclick=()=>showOpen().catch(error=>announce(error.message,true));
byId("duplicate-graph").onclick=()=>{syncName();const copy=structuredClone(pipeline);copy.graph_id=`pipeline:${globalThis.crypto?.randomUUID?.()??Date.now()}`;copy.revision=1;copy.metadata.name+= " copy";loadGraph(copy);announce("Created an independent graph copy. Save to persist it.");};
byId("share").onclick=()=>shareGraph().catch(error=>announce(error.message,true));byId("copy-share").onclick=()=>navigator.clipboard.writeText(byId("share-url").value).then(()=>announce("Share URL copied."));
byId("fit").onclick=()=>cy.fit(undefined,40);byId("run").onclick=runGraph;byId("stop").onclick=()=>stopRunGraph().catch(error=>announce(error.message,true));byId("panic").onclick=()=>panicRunGraph().catch(error=>announce(error.message,true));
byId("undo").onclick=undoGraphEdit;byId("redo").onclick=redoGraphEdit;
byId("copy-selection").onclick=()=>copySelectedObjects();byId("cut-selection").onclick=()=>cutSelectedObjects();byId("paste-selection").onclick=pasteSelectedObjects;byId("duplicate-selection").onclick=duplicateSelectedObjects;
byId("delete-selection").onclick=()=>deleteSelectedObjects();
byId("align-horizontal").onclick=()=>arrangeSelection("Align top",ids=>alignGraphSelection(pipeline,ids,"y","start"));
byId("distribute-horizontal").onclick=()=>arrangeSelection("Distribute horizontally",ids=>distributeGraphSelection(pipeline,ids,"x"),3);
byId("tidy-selection").onclick=()=>arrangeSelection("Tidy selection",ids=>tidyGraphSelection(pipeline,ids));
byId("frame-selection").onclick=()=>openOrganizationDialog("frame");byId("note").onclick=()=>openOrganizationDialog("note");byId("create-subpatch").onclick=()=>openOrganizationDialog("subpatch");
byId("organization-apply").onclick=applyOrganization;
byId("cable-opacity").onchange=event=>{performGraphEdit("Set cable opacity",()=>{pipeline.presentation.global_cable_opacity=Number(event.target.value);touch(pipeline);});renderGraph();scheduleValidation();};
byId("focus-path").onclick=()=>{performGraphEdit("Toggle selected path focus",()=>{pipeline.presentation.selected_path_focus=!pipeline.presentation.selected_path_focus;touch(pipeline);});byId("focus-path").setAttribute("aria-pressed",String(pipeline.presentation.selected_path_focus));renderGraph();};
byId("snap-grid").onclick=()=>{snapToGrid=!snapToGrid;byId("snap-grid").setAttribute("aria-pressed",String(snapToGrid));byId("snap-grid").textContent=snapToGrid?"Snap on":"Snap off";announce(`Grid snapping ${snapToGrid?"enabled":"disabled"}.`);};
byId("fit-selection").onclick=fitSelectedObjects;
byId("pipeline-name").onchange=()=>{syncName();scheduleValidation();};
byId("duplicate").onclick=duplicateSelectedObjects;
byId("delete").onclick=deleteSelectedNode;byId("disable").onclick=disableSelectedNode;
byId("bypass").onclick=()=>{try{toggleNodeBypass(selectedNode);}catch(error){announce(error.message,true);}};
byId("replace").onclick=openReplacementPicker;
byId("apply-config").onclick=applyConfig;byId("cancel-connect").onclick=()=>{connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");};
byId("apply-edge").onclick=()=>{
  const edge=pipeline.edges.find(item=>item.id===selectedEdge);if(!edge)return;
  performGraphEdit("Edit cable",()=>{edge.capacity=Math.max(1,Number(byId("edge-capacity").value)||1);setCablePresentation(pipeline,edge.id,{...(pipeline.presentation.cables[edge.id]??{}),routing:byId("cable-routing").value,emphasized:byId("cable-emphasized").checked});});
  renderGraph();scheduleValidation();
};
byId("add-reroute").onclick=()=>{
  const edge=pipeline.edges.find(item=>item.id===selectedEdge);if(!edge)return;const current=pipeline.presentation.cables[edge.id]??{};
  performGraphEdit("Add cable reroute",()=>setCablePresentation(pipeline,edge.id,{...current,reroute_points:[...(current.reroute_points??[]),canvasCenterPoint()]}));
  renderGraph();scheduleValidation();announce("Presentation-only cable reroute point added.");
};
byId("delete-edge").onclick=()=>{
  const edgeId=selectedEdge;if(!edgeId)return;
  performGraphEdit("Delete cable",()=>removeEdge(pipeline,edgeId),()=>{selectedEdges.delete(edgeId);selectedEdge=[...selectedEdges].at(-1)??null;});
  renderGraph();scheduleValidation();announce("Edge deleted.");
};
byId("toggle-palette").onclick=()=>byId("palette-panel").classList.toggle("open");byId("toggle-inspector").onclick=()=>byId("inspector-panel").classList.toggle("open");
byId("replacement-search").oninput=()=>{replacementRenderLimit=100;renderReplacementCandidates();};
byId("replacement-provider").onchange=()=>{replacementRenderLimit=100;renderReplacementCandidates();};
byId("replacement-readiness").onchange=()=>{replacementRenderLimit=100;renderReplacementCandidates();};
byId("replacement-cancel").onclick=()=>closeReplacementPicker();
byId("replacement-continue").onclick=showReplacementReview;
byId("replacement-back").onclick=()=>{replacementPreviewGeneration++;showReplacementPickerStep();renderReplacementCandidates();byId("replacement-search").focus();};
byId("replacement-use-defaults").onchange=()=>refreshReplacementPlan();
byId("replacement-lossy-ack").onchange=updateReplacementApply;
byId("replacement-apply").onclick=applyReplacement;
byId("replacement-dialog").addEventListener("cancel",event=>{event.preventDefault();closeReplacementPicker();});
byId("replacement-dialog").addEventListener("click",event=>{if(event.target===byId("replacement-dialog"))closeReplacementPicker();});
byId("quick-add-search").oninput=renderQuickAdd;byId("quick-add-cancel").onclick=()=>closeQuickAdd();
byId("quick-add-dialog").addEventListener("cancel",event=>{event.preventDefault();closeQuickAdd();});
window.addEventListener("popstate",()=>{const requested=new URLSearchParams(location.search).get("subpatch");activeSubpatchId=pipeline?.subpatches?.some(item=>item.id===requested)?requested:null;renderGraph();});
document.onkeydown=event=>{
  if(event.key==="Escape"&&connecting){connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");}
  const editable=["INPUT","TEXTAREA","SELECT"].includes(event.target?.tagName)||event.target?.isContentEditable;
  if(editable)return;
  if(["Delete","Backspace"].includes(event.key)){event.preventDefault();deleteSelectedObjects();return;}
  if(!event.ctrlKey&&!event.metaKey&&event.key.toLowerCase()==="i"&&selectedEdge){
    event.preventDefault();openQuickAdd({kind:"insert_edge",edge_id:selectedEdge,position:canvasCenterPoint()});return;
  }
  if(!(event.ctrlKey||event.metaKey))return;
  const key=event.key.toLowerCase();
  if(event.code==="Space"){event.preventDefault();openQuickAdd({kind:"empty",position:canvasCenterPoint()});}
  else if(key==="z"){event.preventDefault();event.shiftKey?redoGraphEdit():undoGraphEdit();}
  else if(key==="y"){event.preventDefault();redoGraphEdit();}
  else if(key==="c"){event.preventDefault();copySelectedObjects();}
  else if(key==="x"){event.preventDefault();cutSelectedObjects();}
  else if(key==="v"){event.preventDefault();pasteSelectedObjects();}
  else if(key==="d"){event.preventDefault();duplicateSelectedObjects();}
  else if(key==="a"){
    event.preventDefault();selectedNodes=new Set(pipeline.nodes.map(node=>node.id));selectedEdges=new Set(pipeline.edges.map(edge=>edge.id));
    selectedNode=pipeline.nodes.at(-1)?.id??null;selectedEdge=null;renderGraph();announce(`Selected all ${selectedNodes.size+selectedEdges.size} graph objects.`);
  }
};

setRunState(runState);
discover().catch(error=>{byId("validation").textContent=`Discovery failed: ${error.message}`;byId("validation").dataset.state="invalid";announce(error.message,true);});
