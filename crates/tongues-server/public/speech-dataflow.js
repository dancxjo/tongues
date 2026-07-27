import {
  addNode,alignGraphSelection,applyNodeConfig,attachReplacementValidation,bypassNode,buildCatalog,commitReplacement,compatibleTargets,connectPorts,consumeNdjson,copyGraphSelection,createEditHistory,createPipeline,
  catalogEntryForNode,deleteGraphSelection,diagnosticsByTarget,distributeGraphSelection,ensureLayout,insertSubgraph,moveGraphSelection,nodeLabel,nodePosition,
  pasteGraphSelection,planNodeReplacement,portsFor,recordEdit,redoEdit,removeEdge,replacementCandidates,setNodePosition,tidyGraphSelection,touch,undoEdit,
} from "./speech-dataflow-model.mjs";
import {createPatchCanvas} from "./speech-patch-canvas.mjs";

const byId=id=>document.getElementById(id);
let discovery=null,catalog=[],starters=[],pipeline=null,cy=null,patchCanvas=null;
let selectedNode=null,selectedEdge=null,connecting=null,validation={valid:false,diagnostics:[]};
let selectedNodes=new Set(),selectedEdges=new Set(),graphClipboard=null,pasteGeneration=0,snapToGrid=false;
let validationGeneration=0,validationTimer=null,runController=null;
let editHistory=createEditHistory(),replacementOptions=[],replacementSelected=null,replacementPlan=null;
let replacementPreviewGeneration=0,replacementRenderLimit=100,replacementReturnFocus=null,replacementOverrides={};

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

async function discover(){
  [discovery,{graphs:starters}]=await Promise.all([request("/api/pipeline/catalog"),request("/api/pipeline/starters")]);
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

function initCanvas(){
  if(!globalThis.cytoscape)throw new Error("The patch-canvas library did not load. Check network access to cdn.jsdelivr.net.");
  cy=globalThis.cytoscape({
    container:byId("canvas"),elements:[],wheelSensitivity:.18,boxSelectionEnabled:true,selectionType:"additive",
    style:[
      {selector:"node",style:{
        "shape":"round-rectangle","width":228,"height":126,
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
}

function renderPalette(){
  const query=byId("palette-search").value.trim().toLowerCase(),groups=new Map();
  catalog.filter(item=>`${item.label} ${item.kind} ${item.detail} ${item.group}`.toLowerCase().includes(query))
    .forEach(item=>{if(!groups.has(item.group))groups.set(item.group,[]);groups.get(item.group).push(item);});
  byId("palette").replaceChildren(...[...groups].map(([group,items])=>{
    const details=document.createElement("details");details.className="palette-group";details.open=true;
    const summary=document.createElement("summary");summary.textContent=`${group} (${items.length})`;details.append(summary);
    const list=document.createElement("div");list.className="palette-list";
    items.forEach(item=>{const button=document.createElement("button");button.className="palette-node";button.dataset.readiness=item.readiness;
      button.innerHTML=`${escapeHtml(item.label)}<small>${escapeHtml(item.kind)} · ${escapeHtml(item.readiness)}</small>`;
      button.title=item.detail;button.onclick=()=>addCatalogNode(item);list.append(button);});
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
  renderGraph();selectNode(node.id);announce(`Added ${item.label}.`);
}

function loadGraph(graph,{preserveHistory=false}={}){
  const previousSelection=selectionState();
  pipeline=structuredClone(graph);pipeline.metadata.labels??={};ensureLayout(pipeline);
  if(!preserveHistory)editHistory=createEditHistory();
  if(preserveHistory)applySelectionState(previousSelection);
  else replaceNodeSelection(pipeline.nodes[0]?.id??null);
  connecting=null;
  byId("pipeline-name").value=pipeline.metadata.name;
  byId("graph-identity").textContent=pipeline.graph_id.startsWith("starter:")
    ?"Editing a configuration draft seeded from a backend template"
    :`Editing saved graph ${pipeline.graph_id}, revision ${pipeline.revision}`;
  document.title=`${pipeline.metadata.name} · Graph Studio · Tongues`;
  renderGraph();ensurePatchCanvas();patchCanvas.render();updateEditControls();scheduleValidation(0);
}

function ensurePatchCanvas(){
  if(patchCanvas)return;
  patchCanvas=createPatchCanvas({
    container:byId("canvas"),cy,
    getPipeline:()=>pipeline,getDiscovery:()=>discovery,getCatalog:()=>catalog,
    nodeLabel,getSelectedEdgeId:()=>selectedEdge,
    isEdgeSelected:id=>selectedEdges.has(id),
    diagnosticsByEdge:()=>diagnosticsByTarget(validation).edges,
    onSelectNode:selectNode,onSelectEdge:selectEdge,
    onGraphEdit:recordCompletedGraphEdit,
    onAnnounce:announce,
  });
}

function graphElements(){
  const grouped=diagnosticsByTarget(validation);
  const nodes=pipeline.nodes.map(node=>{
    const item=catalogEntryForNode(node,catalog);
    const ports=discovery.node_kinds?.[node.kind]?.ports??[];
    const kind=discovery.node_kinds?.[node.kind],theme=NODE_THEMES[item?.group]??FALLBACK_NODE_THEME;
    const inputs=nodePortSummary(node,ports.filter(port=>port.direction==="input"),"input");
    const outputs=nodePortSummary(node,ports.filter(port=>port.direction==="output"),"output");
    const label=[kind?.label??nodeLabel(node,catalog),"",inputs&&`IN   ${inputs}`,outputs&&`OUT  ${outputs}`].filter(line=>line!==false).join("\n");
    const classes=[item?.readiness&&item.readiness!=="ready"?"unavailable":"",grouped.nodes[node.id]?.length?"invalid":"",node.disabled||node.bypassed?"inactive":""].filter(Boolean).join(" ");
    return{group:"nodes",data:{id:node.id,label,accent:theme.accent,surface:theme.surface},position:nodePosition(pipeline,node.id),classes};
  });
  const edges=pipeline.edges.map(edge=>{
    const source=pipeline.nodes.find(node=>node.id===edge.from.node_id);
    const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
    return{group:"edges",data:{id:edge.id,source:edge.from.node_id,target:edge.to.node_id,type:readableType(port?.value_type)},classes:grouped.edges[edge.id]?.length?"invalid":""};
  });
  return[...nodes,...edges];
}

function renderGraph(){
  cy.elements().remove();cy.add(graphElements());
  [...selectedNodes,...selectedEdges].forEach(id=>cy.getElementById(id).select());
  renderOutline();renderInspector();patchCanvas?.render();byId("pipeline-name").value=pipeline.metadata.name;
}

function renderOutline(){
  byId("graph-outline").replaceChildren(...pipeline.nodes.map(node=>{
    const item=document.createElement("li"),button=document.createElement("button");
    button.textContent=nodeLabel(node,catalog);
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
  announce(`${selectedNodes.size+selectedEdges.size} object${selectedNodes.size+selectedEdges.size===1?"":"s"} selected. ${nodeLabel(pipeline.nodes.find(node=>node.id===id),catalog)} is ${selectedNodes.has(id)?"included":"not included"}.`);
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
  byId("node-title").textContent=nodeLabel(node,catalog);byId("node-detail").textContent=item?.detail??node.kind;
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
  if(connecting)byId("connection-source").textContent=`Connecting ${nodeLabel(pipeline.nodes.find(n=>n.id===connecting.node_id),catalog)}.${connecting.port_id} (${connecting.value_type})`;
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
  const entries=Object.entries(properties);
  if(!entries.length){byId("config-fields").innerHTML='<span class="muted">This node has no configurable fields.</span>';return;}
  byId("config-fields").replaceChildren(...entries.map(([name,spec])=>{
    const label=document.createElement("label");label.textContent=`${spec.title??name}${required.has(name)?" *":""}`;
    let input;if(spec.enum){input=document.createElement("select");input.replaceChildren(...spec.enum.map(value=>new Option(String(value),String(value))));}
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
    renderGraph();scheduleValidation();announce("Configuration applied.");
  }catch(error){announce(error.message,true);}
}

function renderEdgeInspector(edge){
  const source=pipeline.nodes.find(node=>node.id===edge.from.node_id),target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
  const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
  byId("edge-title").textContent=`${nodeLabel(source,catalog)} → ${nodeLabel(target,catalog)}`;
  byId("edge-type").textContent=`${edge.from.port_id} → ${edge.to.port_id} · ${port?.value_type??"unknown"}`;byId("edge-capacity").value=edge.capacity;
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
  byId("replacement-context").textContent=`Replacing ${nodeLabel(node,catalog)} (${node.component_id??node.kind}). The graph is unchanged until Apply.`;
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
    &&(!query||`${option.label} ${option.provider} ${option.model} ${option.component_id} ${option.detail} ${option.reason}`.toLowerCase().includes(query)));
}

function renderReplacementCandidates(){
  const filtered=filteredReplacementOptions(),visible=filtered.slice(0,replacementRenderLimit),fragment=document.createDocumentFragment();
  for(const option of visible){
    const button=document.createElement("button");button.type="button";button.className="replacement-candidate";button.setAttribute("role","option");
    button.setAttribute("aria-selected",String(replacementSelected?.id===option.id));button.dataset.applicable=String(option.applyable);
    button.setAttribute("aria-description",option.applyable?option.reason:`Cannot apply: ${option.reason}`);
    button.dataset.compatible=String(option.compatibility!=="incompatible");
    button.innerHTML=`<span><strong>${escapeHtml(option.label)}</strong><small>${escapeHtml(option.provider)} · ${escapeHtml(option.model)} · ${escapeHtml(option.component_id??option.kind)}</small>${option.detail?`<small>${escapeHtml(option.detail)}</small>`:""}<small>${escapeHtml(option.reason)}</small></span><span class="replacement-badge ${escapeHtml(option.readiness)}">${escapeHtml(replacementCompatibilityLabel(option))} · ${escapeHtml(option.readiness)}</span>`;
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
  const diagnostics=plan.validation?.diagnostics??[];renderDiagnostics(byId("replacement-diagnostics"),diagnostics);
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
    const report=await request("/api/pipeline/validate",jsonOptions("POST",replacementPlan.preview_graph));
    if(generation!==replacementPreviewGeneration)return;
    replacementPlan=attachReplacementValidation(replacementPlan,report);renderReplacementImpact();
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

async function runGraph(){
  if(runController)return;syncName();runController=new AbortController();byId("run").disabled=true;byId("cancel").disabled=false;byId("run-events").replaceChildren();
  try{
    const response=await fetch("/api/pipeline/run",{...jsonOptions("POST",pipeline),signal:runController.signal});
    if(!response.ok){const value=await response.json();throw new Error(value.validation?.diagnostics?.map(item=>item.message).join(" ")??value.error??"Run rejected");}
    await consumeNdjson(response.body.getReader(),event=>{
      renderRunEvent(event);
      if(event.run_id){
        byId("run-context").hidden=false;
        byId("run-tracks-link").href=`/runs/${encodeURIComponent(event.run_id)}/tracks`;
      }
    });
    announce("Graph run completed with streamed lifecycle evidence.");
  }catch(error){if(error.name==="AbortError"){renderRunEvent({kind:"cancelled",node_id:"graph",detail:"Cancelled by operator"});announce("Graph run cancelled.");}else announce(error.message,true);}
  finally{runController=null;byId("run").disabled=false;byId("cancel").disabled=true;}
}
function renderRunEvent(event){const item=document.createElement("li");item.className=event.kind;const output=event.output?` · ${event.output.port_id}=${JSON.stringify(event.output.value)}`:"";item.textContent=`${event.node_id} · ${event.kind}${event.elapsed_ms==null?"":` · ${event.elapsed_ms} ms`}${output}${event.detail?` · ${event.detail}`:""}`;byId("run-events").append(item);item.scrollIntoView({block:"nearest"});if(event.node_id&&pipeline.nodes.some(node=>node.id===event.node_id)){cy.nodes().removeClass("compatible");cy.getElementById(event.node_id).addClass("compatible");}}

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
function deleteSelectedNode(){deleteSelectedObjects();}
function disableSelectedNode(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;
  const disabling=!node.disabled;
  performGraphEdit(disabling?"Disable node":"Enable node",()=>{
    if(node.disabled)node.disabled=false;
    else{
      pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==selectedNode&&edge.to.node_id!==selectedNode);
      pipeline.selected_sinks=pipeline.selected_sinks.filter(sink=>sink.node_id!==selectedNode);node.disabled=true;
    }
    touch(pipeline);
  });
  announce(disabling?"Node disabled and removed from execution; its connections were removed explicitly.":"Node enabled; reconnect any relationships it needs.");
  renderGraph();scheduleValidation();
}

byId("palette-search").oninput=renderPalette;
byId("template").onchange=event=>{const starter=starters.find(graph=>graph.graph_id===event.target.value);if(starter)loadGraph(starter);};
byId("new").onclick=()=>loadGraph(createPipeline());byId("save").onclick=()=>saveGraph().catch(error=>announce(error.message,true));byId("open").onclick=()=>showOpen().catch(error=>announce(error.message,true));
byId("duplicate-graph").onclick=()=>{syncName();const copy=structuredClone(pipeline);copy.graph_id=`pipeline:${globalThis.crypto?.randomUUID?.()??Date.now()}`;copy.revision=1;copy.metadata.name+= " copy";loadGraph(copy);announce("Created an independent graph copy. Save to persist it.");};
byId("share").onclick=()=>shareGraph().catch(error=>announce(error.message,true));byId("copy-share").onclick=()=>navigator.clipboard.writeText(byId("share-url").value).then(()=>announce("Share URL copied."));
byId("fit").onclick=()=>cy.fit(undefined,40);byId("run").onclick=runGraph;byId("cancel").onclick=()=>runController?.abort();
byId("undo").onclick=undoGraphEdit;byId("redo").onclick=redoGraphEdit;
byId("copy-selection").onclick=()=>copySelectedObjects();byId("cut-selection").onclick=()=>cutSelectedObjects();byId("paste-selection").onclick=pasteSelectedObjects;byId("duplicate-selection").onclick=duplicateSelectedObjects;
byId("delete-selection").onclick=()=>deleteSelectedObjects();
byId("align-horizontal").onclick=()=>arrangeSelection("Align top",ids=>alignGraphSelection(pipeline,ids,"y","start"));
byId("distribute-horizontal").onclick=()=>arrangeSelection("Distribute horizontally",ids=>distributeGraphSelection(pipeline,ids,"x"),3);
byId("tidy-selection").onclick=()=>arrangeSelection("Tidy selection",ids=>tidyGraphSelection(pipeline,ids));
byId("snap-grid").onclick=()=>{snapToGrid=!snapToGrid;byId("snap-grid").setAttribute("aria-pressed",String(snapToGrid));byId("snap-grid").textContent=snapToGrid?"Snap on":"Snap off";announce(`Grid snapping ${snapToGrid?"enabled":"disabled"}.`);};
byId("fit-selection").onclick=fitSelectedObjects;
byId("pipeline-name").onchange=()=>{syncName();scheduleValidation();};
byId("duplicate").onclick=duplicateSelectedObjects;
byId("delete").onclick=deleteSelectedNode;byId("disable").onclick=disableSelectedNode;
byId("bypass").onclick=()=>{try{performGraphEdit("Bypass node",()=>bypassNode(pipeline,selectedNode,discovery));renderGraph();scheduleValidation();announce("Node bypassed by explicit compatible rewiring.");}catch(error){announce(error.message,true);}};
byId("replace").onclick=openReplacementPicker;
byId("apply-config").onclick=applyConfig;byId("cancel-connect").onclick=()=>{connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");};
byId("apply-edge").onclick=()=>{
  const edge=pipeline.edges.find(item=>item.id===selectedEdge);if(!edge)return;
  performGraphEdit("Edit cable capacity",()=>{edge.capacity=Math.max(1,Number(byId("edge-capacity").value)||1);touch(pipeline);});
  renderGraph();scheduleValidation();
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
document.onkeydown=event=>{
  if(event.key==="Escape"&&connecting){connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");}
  const editable=["INPUT","TEXTAREA","SELECT"].includes(event.target?.tagName)||event.target?.isContentEditable;
  if(editable)return;
  if(["Delete","Backspace"].includes(event.key)){event.preventDefault();deleteSelectedObjects();return;}
  if(!(event.ctrlKey||event.metaKey))return;
  const key=event.key.toLowerCase();
  if(key==="z"){event.preventDefault();event.shiftKey?redoGraphEdit():undoGraphEdit();}
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

discover().catch(error=>{byId("validation").textContent=`Discovery failed: ${error.message}`;byId("validation").dataset.state="invalid";announce(error.message,true);});
