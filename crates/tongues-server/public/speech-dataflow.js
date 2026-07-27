import {
  addNode,bypassNode,buildCatalog,compatibleTargets,connectPorts,consumeNdjson,createPipeline,
  diagnosticsByTarget,duplicateNode,ensureLayout,insertSubgraph,nodeLabel,nodePosition,
  portsFor,removeEdge,removeNode,replaceNode,setNodePosition,touch,
} from "./speech-dataflow-model.mjs";

const byId=id=>document.getElementById(id);
let discovery=null,catalog=[],starters=[],pipeline=null,cy=null;
let selectedNode=null,selectedEdge=null,connecting=null,validation={valid:false,diagnostics:[]};
let validationGeneration=0,validationTimer=null,runController=null;

function announce(text,error=false){byId("status").textContent=text;byId("status").classList.toggle("error",error);}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
async function request(path,options={}){
  const response=await fetch(path,options),text=await response.text();
  let value={};try{value=text?JSON.parse(text):{};}catch{value={error:text};}
  if(!response.ok)throw new Error(value.error??value.validation?.diagnostics?.map(item=>item.message).join(" ")??`${path}: ${response.status}`);
  return value;
}
function jsonOptions(method,value){return{method,headers:{"Content-Type":"application/json"},body:JSON.stringify(value)};}

async function discover(){
  [discovery,{graphs:starters}]=await Promise.all([request("/api/pipeline/catalog"),request("/api/pipeline/starters")]);
  catalog=buildCatalog(discovery);renderPalette();renderTemplates();
  byId("template").replaceChildren(...starters.map(graph=>new Option(graph.metadata.name,graph.graph_id)));
  const requestedStarter=new URLSearchParams(location.search).get("starter");
  const selectedStarter=starters.find(graph=>graph.graph_id===`starter:${requestedStarter}`)
    ??starters.find(graph=>graph.graph_id===requestedStarter)
    ??starters[0];
  initCanvas();loadGraph(selectedStarter??createPipeline());
  announce(`Loaded ${catalog.length} backend-discovered choices and ${starters.length} starter graphs.`);
}

function initCanvas(){
  if(!globalThis.cytoscape)throw new Error("The patch-canvas library did not load. Check network access to cdn.jsdelivr.net.");
  cy=globalThis.cytoscape({
    container:byId("canvas"),elements:[],wheelSensitivity:.18,
    style:[
      {selector:"node",style:{"shape":"round-rectangle","width":190,"height":82,"background-color":"#202c3b","border-width":2,"border-color":"#40536a","label":"data(label)","color":"#edf5ff","font-size":12,"text-wrap":"wrap","text-max-width":168,"text-valign":"center","text-halign":"center"}},
      {selector:"node:selected",style:{"border-color":"#76e2ce","border-width":4}},
      {selector:"node.unavailable",style:{"border-color":"#ff8c91","border-style":"dashed"}},
      {selector:"node.invalid",style:{"border-color":"#ffc86b"}},
      {selector:"node.inactive",style:{"opacity":.45,"border-style":"dashed"}},
      {selector:"node.compatible",style:{"overlay-color":"#76e2ce","overlay-opacity":.18,"overlay-padding":10}},
      {selector:"edge",style:{"curve-style":"bezier","target-arrow-shape":"triangle","line-color":"#71869e","target-arrow-color":"#71869e","width":3,"label":"data(type)","font-size":9,"color":"#a6b7ca","text-background-color":"#0c1118","text-background-opacity":.85,"text-background-padding":2}},
      {selector:"edge:selected",style:{"line-color":"#76e2ce","target-arrow-color":"#76e2ce","width":5}},
      {selector:"edge.invalid",style:{"line-color":"#ffc86b","target-arrow-color":"#ffc86b","line-style":"dashed"}},
    ],
  });
  cy.on("tap","node",event=>selectNode(event.target.id()));
  cy.on("tap","edge",event=>selectEdge(event.target.id()));
  cy.on("tap",event=>{if(event.target===cy)clearSelection();});
  cy.on("dragfree","node",event=>{setNodePosition(pipeline,event.target.id(),event.target.position());touch(pipeline);scheduleValidation();renderOutline();});
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
    button.onclick=()=>{const ids=insertSubgraph(pipeline,graph,{x:cy.extent().x1+80,y:cy.extent().y1+80});renderGraph();selectNode(ids[0]);announce(`Inserted ${graph.metadata.name} as a reusable subgraph.`);};return button;
  }));
}

function addCatalogNode(item){
  const center=cy.extent(),node=addNode(pipeline,item,selectedNode,{x:(center.x1+center.x2)/2,y:(center.y1+center.y2)/2});
  renderGraph();selectNode(node.id);announce(`Added ${item.label}.`);
}

function loadGraph(graph){
  pipeline=structuredClone(graph);pipeline.metadata.labels??={};ensureLayout(pipeline);
  selectedNode=pipeline.nodes[0]?.id??null;selectedEdge=null;connecting=null;
  byId("pipeline-name").value=pipeline.metadata.name;renderGraph();scheduleValidation(0);
}

function graphElements(){
  const grouped=diagnosticsByTarget(validation);
  const nodes=pipeline.nodes.map(node=>{
    const item=catalog.find(entry=>entry.kind===node.kind&&entry.component_id===node.component_id);
    const ports=discovery.node_kinds?.[node.kind]?.ports??[];
    const inputs=ports.filter(port=>port.direction==="input").map(port=>`◀ ${port.label}: ${port.value_type}`);
    const outputs=ports.filter(port=>port.direction==="output").map(port=>`${port.label}: ${port.value_type} ▶`);
    const classes=[item?.readiness&&item.readiness!=="ready"?"unavailable":"",grouped.nodes[node.id]?.length?"invalid":"",node.disabled||node.bypassed?"inactive":""].filter(Boolean).join(" ");
    return{group:"nodes",data:{id:node.id,label:`${nodeLabel(node,catalog)}\n${inputs.slice(0,2).join(" · ")}${inputs.length>2?" …":""}\n${outputs.slice(0,2).join(" · ")}${outputs.length>2?" …":""}`},position:nodePosition(pipeline,node.id),classes};
  });
  const edges=pipeline.edges.map(edge=>{
    const source=pipeline.nodes.find(node=>node.id===edge.from.node_id);
    const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
    return{group:"edges",data:{id:edge.id,source:edge.from.node_id,target:edge.to.node_id,type:port?.value_type??"unknown"},classes:grouped.edges[edge.id]?.length?"invalid":""};
  });
  return[...nodes,...edges];
}

function renderGraph(){
  const selected=selectedNode??selectedEdge;
  cy.elements().remove();cy.add(graphElements());
  if(selected)cy.getElementById(selected).select();
  renderOutline();renderInspector();byId("pipeline-name").value=pipeline.metadata.name;
}

function renderOutline(){
  byId("graph-outline").replaceChildren(...pipeline.nodes.map(node=>{
    const item=document.createElement("li"),button=document.createElement("button");
    const position=nodePosition(pipeline,node.id)??{x:0,y:0};button.textContent=nodeLabel(node,catalog);
    button.onclick=()=>selectNode(node.id);button.onkeydown=event=>{
      const deltas={ArrowLeft:[-20,0],ArrowRight:[20,0],ArrowUp:[0,-20],ArrowDown:[0,20]};
      if(deltas[event.key]){event.preventDefault();const [x,y]=deltas[event.key];setNodePosition(pipeline,node.id,{x:position.x+x,y:position.y+y});touch(pipeline);renderGraph();cy.getElementById(node.id).select();}
      if(event.key==="Delete"){event.preventDefault();deleteSelectedNode();}
    };item.append(button);return item;
  }));
}

function selectNode(id){selectedNode=id;selectedEdge=null;cy.elements().unselect();cy.getElementById(id).select();renderInspector();announce(`Selected ${nodeLabel(pipeline.nodes.find(node=>node.id===id),catalog)}.`);}
function selectEdge(id){selectedEdge=id;selectedNode=null;cy.elements().unselect();cy.getElementById(id).select();renderInspector();const edge=pipeline.edges.find(item=>item.id===id);announce(`Selected connection from ${edge?.from.node_id}.${edge?.from.port_id} to ${edge?.to.node_id}.${edge?.to.port_id}.`);}
function clearSelection(){selectedNode=null;selectedEdge=null;connecting=null;cy.elements().unselect();renderInspector();}

function renderInspector(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode),edge=pipeline.edges.find(item=>item.id===selectedEdge);
  byId("empty-inspector").hidden=Boolean(node||edge);byId("node-inspector").hidden=!node;byId("edge-inspector").hidden=!edge;
  if(node)renderNodeInspector(node);if(edge)renderEdgeInspector(edge);
}

function renderNodeInspector(node){
  const item=catalog.find(entry=>entry.kind===node.kind&&entry.component_id===node.component_id);
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
  button.innerHTML=`<span class="port-dot"></span><span>${escapeHtml(port.label)}<small>${escapeHtml(port.value_type)} · ${escapeHtml(port.cardinality)}</small></span><span>${connected?"●":missing?"○":"·"}</span>`;
  const diagnostics=portDiagnostics;
  button.setAttribute("aria-label",`${port.direction} ${port.label}, type ${port.value_type}, ${connected?"connected":missing?"required and missing":"not connected"}${diagnostics.length?`. ${diagnostics.flatMap(item=>[item.message,...(item.suggestions??[])]).join(". ")}`:""}`);
  button.onclick=()=>{
    if(port.direction==="output"){connecting={node_id:node.id,port_id:port.id,value_type:port.value_type};highlightCompatible();renderInspector();announce(`Choose an input compatible with ${port.value_type}.`);}
    else if(connecting){try{connectPorts(pipeline,connecting.node_id,connecting.port_id,node.id,port.id,discovery);connecting=null;renderGraph();scheduleValidation();announce("Typed connection added.");}catch(error){announce(error.message,true);}}
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
    else{input=document.createElement(spec.type==="string"&&spec.format==="multiline"?"textarea":"input");input.type=spec.type==="number"||spec.type==="integer"?"number":spec.type==="boolean"?"checkbox":"text";if(spec.minimum!=null)input.min=spec.minimum;if(spec.maximum!=null)input.max=spec.maximum;}
    input.dataset.config=name;const value=node.config?.[name]??spec.default;
    if(input.type==="checkbox")input.checked=Boolean(value);else if(value!=null)input.value=typeof value==="object"?JSON.stringify(value):String(value);
    if(spec.description)input.title=spec.description;label.append(input);return label;
  }));
}

function applyConfig(){
  const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;
  try{for(const input of byId("config-fields").querySelectorAll("[data-config]")){let value=input.type==="checkbox"?input.checked:input.value;if(input.type==="number")value=Number(value);node.config[input.dataset.config]=value;}touch(pipeline);renderGraph();scheduleValidation();announce("Configuration applied.");}catch(error){announce(error.message,true);}
}

function renderEdgeInspector(edge){
  const source=pipeline.nodes.find(node=>node.id===edge.from.node_id),target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
  const port=portsFor(source,"output",discovery).find(item=>item.id===edge.from.port_id);
  byId("edge-title").textContent=`${nodeLabel(source,catalog)} → ${nodeLabel(target,catalog)}`;
  byId("edge-type").textContent=`${edge.from.port_id} → ${edge.to.port_id} · ${port?.value_type??"unknown"}`;byId("edge-capacity").value=edge.capacity;
  renderDiagnostics(byId("edge-diagnostics"),diagnosticsByTarget(validation).edges[edge.id]??[]);
}
function renderDiagnostics(container,items){container.replaceChildren(...items.map(item=>{const p=document.createElement("p");p.className="diagnostic";p.textContent=[item.message,...(item.suggestions??[])].join(" ");return p;}));}

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

function syncName(){const name=byId("pipeline-name").value.trim()||"Untitled pipeline";if(name!==pipeline.metadata.name){pipeline.metadata.name=name;touch(pipeline);}}
async function saveGraph(){
  syncName();const saved=await request(`/api/pipeline/graphs/${encodeURIComponent(pipeline.graph_id)}`,jsonOptions("PUT",pipeline));pipeline=saved.document??saved;renderGraph();announce(`Saved ${pipeline.metadata.name} revision ${pipeline.revision} through the backend.`);
}
async function showOpen(){
  const {graphs}=await request("/api/pipeline/graphs");byId("saved-graphs").replaceChildren(...graphs.map(summary=>{const button=document.createElement("button");button.textContent=`${summary.name} · revision ${summary.revision}`;button.onclick=async()=>{const value=await request(`/api/pipeline/graphs/${encodeURIComponent(summary.graph_id)}`);loadGraph(value.document??value);byId("open-dialog").close();announce(`Opened ${summary.name}.`);};return button;}));byId("open-dialog").showModal();
}
async function shareGraph(){
  await saveGraph();const url=new URL(`/api/pipeline/graphs/${encodeURIComponent(pipeline.graph_id)}`,location.href).href;
  byId("share-url").value=url;byId("share-json").value=JSON.stringify(pipeline,null,2);byId("share-dialog").showModal();
}

async function runGraph(){
  if(runController)return;syncName();runController=new AbortController();byId("run").disabled=true;byId("cancel").disabled=false;byId("run-events").replaceChildren();
  try{
    const response=await fetch("/api/pipeline/run",{...jsonOptions("POST",pipeline),signal:runController.signal});
    if(!response.ok){const value=await response.json();throw new Error(value.validation?.diagnostics?.map(item=>item.message).join(" ")??value.error??"Run rejected");}
    await consumeNdjson(response.body.getReader(),renderRunEvent);
    announce("Graph run completed with streamed lifecycle evidence.");
  }catch(error){if(error.name==="AbortError"){renderRunEvent({kind:"cancelled",node_id:"graph",detail:"Cancelled by operator"});announce("Graph run cancelled.");}else announce(error.message,true);}
  finally{runController=null;byId("run").disabled=false;byId("cancel").disabled=true;}
}
function renderRunEvent(event){const item=document.createElement("li");item.className=event.kind;item.textContent=`${event.node_id} · ${event.kind}${event.elapsed_ms==null?"":` · ${event.elapsed_ms} ms`}${event.detail?` · ${event.detail}`:""}`;byId("run-events").append(item);item.scrollIntoView({block:"nearest"});if(event.node_id&&pipeline.nodes.some(node=>node.id===event.node_id)){cy.nodes().removeClass("compatible");cy.getElementById(event.node_id).addClass("compatible");}}

function deleteSelectedNode(){if(!selectedNode)return;removeNode(pipeline,selectedNode);selectedNode=null;renderGraph();scheduleValidation();announce("Node deleted.");}
function disableSelectedNode(){const node=pipeline.nodes.find(item=>item.id===selectedNode);if(!node)return;if(node.disabled){node.disabled=false;announce("Node enabled; reconnect any relationships it needs.");}else{pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==selectedNode&&edge.to.node_id!==selectedNode);pipeline.selected_sinks=pipeline.selected_sinks.filter(sink=>sink.node_id!==selectedNode);node.disabled=true;announce("Node disabled and removed from execution; its connections were removed explicitly.");}touch(pipeline);renderGraph();scheduleValidation();}

byId("palette-search").oninput=renderPalette;
byId("template").onchange=event=>{const starter=starters.find(graph=>graph.graph_id===event.target.value);if(starter)loadGraph(starter);};
byId("new").onclick=()=>loadGraph(createPipeline());byId("save").onclick=()=>saveGraph().catch(error=>announce(error.message,true));byId("open").onclick=()=>showOpen().catch(error=>announce(error.message,true));
byId("duplicate-graph").onclick=()=>{syncName();const copy=structuredClone(pipeline);copy.graph_id=`pipeline:${globalThis.crypto?.randomUUID?.()??Date.now()}`;copy.revision=1;copy.metadata.name+= " copy";loadGraph(copy);announce("Created an independent graph copy. Save to persist it.");};
byId("share").onclick=()=>shareGraph().catch(error=>announce(error.message,true));byId("copy-share").onclick=()=>navigator.clipboard.writeText(byId("share-url").value).then(()=>announce("Share URL copied."));
byId("fit").onclick=()=>cy.fit(undefined,40);byId("run").onclick=runGraph;byId("cancel").onclick=()=>runController?.abort();
byId("pipeline-name").onchange=()=>{syncName();scheduleValidation();};
byId("duplicate").onclick=()=>{const copy=duplicateNode(pipeline,selectedNode);renderGraph();if(copy)selectNode(copy.id);scheduleValidation();};
byId("delete").onclick=deleteSelectedNode;byId("disable").onclick=disableSelectedNode;
byId("bypass").onclick=()=>{try{bypassNode(pipeline,selectedNode,discovery);renderGraph();scheduleValidation();announce("Node bypassed by explicit compatible rewiring.");}catch(error){announce(error.message,true);}};
byId("replace").onclick=()=>{const node=pipeline.nodes.find(item=>item.id===selectedNode),choices=catalog.filter(item=>item.kind===node?.kind&&item.component_id!==node.component_id);if(!choices.length)return announce("No compatible backend-discovered replacement is available.",true);const choice=choices.length===1?choices[0]:choices.find(item=>item.label===prompt(`Replacement label:\n${choices.map(value=>value.label).join("\n")}`));if(choice){replaceNode(pipeline,node.id,choice);renderGraph();scheduleValidation();announce(`Replaced with ${choice.label}.`);}};
byId("apply-config").onclick=applyConfig;byId("cancel-connect").onclick=()=>{connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");};
byId("apply-edge").onclick=()=>{const edge=pipeline.edges.find(item=>item.id===selectedEdge);edge.capacity=Math.max(1,Number(byId("edge-capacity").value)||1);touch(pipeline);renderGraph();scheduleValidation();};
byId("delete-edge").onclick=()=>{removeEdge(pipeline,selectedEdge);selectedEdge=null;renderGraph();scheduleValidation();announce("Edge deleted.");};
byId("toggle-palette").onclick=()=>byId("palette-panel").classList.toggle("open");byId("toggle-inspector").onclick=()=>byId("inspector-panel").classList.toggle("open");
document.onkeydown=event=>{if(event.key==="Escape"&&connecting){connecting=null;cy.nodes().removeClass("compatible");renderInspector();announce("Connection cancelled.");}};

discover().catch(error=>{byId("validation").textContent=`Discovery failed: ${error.message}`;byId("validation").dataset.state="invalid";announce(error.message,true);});
