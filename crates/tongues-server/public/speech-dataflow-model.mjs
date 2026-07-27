export const PIPELINE_SCHEMA_VERSION = 2;
const LAYOUT_LABEL = "studio.layout.v1";
const NODE_FACEPLATE_LABEL = "studio.node-faceplate.v1";
const NODE_FACEPLATE_GEOMETRY_LABEL = "studio.node-faceplate-geometry.v1";
export const NODE_FACEPLATE_GEOMETRY_DEFAULT = {
  width: 228,
  height: 126,
  collapsed_height: 52,
};
const NODE_FACEPLATE_GEOMETRY_LIMITS = {
  width: {min: 120, max: 1400},
  height: {min: 48, max: 2400},
  collapsedHeight: {min: 28, max: 2400},
};

const NODE_KIND_ICON = {
  microphone: "◉",
  control_source: "⌘",
  audio_file: "♪",
  audio_source: "◖",
  text_source: "▤",
  text_file: "▥",
  text_url: "↗",
  asr: "◉",
  tts: "◒",
  transcript_sink: "⇥",
  audio_sink: "◒",
  audio_output: "◒",
  transcript_source: "▤",
  adaptation: "◇",
  adapter: "⇄",
};

const NODE_GROUP_ICON = {
  "Sources": "⇥",
  "Audio processing": "◒",
  "Audio & linguistic processing": "◒",
  "Recognition": "◉",
  "Language & speaker analysis": "◎",
  "Linguistic processing": "✦",
  "Response generation": "▤",
  "Synthesis": "◒",
  "Inspection & control": "◇",
};

function iconForKind(kind = "") {
  const normalized = String(kind);
  return NODE_KIND_ICON[normalized]
    || (normalized.includes("microphone") ? "◉" : null)
    || (normalized.includes("tts") || normalized.includes("speak") ? "◒" : null)
    || (normalized.includes("asr") || normalized.includes("speech") ? "◉" : null)
    || (normalized.includes("diarization") || normalized.includes("speaker") ? "◎" : null)
    || (normalized.includes("transcript") || normalized.endsWith("_sink") || normalized.includes("output") ? "▤" : null)
    || (normalized.includes("text") ? "▥" : null)
    || (normalized.includes("control") || normalized.includes("merge") || normalized.includes("adapter") ? "◇" : null);
}

function iconForEntry(entry = {}) {
  const kind = entry.kind ?? "";
  const direct = iconForKind(kind);
  if (direct) return direct;
  const group = entry.group ?? "";
  return NODE_GROUP_ICON[group] ?? "▦";
}

function itemForNode(node, catalog) {
  return catalogEntryForNode(node, catalog) ?? {kind: node?.kind ?? "node"};
}

export function nodeIcon(node, catalog) {
  return iconForEntry(itemForNode(node, catalog));
}

export function catalogItemIcon(item) {
  return iconForEntry(item);
}

export function nodeLabelWithIcon(node, catalog) {
  const icon = nodeIcon(node, catalog);
  const label = nodeLabel(node, catalog);
  return `${icon} ${label}`;
}

export function buildCatalog(discovery) {
  const kinds = discovery.node_kinds ?? {};
  const nodes = [];
  for (const kind of Object.values(kinds)) {
    if (!kind.requires_component) nodes.push(catalogEntry(kind));
  }
  for (const component of Object.values(discovery.components ?? {})) {
    const kind = kinds[component.node_kind];
    if (!kind) continue;
    nodes.push(catalogEntry(kind, component));
  }
  return nodes.sort((a,b)=>a.group.localeCompare(b.group)||a.label.localeCompare(b.label));
}

export function catalogEntryForNode(node,catalog) {
  const componentId=node?.component_id??null;
  return catalog.find(item=>item.kind===node?.kind&&(item.component_id??null)===componentId);
}

function catalogEntry(kind, component=null) {
  const replacement=component?.replacement?.family?component.replacement:kind.replacement;
  return {
    id:component?`component:${component.id}`:`kind:${kind.kind}`,
    kind:kind.kind,
    label:component?`${kind.label} · ${component.provider} / ${component.model}`:kind.label,
    component_id:component?.id??null,
    provider:component?.provider??"Tongues",
    model:component?.model??kind.kind,
    config:structuredClone(component?.default_config??kind.default_config??{}),
    schema:structuredClone(component?.configuration_schema??kind.configuration_schema??{}),
    ports:structuredClone(kind.ports??[]),
    readiness:component?.readiness??"ready",
    detail:component?.detail??"",
    capabilities:[...(component?.capabilities??kind.required_capabilities??[])],
    required_capabilities:[...(kind.required_capabilities??[])],
    replacement:structuredClone(replacement??{}),
    support:structuredClone(component?.support??{}),
    adapter:kind.adapter??null,
    merge:kind.merge??null,
    group:capabilityGroup(kind),
  };
}

function capabilityGroup(kind) {
  const capabilities=[...(kind.required_capabilities??[])];
  const portTypes=(kind.ports??[]).map(port=>port.value_type);
  if (kind.adapter) return "Audio & linguistic processing";
  if (kind.merge) return "Inspection & control";
  if (["microphone","audio_file","text_source","text_file","text_url","control_source"].includes(kind.kind)) return "Sources";
  if (kind.kind==="asr") return "Recognition";
  if (kind.kind==="diarization") return "Language & speaker analysis";
  if (["linguistic","interpretation"].includes(kind.kind)) return "Linguistic processing";
  if (capabilities.includes("text_generation")) return "Response generation";
  if (kind.kind==="tts") return "Synthesis";
  if (kind.kind.endsWith("_sink")||kind.kind==="audio_output") return "Outputs";
  if (portTypes.some(type=>type.startsWith("audio"))) return "Audio processing";
  return "Inspection & control";
}

export function createPipeline(name="Untitled pipeline") {
  return {
    schema_version:PIPELINE_SCHEMA_VERSION,graph_id:`pipeline:${cryptoId()}`,
    revision:1,metadata:{name,description:"",allow_unsafe_execution:false,labels:{}},
    nodes:[],edges:[],selected_sinks:[],
  };
}

function readNodeFaceplateMetadata(pipeline) {
  try{return JSON.parse(pipeline?.metadata?.labels?.[NODE_FACEPLATE_LABEL]??"{}");}catch{return{};}
}
function writeNodeFaceplateMetadata(pipeline,metadata) {
  pipeline.metadata.labels??={};
  if(!metadata||Object.keys(metadata).length===0) {
    delete pipeline.metadata.labels[NODE_FACEPLATE_LABEL];
    return;
  }
  pipeline.metadata.labels[NODE_FACEPLATE_LABEL]=JSON.stringify(metadata);
}
function readNodeFaceplateGeometryMetadata(pipeline) {
  try{return JSON.parse(pipeline?.metadata?.labels?.[NODE_FACEPLATE_GEOMETRY_LABEL]??"{}");}catch{return{};}
}
function writeNodeFaceplateGeometryMetadata(pipeline,metadata) {
  pipeline.metadata.labels??={};
  if(!metadata||Object.keys(metadata).length===0) {
    delete pipeline.metadata.labels[NODE_FACEPLATE_GEOMETRY_LABEL];
    return;
  }
  pipeline.metadata.labels[NODE_FACEPLATE_GEOMETRY_LABEL]=JSON.stringify(metadata);
}
function clampNodeFaceplateValue(value,limits) {
  if (!Number.isFinite(value)) return null;
  return Math.max(limits.min, Math.min(limits.max, Math.round(value)));
}
function normalizeFaceplateGeometryValue(raw) {
  const width = clampNodeFaceplateValue(Number(raw?.width), NODE_FACEPLATE_GEOMETRY_LIMITS.width);
  const height = clampNodeFaceplateValue(Number(raw?.height), NODE_FACEPLATE_GEOMETRY_LIMITS.height);
  const collapsedHeight = clampNodeFaceplateValue(Number(raw?.collapsed_height), NODE_FACEPLATE_GEOMETRY_LIMITS.collapsedHeight);
  if (width == null && height == null && collapsedHeight == null) return null;
  return {
    width: width ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.width,
    height: height ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.height,
    collapsed_height: collapsedHeight ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.collapsed_height,
  };
}
export function isNodeFaceplateCollapsed(pipeline,nodeId){
  return Boolean(readNodeFaceplateMetadata(pipeline)?.collapsed?.[nodeId]);
}
export function setNodeFaceplateCollapsed(pipeline,nodeId,collapsed){
  const metadata=readNodeFaceplateMetadata(pipeline);
  metadata.collapsed=metadata.collapsed??{};
  if(collapsed)metadata.collapsed[nodeId]=true;
  else delete metadata.collapsed[nodeId];
  writeNodeFaceplateMetadata(pipeline,metadata);
}
export function readNodeFaceplateGeometry(pipeline,nodeId){
  const metadata=readNodeFaceplateGeometryMetadata(pipeline);
  return metadata[nodeId] ? normalizeFaceplateGeometryValue(metadata[nodeId]) : {...NODE_FACEPLATE_GEOMETRY_DEFAULT};
}
export function setNodeFaceplateGeometry(pipeline,nodeId,geometry){
  const normalized=normalizeFaceplateGeometryValue(geometry);
  if(!normalized||!nodeId) return;
  const metadata=readNodeFaceplateGeometryMetadata(pipeline);
  const isDefault=geometryEquals(normalized,NODE_FACEPLATE_GEOMETRY_DEFAULT);
  if(geometryEquals(metadata[nodeId],normalized)){
    if(!isDefault) return;
    delete metadata[nodeId];
    writeNodeFaceplateGeometryMetadata(pipeline,metadata);
    return;
  }
  if (isDefault && !metadata[nodeId]) return;
  if (isDefault) delete metadata[nodeId];
  else metadata[nodeId]=normalized;
  writeNodeFaceplateGeometryMetadata(pipeline,metadata);
}
function geometryEquals(left,right){
  if(!left||!right)return false;
  return left.width===right.width&&left.height===right.height&&left.collapsed_height===right.collapsed_height;
}
export function deleteNodeFaceplateGeometry(pipeline,nodeId){
  if(!nodeId)return;
  const metadata=readNodeFaceplateGeometryMetadata(pipeline);
  if(!metadata[nodeId]) return;
  delete metadata[nodeId];
  writeNodeFaceplateGeometryMetadata(pipeline,metadata);
}
function deleteNodeFaceplateState(pipeline,nodeId) {
  const collapsed=readNodeFaceplateMetadata(pipeline);
  const geometry=readNodeFaceplateGeometryMetadata(pipeline);
  if (collapsed?.collapsed?.[nodeId]) delete collapsed.collapsed[nodeId];
  if (Object.keys(collapsed?.collapsed ?? {}).length===0) delete collapsed.collapsed;
  writeNodeFaceplateMetadata(pipeline,collapsed);
  if (geometry[nodeId]) delete geometry[nodeId];
  writeNodeFaceplateGeometryMetadata(pipeline,geometry);
}

function cryptoId() {
  return globalThis.crypto?.randomUUID?.()??`${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function addNode(pipeline,catalogNode,afterId=null,position=null) {
  if (!catalogNode) throw new Error("Choose a backend-discovered node first.");
  const node={
    id:`node:${cryptoId()}`,kind:catalogNode.kind,
    component_id:catalogNode.component_id,config:structuredClone(catalogNode.config??{}),
    disabled:false,bypassed:false,
  };
  const index=afterId?pipeline.nodes.findIndex(item=>item.id===afterId)+1:pipeline.nodes.length;
  pipeline.nodes.splice(Math.max(0,index),0,node);
  if(position)setNodePosition(pipeline,node.id,position);
  touch(pipeline); return node;
}

export function removeNode(pipeline,id) {
  pipeline.nodes=pipeline.nodes.filter(node=>node.id!==id);
  pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==id&&edge.to.node_id!==id);
  pipeline.selected_sinks=pipeline.selected_sinks.filter(sink=>sink.node_id!==id);
  deleteNodeFaceplateState(pipeline,id);
  const layout=readLayout(pipeline);delete layout[id];writeLayout(pipeline,layout);touch(pipeline);
}

export function duplicateNode(pipeline,id,offset={x:36,y:36}) {
  const source=pipeline.nodes.find(node=>node.id===id);
  if(!source)return null;
  const copy={...structuredClone(source),id:`node:${cryptoId()}`};
  pipeline.nodes.splice(pipeline.nodes.indexOf(source)+1,0,copy);
  const position=nodePosition(pipeline,id);
  if(position)setNodePosition(pipeline,copy.id,{x:position.x+offset.x,y:position.y+offset.y});
  const geometry=readNodeFaceplateGeometry(pipeline,id);
  if(geometry)setNodeFaceplateGeometry(pipeline,copy.id,geometry);
  touch(pipeline);return copy;
}

export function applyNodeConfig(pipeline,id,values) {
  const node=pipeline.nodes.find(item=>item.id===id);
  if(!node)throw new Error("The selected node is no longer present.");
  Object.assign(node.config,structuredClone(values));
  touch(pipeline);return node;
}

function canonical(value) {
  if(Array.isArray(value))return value.map(canonical);
  if(value&&typeof value==="object")return Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])]));
  return value;
}

function sameValue(left,right){return JSON.stringify(canonical(left))===JSON.stringify(canonical(right));}
function pipelineFingerprint(pipeline){return JSON.stringify(pipeline);}
function replacementMetadata(entry){return entry?.replacement??{};}
function portKey(port){return `${port.direction}:${port.id}:${port.value_type}:${port.cardinality}:${Boolean(port.streaming)}`;}
function exactPortContract(left,right){
  return left.length===right.length&&left.map(portKey).sort().join("|")===right.map(portKey).sort().join("|");
}
function destinationPort(sourcePort,candidate){
  const aliases=replacementMetadata(candidate).port_aliases??{};
  const id=aliases[sourcePort.id]??sourcePort.id;
  return candidate.ports.find(port=>port.id===id&&port.direction===sourcePort.direction);
}
function connectedPortIds(pipeline,nodeId){
  const ids=new Set();
  pipeline.edges.forEach(edge=>{
    if(edge.from.node_id===nodeId)ids.add(edge.from.port_id);
    if(edge.to.node_id===nodeId)ids.add(edge.to.port_id);
  });
  pipeline.selected_sinks.filter(sink=>sink.node_id===nodeId).forEach(sink=>ids.add(sink.port_id));
  return ids;
}

export function validateSchemaValue(value,schema,path="$") {
  const errors=[];
  if(schema==null||typeof schema!=="object")return errors;
  if(schema.const!==undefined&&!sameValue(value,schema.const))errors.push(`${path} must equal the declared constant.`);
  if(Array.isArray(schema.enum)&&!schema.enum.some(item=>sameValue(item,value)))errors.push(`${path} is not an allowed value.`);
  const types=Array.isArray(schema.type)?schema.type:[schema.type].filter(Boolean);
  const matches=type=>{
    if(type==="null")return value===null;
    if(type==="object")return value!==null&&!Array.isArray(value)&&typeof value==="object";
    if(type==="array")return Array.isArray(value);
    if(type==="integer")return Number.isInteger(value);
    if(type==="number")return typeof value==="number"&&Number.isFinite(value);
    return typeof value===type;
  };
  if(types.length&&!types.some(matches)){errors.push(`${path} must be ${types.join(" or ")}.`);return errors;}
  if(typeof value==="number"){
    if(schema.minimum!=null&&value<schema.minimum)errors.push(`${path} must be at least ${schema.minimum}.`);
    if(schema.maximum!=null&&value>schema.maximum)errors.push(`${path} must be at most ${schema.maximum}.`);
    if(schema.exclusiveMinimum!=null&&value<=schema.exclusiveMinimum)errors.push(`${path} must be greater than ${schema.exclusiveMinimum}.`);
    if(schema.exclusiveMaximum!=null&&value>=schema.exclusiveMaximum)errors.push(`${path} must be less than ${schema.exclusiveMaximum}.`);
  }
  if(typeof value==="string"){
    if(schema.minLength!=null&&value.length<schema.minLength)errors.push(`${path} is too short.`);
    if(schema.maxLength!=null&&value.length>schema.maxLength)errors.push(`${path} is too long.`);
    if(schema.pattern)try{if(!new RegExp(schema.pattern).test(value))errors.push(`${path} does not match the required pattern.`);}catch{}
  }
  if(Array.isArray(value)){
    if(schema.minItems!=null&&value.length<schema.minItems)errors.push(`${path} needs at least ${schema.minItems} items.`);
    if(schema.maxItems!=null&&value.length>schema.maxItems)errors.push(`${path} allows at most ${schema.maxItems} items.`);
    value.forEach((item,index)=>errors.push(...validateSchemaValue(item,schema.items??{},`${path}[${index}]`)));
  }
  if(value!==null&&!Array.isArray(value)&&typeof value==="object"){
    for(const required of schema.required??[])if(value[required]===undefined)errors.push(`${path}.${required} is required.`);
    for(const [name,item] of Object.entries(value)){
      const property=schema.properties?.[name];
      if(property)errors.push(...validateSchemaValue(item,property,`${path}.${name}`));
      else if(schema.additionalProperties===false)errors.push(`${path}.${name} is not allowed.`);
      else if(schema.additionalProperties&&typeof schema.additionalProperties==="object")errors.push(...validateSchemaValue(item,schema.additionalProperties,`${path}.${name}`));
    }
  }
  return errors;
}

export function migrateReplacementConfig(currentConfig,currentEntry,candidate,{useDefaults=false,overrides={}}={}) {
  const source=currentConfig??{},sourceSchema=currentEntry?.schema??{},targetSchema=candidate?.schema??{};
  const targetProperties=targetSchema.properties??{},sourceProperties=sourceSchema.properties??{};
  const required=new Set(targetSchema.required??[]),defaults=candidate?.config??{};
  const aliases=replacementMetadata(candidate).configuration_aliases??{};
  const targetToSource=new Map(Object.entries(aliases).map(([from,to])=>[to,from]));
  const sameSchema=replacementMetadata(currentEntry).configuration_schema_id
    &&replacementMetadata(currentEntry).configuration_schema_id===replacementMetadata(candidate).configuration_schema_id
    &&replacementMetadata(currentEntry).configuration_schema_version===replacementMetadata(candidate).configuration_schema_version;
  const config={},changes=[],used=new Set(),blocking=[];
  for(const [targetName,targetSpec] of Object.entries(targetProperties)){
    const mappedSource=targetToSource.get(targetName),sourceName=mappedSource??targetName;
    const sourceSpec=sourceProperties[sourceName],hasSource=Object.prototype.hasOwnProperty.call(source,sourceName);
    const explicitMap=Boolean(mappedSource),equivalent=explicitMap||sameSchema||sameValue(sourceSpec,targetSpec);
    let value,state,reason;
    if(Object.prototype.hasOwnProperty.call(overrides,targetName)){
      value=structuredClone(overrides[targetName]);state="provided";reason="Entered for the replacement.";
    }else if(!useDefaults&&hasSource&&equivalent&&!validateSchemaValue(source[sourceName],targetSpec).length){
      value=structuredClone(source[sourceName]);state=explicitMap?"mapped":"preserved";
      reason=explicitMap?`Mapped from ${sourceName} by backend metadata.`:"Compatible value preserved.";
      used.add(sourceName);
    }else if(Object.prototype.hasOwnProperty.call(defaults,targetName)){
      value=structuredClone(defaults[targetName]);state="defaulted";reason=useDefaults?"Replacement defaults requested.":"Existing value is absent, incompatible, or invalid.";
    }else if(required.has(targetName)){
      state="requires_input";reason="Required by the replacement and no valid default is available.";
      blocking.push({field:targetName,code:"config.required_input",message:reason});
    }else continue;
    if(state!=="requires_input"){
      const errors=validateSchemaValue(value,targetSpec);
      if(errors.length){state="invalid";reason=errors.join(" ");blocking.push({field:targetName,code:"config.invalid",message:reason});}
      else config[targetName]=value;
    }
    changes.push({field:targetName,source_field:sourceName,state,before:source[sourceName],after:value,reason});
  }
  for(const [name,value] of Object.entries(source))if(!used.has(name)&&!changes.some(change=>change.source_field===name&&["preserved","mapped"].includes(change.state))){
    changes.push({field:name,source_field:name,state:"removed",before:value,reason:"The replacement schema does not accept this field."});
  }
  const wholeErrors=validateSchemaValue(config,targetSchema);
  for(const message of wholeErrors)if(!blocking.some(item=>item.message===message))blocking.push({field:null,code:"config.invalid",message});
  return{config,changes,blocking};
}

export function classifyReplacement(pipeline,nodeId,currentEntry,candidate) {
  const node=pipeline.nodes.find(item=>item.id===nodeId),currentMeta=replacementMetadata(currentEntry),candidateMeta=replacementMetadata(candidate);
  const base={candidate,compatibility:"incompatible",code:"replacement.incompatible",reason:"This candidate cannot replace the selected node.",applyable:false,port_changes:[]};
  if(!node||!currentEntry)return{...base,code:"replacement.source_missing",reason:"The selected node or its discovery metadata is no longer available."};
  if(!currentMeta.family||!candidateMeta.family||!currentMeta.configuration_schema_id||!candidateMeta.configuration_schema_id){
    return{...base,code:"replacement.metadata_missing",reason:"Backend replacement family or schema identity is missing; replacement fails closed."};
  }
  if(currentMeta.family!==candidateMeta.family)return{...base,code:"replacement.family_mismatch",reason:`Backend families differ (${currentMeta.family} vs ${candidateMeta.family}).`};
  const required=new Set(candidate.required_capabilities??[]),capabilities=new Set(candidate.capabilities??[]);
  const missing=[...required].filter(value=>!capabilities.has(value));
  if(missing.length)return{...base,code:"replacement.capability_missing",reason:`Candidate is missing required capability: ${missing.join(", ")}.`};
  const connected=connectedPortIds(pipeline,nodeId),disconnect=new Set(candidateMeta.disconnect_ports??[]);
  const portChanges=[],blocking=[];
  for(const sourcePort of currentEntry.ports){
    const target=destinationPort(sourcePort,candidate);
    if(target&&target.value_type===sourcePort.value_type){
      if(target.id!==sourcePort.id)portChanges.push({from:sourcePort.id,to:target.id,state:"remapped"});
      continue;
    }
    if(connected.has(sourcePort.id)&&!disconnect.has(sourcePort.id))blocking.push(sourcePort.id);
    portChanges.push({from:sourcePort.id,to:null,state:disconnect.has(sourcePort.id)?"disconnect":"missing"});
  }
  if(blocking.length)return{...base,code:"replacement.connected_port_missing",reason:`Connected port${blocking.length===1?"":"s"} ${blocking.join(", ")} have no backend-declared mapping.`,port_changes:portChanges};
  const exact=node.kind===candidate.kind&&exactPortContract(currentEntry.ports,candidate.ports);
  const compatibility=exact?"exact_drop_in":"migration";
  const readiness=candidate.readiness??"unknown";
  const ready=readiness==="ready";
  const reason=exact
    ?(ready?"Same backend family and exact port contract; connections can be preserved.":`Exact port contract, but readiness is ${readiness}.`)
    :(ready?"Backend-declared family match with an explicit port impact plan.":`Structurally replaceable, but readiness is ${readiness}.`);
  return{candidate,compatibility,code:ready?`replacement.${compatibility}`:"replacement.not_ready",reason,applyable:ready,port_changes:portChanges};
}

export function replacementCandidates(pipeline,nodeId,catalog) {
  const node=pipeline.nodes.find(item=>item.id===nodeId),currentEntry=catalogEntryForNode(node,catalog);
  if(!node||!currentEntry)return[];
  const currentFamily=replacementMetadata(currentEntry).family;
  return catalog.filter(candidate=>candidate.id!==currentEntry.id
    &&(candidate.kind===node.kind||(currentFamily&&replacementMetadata(candidate).family===currentFamily))).map(candidate=>{
    const result=classifyReplacement(pipeline,nodeId,currentEntry,candidate);
    const migration=migrateReplacementConfig(node.config,currentEntry,candidate);
    const preserved=migration.changes.filter(change=>["preserved","mapped"].includes(change.state)).length;
    return{...candidate,...result,config_preview:migration,preserved_config_count:preserved};
  }).sort((left,right)=>{
    const rank={exact_drop_in:0,migration:1,incompatible:2};
    return rank[left.compatibility]-rank[right.compatibility]
      ||Number(right.readiness==="ready")-Number(left.readiness==="ready")
      ||right.preserved_config_count-left.preserved_config_count
      ||left.label.localeCompare(right.label)
      ||left.id.localeCompare(right.id);
  });
}

export function planNodeReplacement(pipeline,nodeId,candidate,catalog,{useDefaults=false,overrides={},catalogRevision=null}={}) {
  const node=pipeline.nodes.find(item=>item.id===nodeId),currentEntry=catalogEntryForNode(node,catalog);
  const classification=classifyReplacement(pipeline,nodeId,currentEntry,candidate);
  const beforeFingerprint=pipelineFingerprint(pipeline),preview=structuredClone(pipeline);
  const config=migrateReplacementConfig(node?.config,currentEntry,candidate,{useDefaults,overrides});
  const edgeChanges=[],sinkChanges=[],blocking=[...config.blocking];
  if(classification.compatibility==="incompatible")blocking.push({code:classification.code,message:classification.reason});
  const aliases=replacementMetadata(candidate).port_aliases??{},disconnect=new Set(replacementMetadata(candidate).disconnect_ports??[]);
  const edges=[];
  for(const edge of preview.edges){
    let drop=false,next=structuredClone(edge);
    for(const endpointName of ["from","to"]){
      const endpoint=next[endpointName];if(endpoint.node_id!==nodeId)continue;
      const targetId=aliases[endpoint.port_id]??endpoint.port_id;
      const targetPort=candidate.ports.find(port=>port.id===targetId&&port.direction===(endpointName==="from"?"output":"input"));
      if(targetPort){
        const state=targetId===endpoint.port_id?"preserved":"remapped";
        edgeChanges.push({edge_id:edge.id,endpoint:endpointName,from:edge[endpointName].port_id,to:targetId,state});
        endpoint.port_id=targetId;
      }else if(disconnect.has(endpoint.port_id)){drop=true;edgeChanges.push({edge_id:edge.id,endpoint:endpointName,from:endpoint.port_id,to:null,state:"disconnected"});}
      else blocking.push({code:"replacement.edge_unmapped",edge_id:edge.id,message:`Edge ${edge.id} uses unmapped port ${endpoint.port_id}.`});
    }
    if(!drop)edges.push(next);
  }
  preview.edges=edges;
  preview.selected_sinks=preview.selected_sinks.flatMap(sink=>{
    if(sink.node_id!==nodeId)return[sink];
    const targetId=aliases[sink.port_id]??sink.port_id;
    if(candidate.ports.some(port=>port.id===targetId&&port.direction==="output")){
      sinkChanges.push({from:sink.port_id,to:targetId,state:targetId===sink.port_id?"preserved":"remapped"});
      return[{...sink,port_id:targetId}];
    }
    if(disconnect.has(sink.port_id)){sinkChanges.push({from:sink.port_id,to:null,state:"disconnected"});return[];}
    blocking.push({code:"replacement.sink_unmapped",message:`Selected sink ${sink.port_id} has no mapping.`});return[sink];
  });
  const previewNode=preview.nodes.find(item=>item.id===nodeId);
  if(previewNode)Object.assign(previewNode,{kind:candidate?.kind,component_id:candidate?.component_id,config:structuredClone(config.config)});
  preview.revision=Math.max(1,Number(pipeline.revision)||1)+1;
  const lossy=edgeChanges.some(change=>change.state==="disconnected")||sinkChanges.some(change=>change.state==="disconnected")
    ||config.changes.some(change=>["defaulted","removed","invalid","requires_input"].includes(change.state));
  return{
    schema_version:1,node_id:nodeId,candidate_id:candidate?.id,current_component_id:node?.component_id??null,
    candidate_component_id:candidate?.component_id??null,graph_revision:pipeline.revision,
    graph_fingerprint:beforeFingerprint,catalog_revision:catalogRevision,
    classification,config_changes:config.changes,edge_changes:edgeChanges,sink_changes:sinkChanges,
    lossless:!lossy,lossy,blocking,preview_graph:preview,validation:null,applyable:false,
  };
}

export function attachReplacementValidation(plan,report) {
  const next=structuredClone(plan);next.validation=structuredClone(report);
  if(!report?.valid)next.blocking.push({code:"replacement.validation_failed",message:report?.diagnostics?.[0]?.message??"The replacement preview is not valid."});
  next.applyable=next.classification.applyable&&next.blocking.length===0&&Boolean(report?.valid);
  return next;
}

export function applyReplacementPlan(pipeline,plan,catalogRevision) {
  if(!plan?.applyable)throw new Error("The displayed replacement plan is not applyable.");
  if(pipeline.revision!==plan.graph_revision||pipelineFingerprint(pipeline)!==plan.graph_fingerprint)throw new Error("The graph changed after this replacement preview. Review it again.");
  if((catalogRevision??null)!==(plan.catalog_revision??null))throw new Error("Backend discovery changed after this replacement preview. Review it again.");
  const node=pipeline.nodes.find(item=>item.id===plan.node_id);
  if(!node||(node.component_id??null)!==plan.current_component_id)throw new Error("The selected node changed after this replacement preview.");
  return structuredClone(plan.preview_graph);
}

export function createEditHistory(){return{undo:[],redo:[]};}
export function clearRedo(history){history.redo.length=0;}
export function recordEdit(history,before,after,{
  label="Graph edit",selectionBefore=null,selectionAfter=null,focusBefore=null,focusAfter=null,
}={}) {
  if(pipelineFingerprint(before)===pipelineFingerprint(after))return false;
  history.undo.push({
    label,before:structuredClone(before),after:structuredClone(after),
    selection_before:structuredClone(selectionBefore),selection_after:structuredClone(selectionAfter),
    focus_before:structuredClone(focusBefore),focus_after:structuredClone(focusAfter),
  });
  history.redo.length=0;
  return true;
}
export function commitReplacement(pipeline,plan,catalogRevision,history,selection) {
  const next=applyReplacementPlan(pipeline,plan,catalogRevision);
  const afterSelection=selection&&typeof selection==="object"
    ?{...structuredClone(selection),node_id:plan.node_id,edge_id:null}
    :plan.node_id;
  recordEdit(history,pipeline,next,{
    label:"Replace node",selectionBefore:selection,selectionAfter:afterSelection,
  });
  return next;
}
export function undoEdit(history){
  const entry=history.undo.pop();if(!entry)return null;history.redo.push(entry);
  return{
    pipeline:structuredClone(entry.before),selection:structuredClone(entry.selection_before),
    focus:structuredClone(entry.focus_before),label:entry.label??"Graph edit",
  };
}
export function redoEdit(history){
  const entry=history.redo.pop();if(!entry)return null;history.undo.push(entry);
  return{
    pipeline:structuredClone(entry.after),selection:structuredClone(entry.selection_after),
    focus:structuredClone(entry.focus_after),label:entry.label??"Graph edit",
  };
}

export function insertSubgraph(pipeline,template,origin={x:80,y:80}) {
  const copy=structuredClone(template), idMap=new Map();
  copy.nodes.forEach((node,index)=>{
    const old=node.id;node.id=`node:${cryptoId()}`;idMap.set(old,node.id);
    pipeline.nodes.push(node);
    const row=Math.floor(index/3),column=row%2===0?index%3:2-(index%3);
    setNodePosition(pipeline,node.id,{x:origin.x+column*290,y:origin.y+row*220});
  });
  copy.edges.forEach(edge=>{
    edge.id=`edge:${cryptoId()}`;edge.from.node_id=idMap.get(edge.from.node_id);
    edge.to.node_id=idMap.get(edge.to.node_id);pipeline.edges.push(edge);
  });
  touch(pipeline);return copy.nodes.map(node=>node.id);
}

export function portsFor(node,direction,discovery) {
  return (discovery.node_kinds?.[node?.kind]?.ports??[]).filter(port=>port.direction===direction);
}

export function compatibleTargets(pipeline,fromNodeId,fromPortId,discovery) {
  const source=pipeline.nodes.find(node=>node.id===fromNodeId);
  const output=portsFor(source,"output",discovery).find(port=>port.id===fromPortId);
  if(!output)return[];
  return pipeline.nodes.flatMap(node=>portsFor(node,"input",discovery)
    .map(port=>connectionCompatibility(pipeline,fromNodeId,fromPortId,node.id,port.id,discovery))
    .filter(result=>result.compatible)
    .map(result=>({node_id:result.to.node_id,port_id:result.to.port_id,value_type:result.input.value_type})));
}

export function adapterPaths(fromType,toType,discovery) {
  return Object.values(discovery.node_kinds??{})
    .filter(kind=>kind.adapter?.from===fromType&&kind.adapter?.to===toType)
    .map(kind=>({kind:kind.kind,label:kind.label}));
}

export function connectionIntentCandidates(catalog,direction,valueType) {
  const wanted=direction==="from_output"?"input":"output";
  return catalog.map(candidate=>{
    const ports=(candidate.ports??[]).filter(port=>port.direction===wanted&&port.value_type===valueType);
    return{candidate,ports,compatible:ports.length===1,ambiguous:ports.length>1,
      reason:ports.length===1?`${ports[0].label??ports[0].id} accepts ${valueType}.`
        :ports.length>1?`${ports.length} ${valueType} ${wanted} ports require a deliberate choice.`
        :`No ${wanted} port accepts ${valueType}.`};
  }).filter(option=>option.ports.length>0);
}

export function insertionCandidates(pipeline,edgeId,catalog,discovery) {
  const edge=pipeline.edges.find(item=>item.id===edgeId);
  if(!edge)return[];
  const source=pipeline.nodes.find(node=>node.id===edge.from.node_id),target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
  const output=portsFor(source,"output",discovery).find(port=>port.id===edge.from.port_id);
  const input=portsFor(target,"input",discovery).find(port=>port.id===edge.to.port_id);
  if(!output||!input)return[];
  return catalog.map(candidate=>{
    const inputs=candidate.ports.filter(port=>port.direction==="input"&&port.value_type===output.value_type);
    const outputs=candidate.ports.filter(port=>port.direction==="output"&&port.value_type===input.value_type);
    const mappings=inputs.flatMap(inPort=>outputs.map(outPort=>({input_port_id:inPort.id,output_port_id:outPort.id})));
    const ready=(candidate.readiness??"ready")==="ready",compatible=mappings.length===1&&ready;
    return{candidate,mappings,compatible,ambiguous:mappings.length>1,
      reason:!ready?`Readiness is ${candidate.readiness}.`
        :mappings.length===1?`${output.value_type} → ${candidate.label} → ${input.value_type} is explicit and unambiguous.`
        :mappings.length>1?`${mappings.length} typed port mappings require a deliberate choice.`
        :`This module cannot accept ${output.value_type} and produce ${input.value_type}.`};
  }).filter(option=>option.mappings.length>0);
}

function nodeFromCatalog(catalogNode) {
  return{
    id:`node:${cryptoId()}`,kind:catalogNode.kind,component_id:catalogNode.component_id,
    config:structuredClone(catalogNode.config??{}),disabled:false,bypassed:false,
  };
}

export function insertNodeOnEdge(pipeline,edgeId,catalogNode,mapping,discovery,position=null) {
  const edge=pipeline.edges.find(item=>item.id===edgeId);
  if(!edge)throw new Error("The selected cable is no longer present.");
  const option=insertionCandidates(pipeline,edgeId,[catalogNode],discovery)[0];
  const chosen=option?.mappings.find(item=>item.input_port_id===mapping?.input_port_id&&item.output_port_id===mapping?.output_port_id);
  if(!option?.compatible||!chosen)throw new Error(option?.reason??"The module has no unambiguous typed insertion path.");
  const node=nodeFromCatalog(catalogNode),oldTarget=structuredClone(edge.to),capacity=edge.capacity;
  pipeline.nodes.push(node);
  edge.to={node_id:node.id,port_id:chosen.input_port_id};
  const downstream={id:`edge:${cryptoId()}`,from:{node_id:node.id,port_id:chosen.output_port_id},to:oldTarget,capacity};
  pipeline.edges.push(downstream);
  if(position)setNodePosition(pipeline,node.id,position);
  touch(pipeline);return{node,upstream_edge:edge,downstream_edge:downstream};
}

export function addNodeAtConnectionIntent(pipeline,catalogNode,anchor,direction,discovery,position=null) {
  const anchorNode=pipeline.nodes.find(node=>node.id===anchor?.node_id);
  const anchorDirection=direction==="from_output"?"output":"input";
  const anchorPort=portsFor(anchorNode,anchorDirection,discovery).find(port=>port.id===anchor?.port_id);
  if(!anchorNode||!anchorPort)throw new Error("The quick-add connection anchor is no longer present.");
  const option=connectionIntentCandidates([catalogNode],direction,anchorPort.value_type)[0];
  if(!option?.compatible)throw new Error(option?.reason??"The module has no unambiguous compatible port.");
  if(direction==="to_input"){
    const occupied=pipeline.edges.some(edge=>edge.to.node_id===anchor.node_id&&edge.to.port_id===anchor.port_id);
    if(anchorPort.cardinality!=="many"&&occupied)throw new Error(`${anchorNode.kind}.${anchorPort.id} is occupied; add an explicit merge node.`);
  }
  const node=nodeFromCatalog(catalogNode),port=option.ports[0];
  const from=direction==="from_output"?anchor:{node_id:node.id,port_id:port.id};
  const to=direction==="from_output"?{node_id:node.id,port_id:port.id}:anchor;
  pipeline.nodes.push(node);
  const edge={id:`edge:${cryptoId()}`,from,to,capacity:16};pipeline.edges.push(edge);
  if(position)setNodePosition(pipeline,node.id,position);
  touch(pipeline);return{node,edge};
}

export function connectionCompatibility(pipeline,fromNode,fromPort,toNode,toPort,discovery,{ignoreEdgeId=null}={}) {
  const source=pipeline.nodes.find(node=>node.id===fromNode),target=pipeline.nodes.find(node=>node.id===toNode);
  const base={compatible:false,from:{node_id:fromNode,port_id:fromPort},to:{node_id:toNode,port_id:toPort},source,target,output:null,input:null};
  if(!source||!target)return{...base,code:"connection.endpoint_missing",reason:"Both connection endpoints must exist."};
  const output=portsFor(source,"output",discovery).find(port=>port.id===fromPort);
  const input=portsFor(target,"input",discovery).find(port=>port.id===toPort);
  const detail={...base,output,input};
  if(!output||!input)return{...detail,code:"connection.direction",reason:"Choose an output port and an input port."};
  if(output.value_type!==input.value_type){
    const adapters=adapterPaths(output.value_type,input.value_type,discovery);
    const route=adapters.length?` Add ${adapters.map(item=>item.label).join(" or ")} between them.`:" No registered adapter path is available.";
    return{...detail,code:adapters.length?"connection.adapter_available":"connection.type_mismatch",adapters,
      reason:`${source.kind}.${output.id} emits ${output.value_type}; ${target.kind}.${input.id} requires ${input.value_type}.${route}`};
  }
  const otherEdges=pipeline.edges.filter(edge=>edge.id!==ignoreEdgeId);
  const duplicate=otherEdges.find(edge=>edge.from.node_id===fromNode&&edge.from.port_id===fromPort&&edge.to.node_id===toNode&&edge.to.port_id===toPort);
  if(duplicate)return{...detail,code:"connection.duplicate",reason:"That typed connection already exists.",duplicate_edge_id:duplicate.id};
  const occupied=otherEdges.find(edge=>edge.to.node_id===toNode&&edge.to.port_id===toPort);
  if(input.cardinality!=="many"&&occupied){
    return{...detail,code:"connection.input_occupied",occupied_edge_id:occupied.id,
      reason:`${target.kind}.${input.id} accepts one connection and is already occupied. Insert an explicit merge node or reconnect the existing cable.`};
  }
  return{...detail,compatible:true,code:"connection.compatible",reason:`${output.value_type} output is compatible with ${input.value_type} input.`};
}

export function connectPorts(pipeline,fromNode,fromPort,toNode,toPort,discovery) {
  const compatibility=connectionCompatibility(pipeline,fromNode,fromPort,toNode,toPort,discovery);
  if(!compatibility.compatible){
    if(compatibility.code==="connection.duplicate")return pipeline.edges.find(edge=>edge.id===compatibility.duplicate_edge_id);
    throw new Error(compatibility.reason);
  }
  const edge={id:`edge:${cryptoId()}`,from:{node_id:fromNode,port_id:fromPort},to:{node_id:toNode,port_id:toPort},capacity:16};
  pipeline.edges.push(edge);
  touch(pipeline);
  return edge;
}

export function reconnectEdge(pipeline,edgeId,endpoint,nodeId,portId,discovery) {
  const edge=pipeline.edges.find(item=>item.id===edgeId);
  if(!edge)throw new Error("The selected connection is no longer present.");
  if(!["from","to"].includes(endpoint))throw new Error("Choose which cable plug to reconnect.");
  const from=endpoint==="from"?{node_id:nodeId,port_id:portId}:edge.from;
  const to=endpoint==="to"?{node_id:nodeId,port_id:portId}:edge.to;
  const compatibility=connectionCompatibility(
    pipeline,from.node_id,from.port_id,to.node_id,to.port_id,discovery,{ignoreEdgeId:edgeId},
  );
  if(!compatibility.compatible)throw new Error(compatibility.reason);
  edge[endpoint]={node_id:nodeId,port_id:portId};
  touch(pipeline);
  return edge;
}

export function connect(pipeline,from,to,discovery) {
  const source=pipeline.nodes.find(node=>node.id===from),target=pipeline.nodes.find(node=>node.id===to);
  const pairs=portsFor(source,"output",discovery).flatMap(output=>
    portsFor(target,"input",discovery).filter(input=>input.value_type===output.value_type).map(input=>({output,input})));
  if(!pairs.length){
    const outputs=portsFor(source,"output",discovery).map(port=>port.value_type);
    const inputs=portsFor(target,"input",discovery).map(port=>port.value_type);
    const adapters=outputs.flatMap(fromType=>inputs.flatMap(toType=>adapterPaths(fromType,toType,discovery)));
    throw new Error(`${source?.kind??"source"} emits ${outputs.join(" or ")||"nothing"}; ${target?.kind??"target"} requires ${inputs.join(" or ")||"no input"}.${adapters.length?` Add ${adapters.map(item=>item.label).join(" or ")}.`:""}`);
  }
  connectPorts(pipeline,from,pairs[0].output.id,to,pairs[0].input.id,discovery);
}

export function removeEdge(pipeline,id){pipeline.edges=pipeline.edges.filter(edge=>edge.id!==id);touch(pipeline);}

export function bypassNode(pipeline,id,discovery) {
  const incoming=pipeline.edges.filter(edge=>edge.to.node_id===id);
  const outgoing=pipeline.edges.filter(edge=>edge.from.node_id===id);
  if(incoming.length!==1||outgoing.length<1)throw new Error("Bypass needs exactly one input edge and at least one output edge.");
  const upstream=incoming[0];
  const source=pipeline.nodes.find(node=>node.id===upstream.from.node_id);
  const sourcePort=portsFor(source,"output",discovery).find(port=>port.id===upstream.from.port_id);
  const replacements=outgoing.map(edge=>{
    const target=pipeline.nodes.find(node=>node.id===edge.to.node_id);
    const input=portsFor(target,"input",discovery).find(port=>port.id===edge.to.port_id);
    if(!sourcePort||!input||sourcePort.value_type!==input.value_type)throw new Error(`Cannot bypass ${source?.kind}: ${sourcePort?.value_type??"unknown"} does not satisfy ${target?.kind}.${input?.id??"input"}.`);
    return{to:edge.to};
  });
  pipeline.edges=pipeline.edges.filter(edge=>edge.from.node_id!==id&&edge.to.node_id!==id);
  replacements.forEach(item=>connectPorts(pipeline,upstream.from.node_id,upstream.from.port_id,item.to.node_id,item.to.port_id,discovery));
  const node=pipeline.nodes.find(item=>item.id===id);if(node)node.bypassed=true;touch(pipeline);
}

export function nodeLabel(node,catalog) {
  return catalogEntryForNode(node,catalog)?.label??node?.kind??"Unknown node";
}

export function readLayout(pipeline) {
  try{return JSON.parse(pipeline.metadata?.labels?.[LAYOUT_LABEL]??"{}");}catch{return{};}
}
function writeLayout(pipeline,layout) {
  pipeline.metadata.labels??={};pipeline.metadata.labels[LAYOUT_LABEL]=JSON.stringify(layout);
}
export function nodePosition(pipeline,id){return readLayout(pipeline)[id]??null;}
export function setNodePosition(pipeline,id,position){const layout=readLayout(pipeline);layout[id]={x:Math.round(position.x),y:Math.round(position.y)};writeLayout(pipeline,layout);}

export function copyGraphSelection(pipeline,nodeIds) {
  const selected=new Set(nodeIds),layout=readLayout(pipeline);
  const geometry=readNodeFaceplateGeometryMetadata(pipeline);
  return{
    schema_version:1,
    nodes:pipeline.nodes.filter(node=>selected.has(node.id)).map(node=>structuredClone(node)),
    edges:pipeline.edges.filter(edge=>selected.has(edge.from.node_id)&&selected.has(edge.to.node_id)).map(edge=>structuredClone(edge)),
    positions:Object.fromEntries([...selected].filter(id=>layout[id]).map(id=>[id,structuredClone(layout[id])])),
    faceplate_geometry:Object.fromEntries([...selected].filter(id=>geometry[id]).map(id=>[id,structuredClone(geometry[id])])),
  };
}

export function pasteGraphSelection(pipeline,selection,offset={x:36,y:36}) {
  const idMap=new Map(),layout=readLayout(pipeline),pasted=[];
  for(const source of selection?.nodes??[]){
    const node={...structuredClone(source),id:`node:${cryptoId()}`};
    idMap.set(source.id,node.id);pipeline.nodes.push(node);pasted.push(node.id);
    const position=selection.positions?.[source.id]??{x:100,y:100};
    layout[node.id]={x:Math.round(position.x+offset.x),y:Math.round(position.y+offset.y)};
    if(selection?.faceplate_geometry?.[source.id]) setNodeFaceplateGeometry(pipeline,node.id,selection.faceplate_geometry[source.id]);
  }
  for(const source of selection?.edges??[]){
    const from=idMap.get(source.from.node_id),to=idMap.get(source.to.node_id);
    if(!from||!to)continue;
    pipeline.edges.push({...structuredClone(source),id:`edge:${cryptoId()}`,from:{...source.from,node_id:from},to:{...source.to,node_id:to}});
  }
  if(!pasted.length)return[];
  writeLayout(pipeline,layout);touch(pipeline);return pasted;
}

export function deleteGraphSelection(pipeline,nodeIds,edgeIds=[]) {
  const nodes=new Set(nodeIds),edges=new Set(edgeIds);
  if(!nodes.size&&!edges.size)return false;
  pipeline.nodes=pipeline.nodes.filter(node=>!nodes.has(node.id));
  pipeline.edges=pipeline.edges.filter(edge=>!edges.has(edge.id)&&!nodes.has(edge.from.node_id)&&!nodes.has(edge.to.node_id));
  pipeline.selected_sinks=pipeline.selected_sinks.filter(sink=>!nodes.has(sink.node_id));
  const layout=readLayout(pipeline),geometry=readNodeFaceplateGeometryMetadata(pipeline),collapsed=readNodeFaceplateMetadata(pipeline);
  nodes.forEach(id=>{
    delete layout[id];
    delete geometry[id];
    if(collapsed?.collapsed?.[id]) delete collapsed.collapsed[id];
  });
  if(collapsed?.collapsed&&Object.keys(collapsed.collapsed).length===0) delete collapsed.collapsed;
  writeLayout(pipeline,layout);writeNodeFaceplateGeometryMetadata(pipeline,geometry);writeNodeFaceplateMetadata(pipeline,collapsed);touch(pipeline);
  return true;
}

export function moveGraphSelection(pipeline,nodeIds,delta,{snap=0}={}) {
  const layout=readLayout(pipeline),ids=[...new Set(nodeIds)].filter(id=>layout[id]);
  if(!ids.length||(delta.x===0&&delta.y===0))return false;
  const rounded=value=>snap>0?Math.round(value/snap)*snap:Math.round(value);
  ids.forEach(id=>{layout[id]={x:rounded(layout[id].x+delta.x),y:rounded(layout[id].y+delta.y)};});
  writeLayout(pipeline,layout);touch(pipeline);return true;
}

export function alignGraphSelection(pipeline,nodeIds,axis,mode="start") {
  const layout=readLayout(pipeline),ids=[...new Set(nodeIds)].filter(id=>layout[id]);
  if(ids.length<2||!["x","y"].includes(axis))return false;
  const values=ids.map(id=>layout[id][axis]);
  const target=mode==="end"?Math.max(...values):mode==="center"?values.reduce((sum,value)=>sum+value,0)/values.length:Math.min(...values);
  ids.forEach(id=>{layout[id][axis]=Math.round(target);});
  writeLayout(pipeline,layout);touch(pipeline);return true;
}

export function distributeGraphSelection(pipeline,nodeIds,axis) {
  if(!["x","y"].includes(axis))return false;
  const layout=readLayout(pipeline),ids=[...new Set(nodeIds)].filter(id=>layout[id]).sort((left,right)=>layout[left][axis]-layout[right][axis]);
  if(ids.length<3)return false;
  const start=layout[ids[0]][axis],end=layout[ids.at(-1)][axis],step=(end-start)/(ids.length-1);
  ids.forEach((id,index)=>{layout[id][axis]=Math.round(start+step*index);});
  writeLayout(pipeline,layout);touch(pipeline);return true;
}

export function tidyGraphSelection(pipeline,nodeIds,{columns=3,gapX=290,gapY=220}={}) {
  const layout=readLayout(pipeline),ids=[...new Set(nodeIds)].filter(id=>layout[id]);
  if(ids.length<2)return false;
  const origin={x:Math.min(...ids.map(id=>layout[id].x)),y:Math.min(...ids.map(id=>layout[id].y))};
  ids.forEach((id,index)=>{layout[id]={x:origin.x+(index%columns)*gapX,y:origin.y+Math.floor(index/columns)*gapY};});
  writeLayout(pipeline,layout);touch(pipeline);return true;
}

export function ensureLayout(pipeline) {
  const layout=readLayout(pipeline);
  pipeline.nodes.forEach((node,index)=>{
    const row=Math.floor(index/3),column=row%2===0?index%3:2-(index%3);
    layout[node.id]??={x:130+column*290,y:140+row*220};
  });
  writeLayout(pipeline,layout);return layout;
}

export function diagnosticsByTarget(report) {
  const result={graph:[],nodes:{},ports:{},edges:{}};
  for(const item of report?.diagnostics??[]){
    const target=item.target??item;
    if(target.edge_id)(result.edges[target.edge_id]??=[]).push(item);
    else if(target.node_id&&target.port_id)(result.ports[`${target.node_id}:${target.port_id}`]??=[]).push(item);
    else if(target.node_id)(result.nodes[target.node_id]??=[]).push(item);
    else result.graph.push(item);
  }
  return result;
}

export function requiredPortState(pipeline,node,port,report) {
  const connected=pipeline.edges.some(edge=>edge.to.node_id===node.id&&edge.to.port_id===port.id);
  const diagnostics=diagnosticsByTarget(report).ports[`${node.id}:${port.id}`]??[];
  return {connected,missing:!connected&&port.direction==="input"&&port.cardinality==="one",diagnostics};
}

export function touch(pipeline){pipeline.revision=Math.max(1,Number(pipeline.revision)||1)+1;}

export async function consumeNdjson(reader,onEvent) {
  const decoder=new TextDecoder();let buffer="";
  while(true){
    const{done,value}=await reader.read();
    if(done)break;
    buffer+=decoder.decode(value,{stream:true});
    const lines=buffer.split("\n");buffer=lines.pop();
    for(const line of lines)if(line.trim())onEvent(JSON.parse(line));
  }
  buffer+=decoder.decode();
  if(buffer.trim())onEvent(JSON.parse(buffer));
}
