export const PIPELINE_SCHEMA_VERSION = 2;

export function buildCatalog(discovery) {
  const kinds = discovery.node_kinds ?? {};
  const nodes = [];
  for (const kind of Object.values(kinds)) {
    if (!kind.requires_component) {
      nodes.push({
        id:`kind:${kind.kind}`,kind:kind.kind,label:kind.label,component_id:null,
        config:structuredClone(kind.default_config ?? {}),ports:kind.ports ?? [],
      });
    }
  }
  for (const component of Object.values(discovery.components ?? {})) {
    const kind = kinds[component.node_kind];
    if (!kind) continue;
    nodes.push({
      id:`component:${component.id}`,kind:component.node_kind,
      label:`${kind.label} · ${component.provider} / ${component.model}`,
      component_id:component.id,
      config:structuredClone(component.default_config ?? kind.default_config ?? {}),
      ports:kind.ports ?? [],readiness:component.readiness,detail:component.detail,
    });
  }
  return nodes.sort((a,b)=>a.label.localeCompare(b.label));
}

export function createPipeline(name = "Untitled pipeline") {
  return {
    schema_version:PIPELINE_SCHEMA_VERSION,graph_id:`pipeline:${Date.now()}`,
    revision:1,metadata:{name,description:"",allow_unsafe_execution:false,labels:{}},
    nodes:[],edges:[],selected_sinks:[],
  };
}

export function addNode(pipeline, catalogNode, afterId = null) {
  const node = {
    id:`node:${Date.now()}:${pipeline.nodes.length}`,kind:catalogNode.kind,
    component_id:catalogNode.component_id,config:structuredClone(catalogNode.config ?? {}),
  };
  const index = afterId ? pipeline.nodes.findIndex(item => item.id === afterId) + 1 : pipeline.nodes.length;
  pipeline.nodes.splice(Math.max(0,index),0,node); pipeline.revision++; return node;
}

export function removeNode(pipeline, id) {
  pipeline.nodes = pipeline.nodes.filter(node => node.id !== id);
  pipeline.edges = pipeline.edges.filter(edge => edge.from.node_id !== id && edge.to.node_id !== id);
  pipeline.selected_sinks = pipeline.selected_sinks.filter(sink => sink.node_id !== id);
  pipeline.revision++;
}

export function duplicateNode(pipeline, id) {
  const source = pipeline.nodes.find(node => node.id === id);
  if (!source) return null;
  const copy = {...structuredClone(source),id:`node:${Date.now()}:${pipeline.nodes.length}`};
  pipeline.nodes.splice(pipeline.nodes.indexOf(source)+1,0,copy); pipeline.revision++; return copy;
}

export function replaceNode(pipeline, id, catalogNode) {
  const node = pipeline.nodes.find(item => item.id === id);
  if (!node || node.kind !== catalogNode.kind) throw new Error("Replacement must have the same backend node kind.");
  Object.assign(node,{component_id:catalogNode.component_id,config:structuredClone(catalogNode.config ?? {})});
  pipeline.revision++;
}

export function moveNode(pipeline, id, delta) {
  const index = pipeline.nodes.findIndex(node => node.id === id);
  const target = Math.max(0,Math.min(pipeline.nodes.length-1,index+delta));
  if (index < 0 || index === target) return;
  pipeline.nodes.splice(target,0,pipeline.nodes.splice(index,1)[0]); pipeline.revision++;
}

function kindFor(node, discovery) { return discovery.node_kinds?.[node?.kind]; }
function compatiblePorts(source, target, discovery) {
  const outputs=(kindFor(source,discovery)?.ports??[]).filter(port=>port.direction==="output");
  const inputs=(kindFor(target,discovery)?.ports??[]).filter(port=>port.direction==="input");
  return outputs.flatMap(output=>inputs.filter(input=>input.value_type===output.value_type).map(input=>({output,input})));
}

export function connect(pipeline, from, to, discovery) {
  const source=pipeline.nodes.find(node=>node.id===from), target=pipeline.nodes.find(node=>node.id===to);
  if (!source || !target) throw new Error("Both connection endpoints must exist.");
  const pairs=compatiblePorts(source,target,discovery);
  if (!pairs.length) {
    const outputs=(kindFor(source,discovery)?.ports??[]).filter(port=>port.direction==="output").map(port=>port.value_type);
    const inputs=(kindFor(target,discovery)?.ports??[]).filter(port=>port.direction==="input").map(port=>port.value_type);
    throw new Error(`${source.kind} emits ${outputs.join(" or ")||"nothing"}; ${target.kind} requires ${inputs.join(" or ")||"no input"}.`);
  }
  const {output,input}=pairs[0];
  if (input.cardinality !== "many") pipeline.edges=pipeline.edges.filter(edge=>!(edge.to.node_id===to&&edge.to.port_id===input.id));
  const id=`edge:${from}:${output.id}:${to}:${input.id}`;
  pipeline.edges=pipeline.edges.filter(edge=>edge.id!==id);
  pipeline.edges.push({id,from:{node_id:from,port_id:output.id},to:{node_id:to,port_id:input.id},capacity:16});
  pipeline.revision++;
}

export function nodeLabel(node, catalog) {
  return catalog.find(item=>item.kind===node.kind&&item.component_id===node.component_id)?.label ?? node.kind;
}
