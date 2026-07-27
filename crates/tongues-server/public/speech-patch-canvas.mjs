import {
  catalogEntryForNode,
  connectionCompatibility,
  connectPorts,
  portsFor,
  reconnectEdge,
  removeEdge,
  NODE_FACEPLATE_GEOMETRY_DEFAULT,
} from "./speech-dataflow-model.mjs";

const NODE_WIDTH = NODE_FACEPLATE_GEOMETRY_DEFAULT.width;
const NODE_HEIGHT = NODE_FACEPLATE_GEOMETRY_DEFAULT.height;
const PORT_SPACING = 22;
const AUTO_PAN_MARGIN = 44;
const CARD_RENDER_THROTTLE_MS = 90;
const CARD_COLLAPSED_HEIGHT = NODE_FACEPLATE_GEOMETRY_DEFAULT.collapsed_height;
const NODE_GEOMETRY_LIMITS = {width:{min:120,max:1400},height:{min:48,max:2400},collapsedHeight:{min:28,max:2400}};
const PORT_SPAN_PADDING = 34;

export function signalFamily(valueType) {
  if (String(valueType).startsWith("audio")) return "audio";
  if (String(valueType).startsWith("transcript") || valueType === "text") return "text";
  if (["utterance_plan", "control", "cancellation"].includes(valueType)) return "control";
  if (valueType === "error") return "error";
  if (String(valueType).includes("artifact")) return "artifact";
  return "data";
}

export function cablePath(from, to, points = [], routing = "curved") {
  const route=[from,...points,to];
  if(routing==="straight")return route.map((point,index)=>`${index?"L":"M"} ${point.x} ${point.y}`).join(" ");
  if(routing==="orthogonal"){
    let path=`M ${from.x} ${from.y}`;
    for(const point of route.slice(1)){const previous=route[route.indexOf(point)-1],middle=(previous.x+point.x)/2;path+=` L ${middle} ${previous.y} L ${middle} ${point.y} L ${point.x} ${point.y}`;}
    return path;
  }
  if(points.length){
    let path=`M ${from.x} ${from.y}`;
    for(let index=1;index<route.length;index++){const previous=route[index-1],point=route[index],distance=Math.max(28,Math.abs(point.x-previous.x)*.35);path+=` C ${previous.x+distance} ${previous.y}, ${point.x-distance} ${point.y}, ${point.x} ${point.y}`;}
    return path;
  }
  const distance = Math.max(56, Math.abs(to.x - from.x) * 0.45);
  return `M ${from.x} ${from.y} C ${from.x + distance} ${from.y}, ${to.x - distance} ${to.y}, ${to.x} ${to.y}`;
}

export function portAnchor(position, ports, index, direction, geometry = NODE_FACEPLATE_GEOMETRY_DEFAULT, collapsed = false) {
  const width = geometry?.width ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.width;
  const height = collapsed ? (geometry?.collapsed_height ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.collapsed_height)
    : (geometry?.height ?? NODE_FACEPLATE_GEOMETRY_DEFAULT.height);
  const count = Math.max(1, ports.length);
  const span = Math.min(Math.max(0, height - PORT_SPAN_PADDING), (count - 1) * PORT_SPACING);
  const y = position.y - span / 2 + (count === 1 ? 0 : (span * index) / (count - 1));
  return {
    x: position.x + (direction === "output" ? width / 2 : -width / 2),
    y,
  };
}

function readableType(value) {
  return String(value ?? "unknown").replaceAll("_", " ");
}

function editableTarget(target) {
  return Boolean(target?.closest?.("input, textarea, select, [contenteditable=true]"));
}

function valueText(value) {
  return value === null || value === undefined ? "" : Array.isArray(value) || typeof value === "object"
    ? JSON.stringify(value)
    : String(value);
}

function injectStyles(document) {
  if (document.querySelector("style[data-speech-patch-canvas]")) return;
  const style = document.createElement("style");
  style.dataset.speechPatchCanvas = "";
  style.textContent = `
    .patch-cables,.patch-jacks{position:absolute;inset:0;z-index:2;pointer-events:none}
    .patch-organization{position:absolute;inset:0;z-index:1;pointer-events:none;overflow:hidden}
    .patch-frame{position:absolute;border:2px solid var(--frame-color,#527084);background:color-mix(in srgb,var(--frame-color,#527084) 12%,transparent);border-radius:.65rem;color:#edf5ff;padding:.45rem;font-weight:700}
    .patch-note{position:absolute;max-width:15rem;padding:.45rem .6rem;border:1px solid #8f8050;border-radius:.35rem;background:var(--note-color,#594c2c);color:#fff7d8;white-space:pre-wrap;box-shadow:0 5px 12px #0007}
    .patch-subpatch-summary{position:absolute;transform:translate(-50%,-50%);min-width:15rem;padding:.65rem;border:2px solid #dca3ff;border-radius:.65rem;background:#241d36ee;color:#fff;box-shadow:0 8px 20px #000a}.patch-subpatch-summary small{display:block;color:#cabce0;margin-top:.2rem}
    .patch-cables{width:100%;height:100%;overflow:visible}
    .patch-cable{fill:none;stroke:#8da4ba;stroke-width:4;pointer-events:none}
    .patch-cable-hit{fill:none;stroke:transparent;stroke-width:18;pointer-events:stroke;cursor:pointer}
    .patch-reroute{fill:#101923;stroke:#f7fffd;stroke-width:2;pointer-events:all;cursor:move}
    .patch-cable.signal-audio{stroke:#70d6a4;stroke-width:5}
    .patch-cable.signal-text{stroke:#dca3ff;stroke-dasharray:10 5}
    .patch-cable.signal-control{stroke:#ffca70;stroke-dasharray:3 5}
    .patch-cable.signal-error{stroke:#ff8c91;stroke-dasharray:2 4}
    .patch-cable.signal-artifact{stroke:#72b7ff;stroke-dasharray:14 4 3 4}
    .patch-cable.edge-state-active{filter:drop-shadow(0 0 7px #76e2ce);stroke-width:5}
    .patch-cable.edge-state-failed{stroke:#ff8c91;opacity:.88}
    .patch-cable.edge-state-ready{opacity:1}
    .patch-cable.signal-data{stroke:#a6b7ca}
    .patch-cable.selected{stroke:#f7fffd;stroke-width:7;filter:drop-shadow(0 0 5px #76e2ce)}
    .patch-cable.invalid{stroke:#ffc86b}
    .patch-cable-preview{fill:none;stroke:#f7fffd;stroke-width:4;stroke-dasharray:8 5;pointer-events:none}
    .patch-node-cards{position:absolute;inset:0;z-index:1;pointer-events:none}
    .patch-node-card{position:absolute;transform:translate(-50%,-50%);width:var(--patch-node-card-width,${NODE_WIDTH}px);box-sizing:border-box;background:#162636e8;border:1px solid #4a6380;border-radius:.42rem;padding:.38rem .45rem .42rem;box-shadow:0 8px 20px #000b;backdrop-filter:blur(2px);pointer-events:none;overflow:hidden;color:#edf5ff}
    .patch-node-card[data-state=collapsed]{height:var(--patch-node-card-collapsed-height,${CARD_COLLAPSED_HEIGHT}px);max-height:var(--patch-node-card-collapsed-height,${CARD_COLLAPSED_HEIGHT}px)}
    .patch-node-card[data-state=expanded]{max-height:22rem}
    .patch-node-card .patch-node-card-title{display:flex;align-items:start;justify-content:space-between;gap:.35rem}
    .patch-node-card .patch-node-card-title strong{font-size:.76rem;line-height:1.15}
    .patch-node-card .patch-node-card-title small{display:block;color:#9bb2c9;font-size:.68rem;line-height:1.2}
    .patch-node-card .patch-node-card-meta{margin-top:.15rem;color:#9bb2c9;font-size:.66rem;display:grid;gap:.16rem}
    .patch-node-card .patch-node-card-status{display:inline-flex;align-items:center;gap:.3rem;font-size:.66rem;padding:.1rem .33rem;border-radius:999px;border:1px solid #435a73}
    .patch-node-card .patch-node-card-status.ready{border-color:#76e2ce;color:#76e2ce}
    .patch-node-card .patch-node-card-status.loading{border-color:#dca3ff;color:#dca3ff}
    .patch-node-card .patch-node-card-status.active{border-color:#ffca70;color:#ffca70}
    .patch-node-card .patch-node-card-status.failed{border-color:#ff8c91;color:#ff8c91}
    .patch-node-card .patch-node-card-status.inactive{border-color:#9caec6;color:#9caec6}
    .patch-node-card .patch-node-card-status.bypassed{border-color:#ffc86b;color:#ffc86b}
    .patch-node-card .patch-node-card-jacks{display:grid;gap:.1rem;font-size:.66rem;color:#95aac0}
    .patch-node-card .patch-node-card-jacks ul{margin:.08rem 0 0;padding-left:1rem}
    .patch-node-card .patch-node-card-runtime{margin-top:.2rem;padding:.14rem .3rem;background:#0f1823;border-radius:.3rem;font-size:.64rem;color:#dceaff;white-space:nowrap}
    .patch-node-card .patch-node-card-actions{display:flex;gap:.28rem;flex-wrap:wrap;margin-top:.25rem}
    .patch-node-card .patch-node-card-control{display:grid;gap:.18rem}
    .patch-node-card .patch-node-card-control>span{display:block;color:#9cb2c9;font-size:.64rem}
    .patch-node-card .patch-node-card-control select,
    .patch-node-card .patch-node-card-control input{font-size:.7rem;padding:.22rem .3rem;border-radius:.32rem}
    .patch-node-card .patch-node-card-control input[type=range]{padding:0}
    .patch-node-card .patch-node-card-preview{margin-top:.2rem;color:#b8cce0;font-size:.62rem;line-height:1.2}
    .patch-node-card button{font-size:.66rem;padding:.22rem .45rem}
    .patch-node-card .patch-node-card-controls{display:grid;gap:.2rem}
    .patch-node-card .patch-node-card-meter{margin-top:.16rem;height:.2rem;border-radius:999px;background:#0a1321;overflow:hidden;border:1px solid #40536a}
    .patch-node-card .patch-node-card-meter span{display:block;height:100%;background:#70d6a4;width:0}
    .patch-node-card .patch-node-card-badge{font-size:.62rem;padding:.1rem .3rem;border-radius:999px;background:#0a1321;border:1px solid #2c435e;display:inline-flex}
    .patch-node-card .patch-node-card-toggle{white-space:nowrap}
    .patch-node-card .patch-node-card-controls:focus-within{outline:2px solid #76e2ce;outline-offset:2px}
    .patch-node-card button,.patch-node-card input,.patch-node-card select,.patch-node-card textarea{pointer-events:auto}
    .patch-node-card[data-readonly=true]{opacity:.86}
    .patch-node-card[data-node-bypassed=true],.patch-node-card[data-node-disabled=true]{opacity:.72}
    .patch-jack-wrap{position:absolute;display:flex;align-items:center;gap:.28rem;pointer-events:none;transform:translateY(-50%)}
    .patch-jack-wrap.input{flex-direction:row}.patch-jack-wrap.output{flex-direction:row-reverse;transform:translate(-100%,-50%)}
    .patch-jack-label{max-width:5.9rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:.08rem .25rem;border-radius:.25rem;background:#101923dc;color:#edf5ff;font-size:.68rem;line-height:1.25}
    .patch-jack{width:1.3rem;height:1.3rem;min-width:1.3rem;padding:0;border:3px solid #d8e5f2;background:#1a2633;pointer-events:auto;box-shadow:0 1px 5px #000b}
    .patch-jack.input{border-radius:50%}.patch-jack.output{border-radius:.16rem;transform:rotate(45deg)}
    .patch-jack.connected{background:#edf5ff}.patch-jack.signal-audio{border-color:#70d6a4}.patch-jack.signal-text{border-color:#dca3ff}
    .patch-jack.signal-control{border-color:#ffca70}.patch-jack.signal-error{border-color:#ff8c91}
    .patch-jack.signal-artifact{border-color:#72b7ff}.patch-jack.compatible{outline:4px solid #76e2ce;outline-offset:3px}
    .patch-jack.incompatible{opacity:.48}.patch-jack.drag-origin{outline:4px solid #f7fffd;outline-offset:3px}
    .patch-jack:focus-visible{outline:4px solid #76e2ce;outline-offset:3px;opacity:1}
    .patch-connection-list{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}
    #canvas[data-patching=true]{box-shadow:inset 0 0 0 3px #76e2ce}
  `;
  document.head.append(style);
}

function svgElement(document, name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

export function createPatchCanvas(options) {
  const {
    container,
    cy,
    getPipeline,
    getDiscovery,
    getCatalog,
    nodeLabel,
    nodeIcon = nodeLabel,
    getSelectedEdgeId,
    isEdgeSelected = id => getSelectedEdgeId() === id,
    onSelectNode,
    onSelectEdge,
    onGraphEdit,
    onDropEmpty,
    onDropCatalogOnEdge,
    onDropCatalogOnJack,
    getNodeRuntimeState = () => null,
    getEdgeRuntimeState = () => null,
    getNodeControlState = () => ({}),
    getNodeFaceplateGeometry = () => NODE_FACEPLATE_GEOMETRY_DEFAULT,
    onSetNodeFaceplateGeometry = () => {},
    isNodeCollapsed = () => false,
    onSetNodeCollapsed = () => {},
    onNodeConfigChange = () => {},
    canBypassNode = () => false,
    onBypassNode = () => {},
    onDisableNode = () => {},
    isRunLocked = () => false,
    getVisibleNodeIds = () => null,
    onAnnounce,
  } = options;
  const document = container.ownerDocument;
  const window = document.defaultView;
  injectStyles(document);

  const svg = svgElement(document, "svg", {
    class: "patch-cables",
    "aria-hidden": "true",
  });
  const organizationLayer=document.createElement("div");
  organizationLayer.className="patch-organization";
  const jackLayer = document.createElement("div");
  jackLayer.className = "patch-jacks";
  const cardLayer = document.createElement("div");
  cardLayer.className = "patch-node-cards";
  const connectionList = document.createElement("ol");
  connectionList.className = "patch-connection-list";
  connectionList.setAttribute("aria-label", "Graph connections");
  container.parentElement.append(organizationLayer,svg, jackLayer, cardLayer, connectionList);

  let gesture = null;
  let previewPoint = null;
  let pointerId = null;
  let destroyed = false;
  let cardRenderAt = 0;
  let cardFrameRequested = false;

  const pipeline = () => getPipeline();
  const discovery = () => getDiscovery();
  const catalog = () => getCatalog();
  const label = nodeId => nodeLabel(pipeline().nodes.find(node => node.id === nodeId), catalog());
  const cardTitle = nodeId => {
    const node = pipeline().nodes.find(node => node.id === nodeId);
    return nodeIcon(node, catalog()) ?? "";
  };
  const cardRuntime = nodeId => getNodeRuntimeState(nodeId);
  const cardControlState = nodeId => getNodeControlState(nodeId);
  const edgeRuntime = edgeId => getEdgeRuntimeState(edgeId);
  const geometryState = new Map();

  const clampNodeDimension = (value, limits) => {
    if (!Number.isFinite(value)) return null;
    return Math.max(limits.min, Math.min(limits.max, Math.round(value)));
  };
  const canonicalGeometry = (raw) => {
    const width = clampNodeDimension(Number(raw?.width), NODE_GEOMETRY_LIMITS.width);
    const height = clampNodeDimension(Number(raw?.height), NODE_GEOMETRY_LIMITS.height);
    const collapsedHeight = clampNodeDimension(Number(raw?.collapsed_height), NODE_GEOMETRY_LIMITS.collapsedHeight);
    return {
      width: width ?? NODE_WIDTH,
      height: height ?? NODE_HEIGHT,
      collapsed_height: collapsedHeight ?? CARD_COLLAPSED_HEIGHT,
    };
  };
  const geometryForNode = (nodeId, collapsed) => {
    const raw = getNodeFaceplateGeometry(nodeId) ?? NODE_FACEPLATE_GEOMETRY_DEFAULT;
    const geometry = canonicalGeometry(raw);
    if (collapsed) geometry.height = geometry.collapsed_height;
    return geometry;
  };
  const geometrySignature = (geometry, collapsed) => {
    const candidate = collapsed
      ? {width:geometry.width,height:geometry.collapsed_height}
      : {width:geometry.width,height:geometry.height};
    return JSON.stringify(candidate);
  };
  const geometryMatchesState = (nodeId, next, collapsed) => geometryState.get(nodeId) === geometrySignature(next, collapsed);
  const rememberGeometry = (nodeId, geometry, collapsed) => {
    geometryState.set(nodeId, geometrySignature(geometry, collapsed));
  };
  const pushNodeGeometryToCanvas = (nodeId, geometry, collapsed) => {
    const node = cy.getElementById(nodeId);
    if (!node?.length) return;
    const width = Math.max(1, Math.round(geometry.width));
    const height = Math.max(1, Math.round(collapsed ? geometry.collapsed_height : geometry.height));
    node.data("width", width);
    node.data("height", height);
  };
  const persistNodeGeometryFromCard = (node, card, collapsed) => {
    const nodeId = node.id;
    const rect = card.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const raw = getNodeFaceplateGeometry(nodeId);
    const current = canonicalGeometry(raw);
    const next = collapsed
      ? {
        ...current,
        width: Math.max(1, Math.round(rect.width)),
        collapsed_height: Math.max(1, Math.round(rect.height)),
      }
      : {
        ...current,
        width: Math.max(1, Math.round(rect.width)),
        height: Math.max(1, Math.round(rect.height)),
      };
    if (!geometryMatchesState(nodeId, next, collapsed)) onSetNodeFaceplateGeometry(nodeId, next);
    rememberGeometry(nodeId, next, collapsed);
    pushNodeGeometryToCanvas(nodeId, next, collapsed);
  };

  function nodePorts(node, direction) {
    return portsFor(node, direction, discovery());
  }

  function nodeCardSchema(node) {
    const item = catalogEntryForNode(node, catalog());
    return item?.schema?.properties ?? {};
  }

  function controlUiHint(fieldSpec) {
    return fieldSpec?.["x-ui-widget"] || (fieldSpec?.enum?.length ? "menu" : null)
      || (fieldSpec?.type === "boolean" ? "toggle" : fieldSpec?.type === "number" || fieldSpec?.type === "integer" ? "number" : "short_text");
  }

  function controlPriority(fieldSpec) {
    const value = Number(fieldSpec?.["x-ui-priority"]);
    if (Number.isFinite(value)) return value;
    return 50;
  }

  function controlOptions(fieldSpec) {
    if (!Array.isArray(fieldSpec?.enum) || !fieldSpec.enum.length) return [];
    return fieldSpec.enum.map(value => ({value, label: String(value)}));
  }

  function formatNodeRole(node, item) {
    const fallback = node.kind;
    return item?.provider
      ? `${item.provider} / ${item.model}`
      : discovery()?.node_kinds?.[node.kind]?.label ?? fallback;
  }

  function buildControlLabel(node, field, fieldSpec, value) {
    const nodeName = node?.kind ?? node?.id ?? "module";
    const unit = fieldSpec?.["x-ui-unit"] ? ` (${fieldSpec["x-ui-unit"]})` : "";
    const range = [fieldSpec?.minimum, fieldSpec?.maximum].some(value => value != null)
      ? `, ${fieldSpec.minimum ?? "unbounded"} to ${fieldSpec.maximum ?? "unbounded"}`
      : "";
    return `${nodeName} ${field}: ${valueText(value)}${unit}${range}`;
  }

  function parseNodeControlValue(spec, raw) {
    if (spec?.type === "boolean") return Boolean(raw === true || raw === "true" || raw === "on");
    if (spec?.type === "number" || spec?.type === "integer") {
      const number = Number(raw);
      if (!Number.isFinite(number)) throw new Error("Enter a numeric value.");
      if (spec.type === "integer" && !Number.isInteger(number)) throw new Error("Enter an integer.");
      return number;
    }
    if (spec?.type === "array" || spec?.type === "object") {
      if (raw === "" || raw == null) return raw;
      return typeof raw === "string" ? JSON.parse(raw) : raw;
    }
    return raw;
  }

  function renderControl(node, field, fieldSpec, containerNode) {
    const value = node.config?.[field];
    const state = cardControlState(node.id);
    const control = controlUiHint(fieldSpec);
    const options = controlOptions(fieldSpec);
    const wrapper = document.createElement("label");
    wrapper.className = "patch-node-card-control";
    const label = document.createElement("span");
    label.textContent = `${fieldSpec?.title ?? field} ${fieldSpec?.["x-ui-unit"] ? `(${fieldSpec["x-ui-unit"]})` : ""}`;
    const applyConfig = () => {
      let parsed = input.value;
      try {
        parsed = parseNodeControlValue(fieldSpec, input.value);
      } catch (error) {
        if (error?.message) onAnnounce(`${error.message}`, true);
        return;
      }
      if (parsed === value) return;
      try {
        onNodeConfigChange({nodeId: node.id, field, value: parsed});
      } catch (error) {
        if (error?.message) onAnnounce(`${error.message}`, true);
        if (state?.invalid) onAnnounce(`Update for ${node.id} ${field} was rejected.`, true);
      }
    };
    let input;
    if (control === "toggle") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(value);
      input.onchange = () => onNodeConfigChange({nodeId: node.id, field, value: input.checked});
    } else if (control === "menu") {
      input = document.createElement("select");
      options.forEach(item => input.append(new Option(String(item.label), String(item.value))));
      input.value = value == null ? "" : String(value);
      if (!options.some(item => String(item.value) === input.value)) {
        input.value = options[0]?.value ? String(options[0].value) : "";
      }
      input.onchange = applyConfig;
    } else if (control === "slider" && Number.isFinite(fieldSpec.minimum) && Number.isFinite(fieldSpec.maximum)) {
      input = document.createElement("input");
      input.type = "range";
      input.min = String(fieldSpec.minimum);
      input.max = String(fieldSpec.maximum);
      if (fieldSpec.step != null) input.step = String(fieldSpec.step);
      input.value = Number.isFinite(Number(value)) ? String(value) : String((fieldSpec.minimum + fieldSpec.maximum) / 2);
      input.oninput = applyConfig;
      input.onchange = applyConfig;
    } else {
      input = document.createElement("input");
      input.type = fieldSpec?.type === "number" || fieldSpec?.type === "integer" ? "number" : "text";
      if (Number.isFinite(fieldSpec.minimum)) input.min = String(fieldSpec.minimum);
      if (Number.isFinite(fieldSpec.maximum)) input.max = String(fieldSpec.maximum);
      input.value = value == null ? "" : valueText(value);
      input.onchange = applyConfig;
    }
    input.dataset.nodeId = node.id;
    input.dataset.configField = field;
    input.id = `node-card-${node.id}-${field}`;
    wrapper.setAttribute("for", input.id);
    input.setAttribute("aria-label", buildControlLabel(node, field, fieldSpec, input.value));
    input.setAttribute("title", fieldSpec?.description ?? "");
    wrapper.append(label, input);
    containerNode.append(wrapper);
  }

  function renderNodeCard(node) {
    const cyNode = cy.getElementById(node.id);
    if (!cyNode?.length) return null;
    const entry = catalogEntryForNode(node, catalog());
    const schemaProperties = nodeCardSchema(node);
    const runtime = cardRuntime(node.id) ?? {};
    const collapsed = Boolean(isNodeCollapsed(node.id));
    const ports = discovery()?.node_kinds?.[node.kind]?.ports ?? [];
    const geometry = geometryForNode(node.id, collapsed);
    const inputs = ports.filter(port => port.direction === "input");
    const outputs = ports.filter(port => port.direction === "output");
    const cardDisplayTitle = cardTitle(node.id) ?? (entry?.label ? entry.label.split(" · ")[0] : node.kind);
    const nodeRole = formatNodeRole(node, entry);
    const status = runtime?.status ?? (node.disabled ? "inactive" : node.bypassed ? "bypassed" : "ready");
    const card = document.createElement("article");
    card.dataset.nodeCard = node.id;
    card.className = "patch-node-card";
    card.dataset.nodeId = node.id;
    card.dataset.nodeCollapsed = String(collapsed);
    card.dataset.state = collapsed ? "collapsed" : "expanded";
    card.dataset.nodeBypassed = String(Boolean(node.bypassed));
    card.dataset.nodeDisabled = String(Boolean(node.disabled));
    card.style.setProperty("--patch-node-card-width", `${Math.max(1, Math.round(geometry.width))}px`);
    card.style.setProperty("--patch-node-card-collapsed-height", `${Math.max(1, Math.round(geometry.collapsed_height))}px`);
    card.style.left = `${cyNode.renderedPosition().x}px`;
    card.style.top = `${cyNode.renderedPosition().y}px`;
    const position = nodeRole ? nodeRole : "";
    const title = document.createElement("div");
    title.className = "patch-node-card-title";
    const name = document.createElement("strong");
    name.textContent = cardDisplayTitle;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "patch-node-card-toggle";
    toggle.textContent = collapsed ? "Expand" : "Collapse";
    toggle.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${node.kind} faceplate`);
    toggle.onclick = () => onSetNodeCollapsed(node.id, !collapsed);
    title.append(name);
    title.append(toggle);
    card.append(title);

    const meta = document.createElement("div");
    meta.className = "patch-node-card-meta";
    const roleLabel = document.createElement("small");
    roleLabel.textContent = position;
    const state = document.createElement("span");
    state.className = `patch-node-card-status ${status}`;
    const latency = runtime?.lastElapsedMs != null ? ` · ${runtime.lastElapsedMs} ms` : "";
    state.textContent = node.disabled
      ? "disabled"
      : node.bypassed
      ? "bypassed"
      : runtime?.status
      ? `${runtime.status}${latency}`
      : "ready";
    meta.append(roleLabel, state);
    card.append(meta);

    const controlsPanel = document.createElement("section");
    controlsPanel.className = "patch-node-card-controls";
    const jacks = document.createElement("section");
    jacks.className = "patch-node-card-jacks";
    jacks.innerHTML = `<span>IO</span><ul><li>in: ${inputs.slice(0, 2).map(port => port.label ?? port.id).join(", ") || "none"}${
      inputs.length > 2 ? ` +${inputs.length - 2}` : ""
    }</li><li>out: ${outputs.slice(0, 2).map(port => port.label ?? port.id).join(", ") || "none"}${
      outputs.length > 2 ? ` +${outputs.length - 2}` : ""
    }</li></ul>`;
    controlsPanel.append(jacks);

    const actions = document.createElement("div");
    actions.className = "patch-node-card-actions";
    const disableLabel = node.disabled ? "Enable" : "Disable";
    const disable = document.createElement("button");
    disable.type = "button";
    disable.textContent = disableLabel;
    disable.setAttribute("aria-label", `${disableLabel} ${node.kind}`);
    disable.onclick = () => onDisableNode(node.id, !node.disabled);
    actions.append(disable);
    if (canBypassNode(node.id)) {
      const bypass = document.createElement("button");
      bypass.type = "button";
      bypass.textContent = node.bypassed ? "Unbypass" : "Bypass";
      bypass.setAttribute("aria-label", `${node.bypassed ? "Unbypass" : "Bypass"} ${node.kind}`);
      bypass.onclick = () => onBypassNode(node.id);
      actions.append(bypass);
    }
    const preview = document.createElement("p");
    preview.className = "patch-node-card-preview";
    if (runtime?.error) preview.textContent = runtime.error;
    else if (runtime?.preview) preview.textContent = runtime.preview;
    controlsPanel.append(preview);
    if (!collapsed) {
      const fields = Object.entries(schemaProperties).map(([field, spec]) => ({
        field,
        spec,
        priority: controlPriority(spec),
        kind: controlUiHint(spec),
      })).filter(item => item.spec && ["toggle", "menu", "slider", "number", "short_text"].includes(item.kind))
        .sort((left, right) => left.priority - right.priority || left.field.localeCompare(right.field))
        .slice(0, 3);
      fields.forEach(item => renderControl(node, item.field, item.spec, controlsPanel));
      card.append(controlsPanel);
      card.append(actions);
      if (status === "active" && Number.isFinite(runtime.meter) && runtime.kind === "audio") {
        const meter = document.createElement("div");
        meter.className = "patch-node-card-meter";
        const bar = document.createElement("span");
        bar.style.width = `${Math.max(0, Math.min(100, runtime.meter * 100))}%`;
        meter.append(bar);
        card.append(meter);
      }
    }
    if (!collapsed && runtime?.kind === "control") {
      const badge = document.createElement("span");
      badge.className = "patch-node-card-badge";
      badge.textContent = `events ${runtime.pulse ?? 0}`;
      controlsPanel.append(badge);
    }

    if (collapsed) card.append(actions);
    return card;
  }

  function renderNodeCards() {
    const now = window.performance.now();
    if (now - cardRenderAt < CARD_RENDER_THROTTLE_MS) {
      if (!cardFrameRequested) {
        cardFrameRequested = true;
        window.requestAnimationFrame(() => {
          cardFrameRequested = false;
          renderNodeCards();
        });
      }
      return;
    }
    cardRenderAt = now;
    cardLayer.replaceChildren();
    const visibleIds=getVisibleNodeIds();
    for (const node of pipeline().nodes.filter(node=>!visibleIds||visibleIds.has(node.id))) {
      const card = renderNodeCard(node);
      if (!card) continue;
      const collapsed = card.dataset.state === "collapsed";
      cardLayer.append(card);
      persistNodeGeometryFromCard(node, card, collapsed);
    }
  }

  function anchor(endpoint, direction) {
    const node = pipeline().nodes.find(item => item.id === endpoint.node_id);
    const element = cy.getElementById(endpoint.node_id);
    if (!node || !element?.length) return null;
    const ports = nodePorts(node, direction);
    const index = ports.findIndex(port => port.id === endpoint.port_id);
    if (index < 0) return null;
    const collapsed = Boolean(isNodeCollapsed(endpoint.node_id));
    return portAnchor(element.renderedPosition(), ports, index, direction, geometryForNode(endpoint.node_id, collapsed), collapsed);
  }

  function edgeDescription(edge) {
    const source = pipeline().nodes.find(node => node.id === edge.from.node_id);
    const output = nodePorts(source, "output").find(port => port.id === edge.from.port_id);
    return `${label(edge.from.node_id)} ${output?.label ?? edge.from.port_id}, ${readableType(output?.value_type)}, connected to ${label(edge.to.node_id)} ${edge.to.port_id}`;
  }

  function renderConnections() {
    svg.replaceChildren();
    connectionList.replaceChildren();
    const grouped = options.diagnosticsByEdge?.() ?? {};
    const visibleIds=getVisibleNodeIds();
    for (const edge of pipeline().edges.filter(edge=>!visibleIds||(visibleIds.has(edge.from.node_id)&&visibleIds.has(edge.to.node_id)))) {
      const from = anchor(edge.from, "output");
      const to = anchor(edge.to, "input");
      if (!from || !to) continue;
      const source = pipeline().nodes.find(node => node.id === edge.from.node_id);
      const output = nodePorts(source, "output").find(port => port.id === edge.from.port_id);
      const runtime = edgeRuntime(edge.id);
      const edgeState = runtime?.status ?? "ready";
      const cable=pipeline().presentation?.cables?.[edge.id]??{};
      const pan=cy.pan?.()??{x:0,y:0},zoom=cy.zoom?.()??1;
      const points=(cable.reroute_points??[]).map(point=>({x:point.x*zoom+pan.x,y:point.y*zoom+pan.y}));
      const path = cablePath(from, to, points, cable.routing??"curved");
      const hit = svgElement(document, "path", {d: path, class: "patch-cable-hit", "data-edge-id": edge.id});
      const visible = svgElement(document, "path", {
        d: path,
        "data-edge-id": edge.id,
        class: `patch-cable signal-${signalFamily(output?.value_type)}${edgeState ? ` edge-state-${edgeState}` : ""}${isEdgeSelected(edge.id) ? " selected" : ""}${grouped[edge.id]?.length ? " invalid" : ""}`,
      });
      const focused=pipeline().presentation?.selected_path_focus&&getSelectedEdgeId();
      visible.style.opacity=String(isEdgeSelected(edge.id)||cable.emphasized?1:focused?.15:pipeline().presentation?.global_cable_opacity??1);
      hit.addEventListener("pointerdown", event => {
        event.stopPropagation();
        onSelectEdge(edge.id);
        render();
      });
      hit.addEventListener("dragover",event=>{if(event.dataTransfer?.types.includes("application/x-tongues-catalog-id"))event.preventDefault();});
      hit.addEventListener("drop",event=>{
        const catalogId=event.dataTransfer?.getData("application/x-tongues-catalog-id");if(!catalogId)return;
        event.preventDefault();onDropCatalogOnEdge?.({catalog_id:catalogId,edge_id:edge.id,clientX:event.clientX,clientY:event.clientY});
      });
      svg.append(hit, visible);
      if(isEdgeSelected(edge.id))points.forEach((point,index)=>svg.append(svgElement(document,"circle",{cx:point.x,cy:point.y,r:7,class:"patch-reroute","data-edge-id":edge.id,"data-reroute-index":index,"aria-label":`Cable reroute point ${index+1}`})));

      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.edgeId = edge.id;
      button.textContent = edgeDescription(edge);
      button.onclick = () => onSelectEdge(edge.id);
      button.onkeydown = event => {
        if (!["Delete", "Backspace"].includes(event.key)) return;
        event.preventDefault();
        const before = structuredClone(pipeline());
        removeEdge(pipeline(), edge.id);
        onGraphEdit({kind: "edge.delete", edge_id: edge.id, before});
        onAnnounce("Connection deleted.");
        render();
      };
      item.append(button);
      connectionList.append(item);
    }
    renderPreview();
  }

  function renderOrganizationLayer(){
    organizationLayer.replaceChildren();
    const pan=cy.pan?.()??{x:0,y:0},zoom=cy.zoom?.()??1;
    for(const frame of pipeline().presentation?.frames??[]){
      const element=document.createElement("section");element.className="patch-frame";element.textContent=frame.title;
      element.style.setProperty("--frame-color",frame.color||"#527084");
      element.style.left=`${frame.origin.x*zoom+pan.x}px`;element.style.top=`${frame.origin.y*zoom+pan.y}px`;
      element.style.width=`${frame.size.x*zoom}px`;element.style.height=`${frame.size.y*zoom}px`;
      organizationLayer.append(element);
    }
    for(const note of pipeline().presentation?.notes??[]){
      const element=document.createElement("aside");element.className="patch-note";element.textContent=note.text;
      element.style.setProperty("--note-color",note.color||"#594c2c");
      element.style.left=`${note.position.x*zoom+pan.x}px`;element.style.top=`${note.position.y*zoom+pan.y}px`;
      organizationLayer.append(element);
    }
    const collapsed=new Set(pipeline().presentation?.collapsed_subpatches??[]);
    for(const subpatch of pipeline().subpatches?.filter(item=>collapsed.has(item.id))??[]){
      const points=subpatch.node_ids.map(id=>cy.getElementById(id)).filter(node=>node?.length).map(node=>node.renderedPosition());
      if(!points.length)continue;
      const states=subpatch.node_ids.map(id=>getNodeRuntimeState(id)?.status);
      const unavailable=subpatch.node_ids.filter(id=>{
        const node=pipeline().nodes.find(item=>item.id===id);
        return catalogEntryForNode(node,catalog())?.readiness!=="ready";
      }).length;
      const element=document.createElement("section");element.className="patch-subpatch-summary";
      element.style.left=`${points.reduce((sum,point)=>sum+point.x,0)/points.length}px`;element.style.top=`${points.reduce((sum,point)=>sum+point.y,0)/points.length}px`;
      const title=document.createElement("strong");title.textContent=subpatch.title;
      const detail=document.createElement("small");detail.textContent=`${subpatch.node_ids.length} nodes · ${subpatch.exposed_ports.length} ports · ${unavailable} unavailable · ${states.filter(state=>state==="active").length} active · ${states.filter(state=>state==="failed").length} failed`;
      element.append(title,detail);organizationLayer.append(element);
    }
  }

  function portConnections(nodeId, portId, direction) {
    return pipeline().edges.filter(edge => {
      const endpoint = direction === "output" ? edge.from : edge.to;
      return endpoint.node_id === nodeId && endpoint.port_id === portId;
    });
  }

  function jackDescription(node, port, connections) {
    const state = connections.length
      ? `${connections.length} connection${connections.length === 1 ? "" : "s"}: ${connections.map(edgeDescription).join("; ")}`
      : "not connected";
    return `${label(node.id)}, ${port.direction} ${port.label ?? port.id}, ${readableType(port.value_type)}, ${port.cardinality ?? "one"}, ${state}`;
  }

  function renderJacks() {
    jackLayer.replaceChildren();
    const visibleIds=getVisibleNodeIds();
    for (const node of pipeline().nodes.filter(node=>!visibleIds||visibleIds.has(node.id))) {
      const cyNode = cy.getElementById(node.id);
      if (!cyNode?.length) continue;
      for (const direction of ["input", "output"]) {
      const ports = nodePorts(node, direction);
      ports.forEach((port, index) => {
        const collapsed = Boolean(isNodeCollapsed(node.id));
        const position = portAnchor(cyNode.renderedPosition(), ports, index, direction, geometryForNode(node.id, collapsed), collapsed);
        const connections = portConnections(node.id, port.id, direction);
        const wrap = document.createElement("span");
          wrap.className = `patch-jack-wrap ${direction}`;
          wrap.style.left = `${position.x}px`;
          wrap.style.top = `${position.y}px`;
          const button = document.createElement("button");
          button.type = "button";
          button.className = `patch-jack ${direction} signal-${signalFamily(port.value_type)}${connections.length ? " connected" : ""}`;
          button.dataset.patchJack = "";
          button.dataset.nodeId = node.id;
          button.dataset.portId = port.id;
          button.dataset.direction = direction;
          button.setAttribute("aria-label", jackDescription(node, port, connections));
          button.title = `${port.label ?? port.id} · ${readableType(port.value_type)}`;
          button.addEventListener("pointerdown", event => beginPointerGesture(event, node, port));
          button.addEventListener("dragover",event=>{if(event.dataTransfer?.types.includes("application/x-tongues-catalog-id"))event.preventDefault();});
          button.addEventListener("drop",event=>{
            const catalogId=event.dataTransfer?.getData("application/x-tongues-catalog-id");if(!catalogId)return;
            event.preventDefault();event.stopPropagation();
            onDropCatalogOnJack?.({catalog_id:catalogId,node_id:node.id,port_id:port.id,direction,clientX:event.clientX,clientY:event.clientY});
          });
          button.addEventListener("click", event => {
            if (event.detail !== 0) return;
            activateJack(node, port);
          });
          button.addEventListener("focus", () => explainFocusedTarget(node, port));
          const text = document.createElement("span");
          text.className = "patch-jack-label";
          text.textContent = port.label ?? port.id;
          wrap.append(button, text);
          jackLayer.append(wrap);
        });
      }
    }
    updateTargetStates();
  }

  function renderPreview() {
    svg.querySelector(".patch-cable-preview")?.remove();
    if (!gesture) return;
    const fixedDirection = gesture.moving === "to" ? "output" : "input";
    const fixed = anchor(gesture.fixed, fixedDirection);
    if (!fixed) return;
    const moving = previewPoint ?? fixed;
    const from = gesture.moving === "to" ? fixed : moving;
    const to = gesture.moving === "to" ? moving : fixed;
    svg.append(svgElement(document, "path", {d: cablePath(from, to), class: "patch-cable-preview"}));
  }

  function render() {
    if (destroyed) return;
    renderOrganizationLayer();
    renderNodeCards();
    renderConnections();
    renderJacks();
  }

  function selectedEdgeForJack(node, port) {
    const selected = pipeline().edges.find(edge => edge.id === getSelectedEdgeId());
    if (!selected) return null;
    const endpoint = port.direction === "output" ? selected.from : selected.to;
    return endpoint.node_id === node.id && endpoint.port_id === port.id ? selected : null;
  }

  function startGesture(node, port) {
    if (isRunLocked()) {
      onAnnounce("Stop transport before rewiring cables.", true);
      return false;
    }
    const selected = selectedEdgeForJack(node, port);
    if (selected) {
      const moving = port.direction === "output" ? "from" : "to";
      gesture = {
        edge_id: selected.id,
        moving,
        fixed: structuredClone(moving === "from" ? selected.to : selected.from),
        origin: {node_id: node.id, port_id: port.id, direction: port.direction},
      };
    } else {
      const moving = port.direction === "output" ? "to" : "from";
      gesture = {
        edge_id: null,
        moving,
        fixed: {node_id: node.id, port_id: port.id},
        origin: {node_id: node.id, port_id: port.id, direction: port.direction},
      };
    }
    container.dataset.patching = "true";
    updateTargetStates();
    const action = gesture.edge_id ? "Reconnect the selected cable" : "Patch a new cable";
    onAnnounce(`${action}: choose a compatible ${gesture.moving === "to" ? "input" : "output"} jack. Escape cancels.`);
    return true;
  }

  function endpointsFor(node, port) {
    if (!gesture) return null;
    const candidate = {node_id: node.id, port_id: port.id};
    return gesture.moving === "to"
      ? {from: gesture.fixed, to: candidate}
      : {from: candidate, to: gesture.fixed};
  }

  function compatibilityFor(node, port) {
    const endpoints = endpointsFor(node, port);
    if (!endpoints || port.direction !== (gesture.moving === "to" ? "input" : "output")) return null;
    return connectionCompatibility(
      pipeline(),
      endpoints.from.node_id,
      endpoints.from.port_id,
      endpoints.to.node_id,
      endpoints.to.port_id,
      discovery(),
      {ignoreEdgeId: gesture.edge_id},
    );
  }

  function updateTargetStates() {
    for (const button of jackLayer.querySelectorAll("[data-patch-jack]")) {
      button.classList.remove("compatible", "incompatible", "drag-origin");
      if (!gesture) continue;
      if (
        button.dataset.nodeId === gesture.origin.node_id
        && button.dataset.portId === gesture.origin.port_id
        && button.dataset.direction === gesture.origin.direction
      ) {
        button.classList.add("drag-origin");
        continue;
      }
      const node = pipeline().nodes.find(item => item.id === button.dataset.nodeId);
      const port = nodePorts(node, button.dataset.direction).find(item => item.id === button.dataset.portId);
      const result = compatibilityFor(node, port);
      if (!result) continue;
      button.classList.add(result.compatible ? "compatible" : "incompatible");
      button.title = result.reason;
    }
  }

  function explainFocusedTarget(node, port) {
    const result = compatibilityFor(node, port);
    if (result) onAnnounce(result.reason, !result.compatible);
  }

  function focusJack(endpoint) {
    window.requestAnimationFrame(() => {
      const buttons = [...jackLayer.querySelectorAll("[data-patch-jack]")];
      buttons.find(button => (
        button.dataset.nodeId === endpoint.node_id
        && button.dataset.portId === endpoint.port_id
        && button.dataset.direction === endpoint.direction
      ))?.focus();
    });
  }

  function cancel(message = "Cable gesture cancelled; the graph was not changed.") {
    const origin = gesture?.origin;
    gesture = null;
    previewPoint = null;
    pointerId = null;
    delete container.dataset.patching;
    render();
    if (origin) focusJack(origin);
    onAnnounce(message);
  }

  function commitTarget(node, port) {
    const result = compatibilityFor(node, port);
    if (!result?.compatible) {
      onAnnounce(result?.reason ?? "Choose a compatible cable target.", true);
      return false;
    }
    const current = gesture;
    const before = structuredClone(pipeline());
    if (current.edge_id) {
      reconnectEdge(pipeline(), current.edge_id, current.moving, node.id, port.id, discovery());
      onGraphEdit({kind: "edge.reconnect", edge_id: current.edge_id, endpoint: current.moving, before});
      onSelectEdge(current.edge_id);
      onAnnounce("Cable plug reconnected; edge identity and capacity were preserved.");
    } else {
      const endpoints = endpointsFor(node, port);
      const edge = connectPorts(
        pipeline(),
        endpoints.from.node_id,
        endpoints.from.port_id,
        endpoints.to.node_id,
        endpoints.to.port_id,
        discovery(),
      );
      onGraphEdit({kind: "edge.connect", edge_id: edge.id, before});
      onSelectEdge(edge.id);
      onAnnounce("Typed cable connected.");
    }
    gesture = null;
    previewPoint = null;
    pointerId = null;
    delete container.dataset.patching;
    render();
    focusJack({node_id: node.id, port_id: port.id, direction: port.direction});
    return true;
  }

  function activateJack(node, port) {
    if (!gesture) {
      if (!startGesture(node, port)) return;
      renderPreview();
      return;
    }
    commitTarget(node, port);
  }

  function autoPan(clientX, clientY) {
    const bounds = container.getBoundingClientRect();
    let x = 0;
    let y = 0;
    if (clientX < bounds.left + AUTO_PAN_MARGIN) x = 12;
    else if (clientX > bounds.right - AUTO_PAN_MARGIN) x = -12;
    if (clientY < bounds.top + AUTO_PAN_MARGIN) y = 12;
    else if (clientY > bounds.bottom - AUTO_PAN_MARGIN) y = -12;
    if (x || y) cy.panBy({x, y});
  }

  function pointerMove(event) {
    if (!gesture || event.pointerId !== pointerId) return;
    const bounds = container.getBoundingClientRect();
    previewPoint = {x: event.clientX - bounds.left, y: event.clientY - bounds.top};
    autoPan(event.clientX, event.clientY);
    render();
  }

  function pointerUp(event) {
    if (!gesture || event.pointerId !== pointerId) return;
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest?.("[data-patch-jack]");
    if (!target) {
      const current=gesture,origin=current.origin;
      if(!current.edge_id&&onDropEmpty){
        gesture=null;previewPoint=null;pointerId=null;delete container.dataset.patching;render();
        onDropEmpty({
          kind:current.moving==="to"?"from_output":"to_input",anchor:structuredClone(current.fixed),
          clientX:event.clientX,clientY:event.clientY,
        });
        onAnnounce(`Choose a compatible module for ${origin.node_id}.${origin.port_id}.`);
        return;
      }
      return cancel();
    }
    const node = pipeline().nodes.find(item => item.id === target.dataset.nodeId);
    const port = nodePorts(node, target.dataset.direction).find(item => item.id === target.dataset.portId);
    if (!node || !port) {
      const current=gesture,origin=current.origin;
      if(!current.edge_id&&onDropEmpty){
        gesture=null;previewPoint=null;pointerId=null;delete container.dataset.patching;render();
        onDropEmpty({
          kind:current.moving==="to"?"from_output":"to_input",anchor:structuredClone(current.fixed),
          clientX:event.clientX,clientY:event.clientY,
        });
        onAnnounce(`Choose a compatible module for ${origin.node_id}.${origin.port_id}.`);
        return;
      }
      pointerId = null;
      previewPoint = null;
      renderPreview();
      return;
    }
    if (!commitTarget(node, port)) {
      pointerId = null;
      previewPoint = null;
      renderPreview();
    }
  }

  function beginPointerGesture(event, node, port) {
    if (event.button !== 0) return;
    if (isRunLocked()) {
      event.preventDefault();
      onAnnounce("Stop transport before patching cables.");
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    pointerId = event.pointerId;
    startGesture(node, port);
    const bounds = container.getBoundingClientRect();
    previewPoint = {x: event.clientX - bounds.left, y: event.clientY - bounds.top};
    render();
  }

  function keydown(event) {
    if (event.key === "Escape" && gesture) {
      event.preventDefault();
      cancel();
      return;
    }
    if (!["Delete", "Backspace"].includes(event.key) || editableTarget(event.target)) return;
    if (isRunLocked()) {
      event.preventDefault();
      onAnnounce("Stop transport before removing cables.");
      return;
    }
    const edgeId = getSelectedEdgeId();
    if (!edgeId || !pipeline().edges.some(edge => edge.id === edgeId)) return;
    event.preventDefault();
    const before = structuredClone(pipeline());
    removeEdge(pipeline(), edgeId);
    onGraphEdit({kind: "edge.delete", edge_id: edgeId, before});
    onAnnounce("Selected cable deleted.");
    render();
  }

  document.addEventListener("pointermove", pointerMove);
  document.addEventListener("pointerup", pointerUp);
  document.addEventListener("keydown", keydown);
  cy.on("pan zoom resize position", render);

  render();
  return {
    render,
    cancel,
    renderNodeCards,
    hasGesture: () => Boolean(gesture),
    destroy() {
      destroyed = true;
      document.removeEventListener("pointermove", pointerMove);
      document.removeEventListener("pointerup", pointerUp);
      document.removeEventListener("keydown", keydown);
      cy.off("pan zoom resize position", render);
      svg.remove();
      jackLayer.remove();
      cardLayer.remove();
      connectionList.remove();
      organizationLayer.remove();
    },
  };
}
