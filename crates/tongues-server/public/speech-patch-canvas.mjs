import {
  connectionCompatibility,
  connectPorts,
  portsFor,
  reconnectEdge,
  removeEdge,
} from "./speech-dataflow-model.mjs";

const NODE_WIDTH = 228;
const NODE_HEIGHT = 126;
const PORT_SPACING = 22;
const AUTO_PAN_MARGIN = 44;

export function signalFamily(valueType) {
  if (String(valueType).startsWith("audio")) return "audio";
  if (String(valueType).startsWith("transcript") || valueType === "text") return "text";
  if (["utterance_plan", "control", "cancellation"].includes(valueType)) return "control";
  if (valueType === "error") return "error";
  if (String(valueType).includes("artifact")) return "artifact";
  return "data";
}

export function cablePath(from, to) {
  const distance = Math.max(56, Math.abs(to.x - from.x) * 0.45);
  return `M ${from.x} ${from.y} C ${from.x + distance} ${from.y}, ${to.x - distance} ${to.y}, ${to.x} ${to.y}`;
}

export function portAnchor(position, ports, index, direction) {
  const count = Math.max(1, ports.length);
  const span = Math.min(NODE_HEIGHT - 34, (count - 1) * PORT_SPACING);
  const y = position.y - span / 2 + (count === 1 ? 0 : (span * index) / (count - 1));
  return {
    x: position.x + (direction === "output" ? NODE_WIDTH / 2 : -NODE_WIDTH / 2),
    y,
  };
}

function readableType(value) {
  return String(value ?? "unknown").replaceAll("_", " ");
}

function editableTarget(target) {
  return Boolean(target?.closest?.("input, textarea, select, [contenteditable=true]"));
}

function injectStyles(document) {
  if (document.querySelector("style[data-speech-patch-canvas]")) return;
  const style = document.createElement("style");
  style.dataset.speechPatchCanvas = "";
  style.textContent = `
    .patch-cables,.patch-jacks{position:absolute;inset:0;z-index:2;pointer-events:none}
    .patch-cables{width:100%;height:100%;overflow:visible}
    .patch-cable{fill:none;stroke:#8da4ba;stroke-width:4;pointer-events:stroke;cursor:pointer}
    .patch-cable-hit{fill:none;stroke:transparent;stroke-width:18;pointer-events:stroke;cursor:pointer}
    .patch-cable.signal-audio{stroke:#70d6a4;stroke-width:5}
    .patch-cable.signal-text{stroke:#dca3ff;stroke-dasharray:10 5}
    .patch-cable.signal-control{stroke:#ffca70;stroke-dasharray:3 5}
    .patch-cable.signal-error{stroke:#ff8c91;stroke-dasharray:2 4}
    .patch-cable.signal-artifact{stroke:#72b7ff;stroke-dasharray:14 4 3 4}
    .patch-cable.signal-data{stroke:#a6b7ca}
    .patch-cable.selected{stroke:#f7fffd;stroke-width:7;filter:drop-shadow(0 0 5px #76e2ce)}
    .patch-cable.invalid{stroke:#ffc86b}
    .patch-cable-preview{fill:none;stroke:#f7fffd;stroke-width:4;stroke-dasharray:8 5;pointer-events:none}
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
    getSelectedEdgeId,
    onSelectNode,
    onSelectEdge,
    onGraphEdit,
    onAnnounce,
  } = options;
  const document = container.ownerDocument;
  const window = document.defaultView;
  injectStyles(document);

  const svg = svgElement(document, "svg", {
    class: "patch-cables",
    "aria-hidden": "true",
  });
  const jackLayer = document.createElement("div");
  jackLayer.className = "patch-jacks";
  const connectionList = document.createElement("ol");
  connectionList.className = "patch-connection-list";
  connectionList.setAttribute("aria-label", "Graph connections");
  container.parentElement.append(svg, jackLayer, connectionList);

  let gesture = null;
  let previewPoint = null;
  let pointerId = null;
  let destroyed = false;

  const pipeline = () => getPipeline();
  const discovery = () => getDiscovery();
  const catalog = () => getCatalog();
  const label = nodeId => nodeLabel(pipeline().nodes.find(node => node.id === nodeId), catalog());

  function nodePorts(node, direction) {
    return portsFor(node, direction, discovery());
  }

  function anchor(endpoint, direction) {
    const node = pipeline().nodes.find(item => item.id === endpoint.node_id);
    const element = cy.getElementById(endpoint.node_id);
    if (!node || !element?.length) return null;
    const ports = nodePorts(node, direction);
    const index = ports.findIndex(port => port.id === endpoint.port_id);
    if (index < 0) return null;
    return portAnchor(element.renderedPosition(), ports, index, direction);
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
    for (const edge of pipeline().edges) {
      const from = anchor(edge.from, "output");
      const to = anchor(edge.to, "input");
      if (!from || !to) continue;
      const source = pipeline().nodes.find(node => node.id === edge.from.node_id);
      const output = nodePorts(source, "output").find(port => port.id === edge.from.port_id);
      const path = cablePath(from, to);
      const hit = svgElement(document, "path", {d: path, class: "patch-cable-hit", "data-edge-id": edge.id});
      const visible = svgElement(document, "path", {
        d: path,
        "data-edge-id": edge.id,
        class: `patch-cable signal-${signalFamily(output?.value_type)}${getSelectedEdgeId() === edge.id ? " selected" : ""}${grouped[edge.id]?.length ? " invalid" : ""}`,
      });
      hit.addEventListener("pointerdown", event => {
        event.stopPropagation();
        onSelectEdge(edge.id);
        render();
      });
      svg.append(hit, visible);

      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.edgeId = edge.id;
      button.textContent = edgeDescription(edge);
      button.onclick = () => onSelectEdge(edge.id);
      button.onkeydown = event => {
        if (!["Delete", "Backspace"].includes(event.key)) return;
        event.preventDefault();
        removeEdge(pipeline(), edge.id);
        onGraphEdit({kind: "edge.delete", edge_id: edge.id});
        onAnnounce("Connection deleted.");
        render();
      };
      item.append(button);
      connectionList.append(item);
    }
    renderPreview();
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
    for (const node of pipeline().nodes) {
      const cyNode = cy.getElementById(node.id);
      if (!cyNode?.length) continue;
      for (const direction of ["input", "output"]) {
        const ports = nodePorts(node, direction);
        ports.forEach((port, index) => {
          const position = portAnchor(cyNode.renderedPosition(), ports, index, direction);
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
    if (current.edge_id) {
      reconnectEdge(pipeline(), current.edge_id, current.moving, node.id, port.id, discovery());
      onGraphEdit({kind: "edge.reconnect", edge_id: current.edge_id, endpoint: current.moving});
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
      onGraphEdit({kind: "edge.connect", edge_id: edge.id});
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
      startGesture(node, port);
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
    if (!target) return cancel();
    const node = pipeline().nodes.find(item => item.id === target.dataset.nodeId);
    const port = nodePorts(node, target.dataset.direction).find(item => item.id === target.dataset.portId);
    if (!node || !port || !commitTarget(node, port)) {
      pointerId = null;
      previewPoint = null;
      renderPreview();
    }
  }

  function beginPointerGesture(event, node, port) {
    if (event.button !== 0) return;
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
    const edgeId = getSelectedEdgeId();
    if (!edgeId || !pipeline().edges.some(edge => edge.id === edgeId)) return;
    event.preventDefault();
    removeEdge(pipeline(), edgeId);
    onGraphEdit({kind: "edge.delete", edge_id: edgeId});
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
    hasGesture: () => Boolean(gesture),
    destroy() {
      destroyed = true;
      document.removeEventListener("pointermove", pointerMove);
      document.removeEventListener("pointerup", pointerUp);
      document.removeEventListener("keydown", keydown);
      cy.off("pan zoom resize position", render);
      svg.remove();
      jackLayer.remove();
      connectionList.remove();
    },
  };
}
