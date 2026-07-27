import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {cablePath, portAnchor, signalFamily} from "./speech-patch-canvas.mjs";

test("signal styling uses type families and never depends on color alone", () => {
  assert.equal(signalFamily("audio_stream"), "audio");
  assert.equal(signalFamily("transcript_committed"), "text");
  assert.equal(signalFamily("cancellation"), "control");
  assert.equal(signalFamily("error"), "error");
  assert.equal(signalFamily("wave_artifact"), "artifact");
});

test("input and output jack anchors remain stable around a module face", () => {
  const ports = [{id: "a"}, {id: "b"}, {id: "c"}];
  assert.deepEqual(portAnchor({x: 300, y: 200}, ports, 1, "input"), {x: 186, y: 200});
  assert.deepEqual(portAnchor({x: 300, y: 200}, ports, 1, "output"), {x: 414, y: 200});
  assert.ok(portAnchor({x: 300, y: 200}, ports, 0, "input").y < 200);
  assert.ok(portAnchor({x: 300, y: 200}, ports, 2, "input").y > 200);
});

test("customized geometry shifts jack anchors consistently for both directions", () => {
  const ports = [{id: "a"}, {id: "b"}, {id: "c"}];
  const collapsed = portAnchor({x: 300, y: 200}, ports, 1, "input", {width: 320, height: 200, collapsed_height: 78}, true);
  const expanded = portAnchor({x: 300, y: 200}, ports, 1, "output", {width: 320, height: 220, collapsed_height: 78}, false);
  assert.deepEqual(expanded, {x: 460, y: 200});
  assert.deepEqual(collapsed, {x: 140, y: 200});
});

test("cable geometry bends out of output and into input", () => {
  assert.equal(cablePath({x: 10, y: 20}, {x: 210, y: 90}), "M 10 20 C 100 20, 120 90, 210 90");
  assert.match(cablePath({x: 210, y: 20}, {x: 10, y: 90}), /^M 210 20 C 300 20, -80 90, 10 90$/);
});

test("patch controller exposes pointer, touch, keyboard, focus, and direct cable operations", () => {
  const source = fs.readFileSync(new URL("./speech-patch-canvas.mjs", import.meta.url), "utf8");
  assert.match(source, /addEventListener\("pointerdown"/);
  assert.match(source, /document\.addEventListener\("pointermove"/);
  assert.match(source, /event\.key === "Escape"/);
  assert.match(source, /event\.detail !== 0/);
  assert.match(source, /reconnectEdge\(/);
  assert.match(source, /removeEdge\(/);
  assert.match(source, /focusJack\(/);
  assert.match(source, /cy\.panBy/);
  assert.match(source, /aria-label", "Graph connections"/);
  assert.match(source, /stroke-dasharray/);
});
