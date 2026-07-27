import {test, expect} from "@playwright/test";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import {fileURLToPath} from "node:url";

const publicRoot = path.dirname(fileURLToPath(import.meta.url));
let server;
let baseUrl;

const contentType = file => file.endsWith(".html") ? "text/html"
  : file.endsWith(".css") ? "text/css"
  : file.endsWith(".js") || file.endsWith(".mjs") ? "text/javascript"
  : "text/plain";

test.beforeAll(async () => {
  server = http.createServer((request, response) => {
    const pathname = new URL(request.url, "http://fixture").pathname;
    const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
    const target = path.resolve(publicRoot, relative);
    if (!target.startsWith(`${publicRoot}${path.sep}`) || !fs.existsSync(target)) {
      response.writeHead(404);
      response.end("not found");
      return;
    }
    response.writeHead(200, {"Content-Type": contentType(target)});
    if (target.endsWith(".js") || target.endsWith(".mjs")) {
      response.end("/* Visual fixture: behavior is covered by focused browser/model tests. */");
      return;
    }
    fs.createReadStream(target).pipe(response);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

test.afterAll(async () => new Promise(resolve => server.close(resolve)));

async function openFixture(page, workspace) {
  const routes = {
    workbench: "/",
    graph: "/speech-dataflow.html",
    tracks: "/run-tracks.html",
    wavedeck: "/wavedeck.html",
  };
  await page.goto(`${baseUrl}${routes[workspace]}`);
  await page.evaluate(name => {
    const html = String.raw;
    if (name === "workbench") {
      document.querySelector("#primary-nav").innerHTML = html`
        <div class="nav-group"><div class="nav-heading">Workspaces</div>
          <a class="active">Speech Studio</a><a>Graph Studio</a><a>Execution Tracks</a><a>WaveDeck</a><a>Command Workbench</a>
        </div>`;
      document.querySelector("#page-kicker").textContent = "Speech Studio";
      document.querySelector("#page-title").textContent = "Compose";
      document.querySelector("#page-summary").textContent = "Configure a speech workflow while retaining explicit provider and runtime evidence.";
      document.querySelector("#dashboard-grid").innerHTML = html`
        <a class="command-card"><span>Speech</span><strong>Speak</strong><small>Ready</small><p>Generate speech with a verified model recipe.</p></a>
        <a class="command-card"><span>Runtime</span><strong>Live conversation with an intentionally long workspace label</strong><small>Loading provider inventory</small><p>Stream recognition and synthesis without hiding intermediate state.</p></a>
        <a class="command-card"><span>Evidence</span><strong>Execution Tracks</strong><small>Completed</small><p>Inspect aligned provenance and durable run output.</p></a>`;
    }
    if (name === "graph") {
      document.querySelector("#validation").textContent = "Ready to compile and execute";
      document.querySelector("#validation").dataset.state = "valid";
      document.querySelector("#palette").innerHTML = html`
        <div class="palette-list"><button class="palette-node"><strong>Microphone</strong><small>Audio source · ready</small></button>
        <button class="palette-node"><strong>Streaming recognizer with a deliberately long provider name</strong><small>Transcript · ready</small></button></div>`;
      document.querySelector("#canvas").innerHTML = html`
        <div style="position:absolute;left:8%;top:25%;width:12rem;padding:.75rem;border:1px solid #70d6a4;border-radius:.5rem;background:#162636;color:#edf5ff"><strong>Microphone</strong><p>audio stream</p></div>
        <div style="position:absolute;left:38%;top:44%;width:15rem;padding:.75rem;border:1px solid #dca3ff;border-radius:.5rem;background:#162636;color:#edf5ff"><strong>Fixture ASR</strong><p>English · timestamps on</p></div>
        <div style="position:absolute;left:72%;top:30%;width:12rem;padding:.75rem;border:1px solid #72b7ff;border-radius:.5rem;background:#162636;color:#edf5ff"><strong>Transcript sink</strong><p>committed text</p></div>`;
      document.querySelector("#node-inspector").hidden = false;
      document.querySelector("#empty-inspector").hidden = true;
      document.querySelector("#node-title").textContent = "Fixture ASR";
      document.querySelector("#node-detail").textContent = "Selected node · provider ready";
    }
    if (name === "tracks") {
      document.querySelector("#run-index").hidden = true;
      document.querySelector("#run-view").hidden = false;
      document.querySelector("#run-name").textContent = "run:visual-fixture";
      document.querySelector("#status-badge").textContent = "completed";
      document.querySelector("#status-badge").dataset.state = "completed";
      document.querySelector("#privacy").innerHTML = "<strong>Capture: file transcription</strong><span>Raw audio retained: no</span><span>Biometric speaker data retained: no</span>";
      const rows = [
        ["Audio input", "Observed waveform", "committed"],
        ["Raw transcript", "A representative transcript span with long content", "provisional"],
        ["Normalized", "A representative transcript.", "revised"],
        ["Pipeline", "Completed", "committed"],
      ];
      document.querySelector("#tracks").innerHTML = rows.map(([label, value, state], index) => html`
        <div class="track"><div class="track-label"><strong>${label}</strong><br><small>1 span</small></div>
        <div class="lane"><button class="span" data-status="${state}" style="left:${8 + index * 4}%;width:${34 + index * 6}%">${value}</button></div></div>`).join("");
      document.querySelector("#selection-empty").hidden = true;
      document.querySelector("#selection-details").hidden = false;
      document.querySelector("#selection-details").innerHTML = "<dt>Authority</dt><dd>Observed evidence</dd><dt>Provider / model</dt><dd>Fixture / tiny</dd><dt>State</dt><dd>committed</dd>";
    }
    if (name === "wavedeck") {
      document.querySelector("#session-name").textContent = "session:visual-fixture";
      document.querySelector("#operation-count").textContent = "2 replayable operations";
      document.querySelector("#status").textContent = "Timeline ready. Edited interpretation is separate from observed evidence.";
      document.querySelectorAll("[data-needs-session]").forEach(node => node.disabled = false);
      document.querySelector("#original").innerHTML = html`
        <button class="span"><strong>Hello world</strong><small>20–540 ms · observed word evidence</small></button>
        <button class="span"><strong>from the recognition provider</strong><small>560–1400 ms · observed transcript</small></button>`;
      document.querySelector("#edited").innerHTML = html`
        <button class="span selected"><strong>Hello, world</strong><small>20–560 ms · corrected interpretation</small></button>
        <button class="span"><strong>from the recognition provider</strong><small>580–1400 ms · unchanged interpretation</small></button>`;
    }
  }, workspace);
}

for (const workspace of ["workbench", "graph", "tracks", "wavedeck"]) {
  test(`${workspace} desktop and narrow visual contract`, async ({page}) => {
    await page.emulateMedia({colorScheme: "light", reducedMotion: "reduce"});
    await page.setViewportSize({width: 1280, height: 820});
    await openFixture(page, workspace);
    await expect(page.locator("body")).toHaveScreenshot(`${workspace}-desktop-light.png`, {animations: "disabled"});

    await page.emulateMedia({colorScheme: "dark", reducedMotion: "reduce"});
    await page.setViewportSize({width: 390, height: 844});
    await expect(page.locator("body")).toHaveScreenshot(`${workspace}-narrow-dark.png`, {animations: "disabled"});
  });
}

test("shared shell exposes current location and visible keyboard focus", async ({page}) => {
  await openFixture(page, "wavedeck");
  await expect(page.locator('.tongues-nav a[aria-current="page"]')).toHaveText("WaveDeck");
  const start = page.getByRole("button", {name: "Start live microphone"});
  await start.focus();
  await expect(start).toBeFocused();
  await expect.poll(() => start.evaluate(element => getComputedStyle(element).boxShadow)).not.toBe("none");
});
