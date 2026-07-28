import {expect, test} from "@playwright/test";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import {fileURLToPath} from "node:url";

const publicRoot = path.dirname(fileURLToPath(import.meta.url));
let server;
let baseUrl;

test.beforeAll(async () => {
  server = http.createServer((request, response) => {
    const pathname = new URL(request.url, "http://fixture").pathname;
    if (pathname === "/fixture") {
      response.writeHead(200, {"Content-Type": "text/html"});
      response.end(`<!doctype html>
        <html lang="en"><head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width,initial-scale=1">
          <link rel="stylesheet" href="/styles.css">
          <script src="/speech-studio.js"></script>
        </head><body><main id="speech-page"></main></body></html>`);
      return;
    }
    const target = path.resolve(publicRoot, pathname.replace(/^\/+/, ""));
    if (!target.startsWith(`${publicRoot}${path.sep}`) || !fs.existsSync(target)) {
      response.writeHead(404);
      response.end("not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type": target.endsWith(".css") ? "text/css" : "text/javascript",
    });
    fs.createReadStream(target).pipe(response);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

test.afterAll(async () => new Promise(resolve => server.close(resolve)));

const projection = {
  schema_version: 2,
  run_id: "ambiguous-run",
  replay_verified: true,
  deep_link: "/speech/operate?duplex_fixture=ambiguous",
  final_state: {committed: [{surface: "right"}]},
  interpretation: {
    schema_version: 1,
    utterance_id: "utterance-1",
    evidence_status: "available",
    cursor: 0,
    limit: 20,
    returned: 1,
    total: 1,
    targets: [{
      target_id: "resolution:word-right",
      target: {
        utterance_id: "utterance-1",
        scope: {scope: "word", id: "word-right"},
        text_range: {start: 0, end: 5},
      },
      kind: "lexical_identity",
      status: "resolved",
      winner: {
        claim_id: "claim-right",
        target: {},
        kind: "lexical_identity",
        value: {type: "lexical_identity", value: {lexeme_id: "right"}},
        authority: "acoustic_evidence",
        provenance: {source: "acoustic_model", method: "aligned-asr"},
        confidence: 0.91,
        calibration: "ambiguity-corpus-v1",
        lifecycle: "stable",
        selected: true,
        conflicts_with_winner: false,
        supports: [],
        conflicts_with: ["claim-write"],
        rationale_code: "acoustic.alignment",
        rationale: "Aligned acoustics favor right.",
        resolution_explanation: "Selected by source priority.",
      },
      alternatives: [{
        claim_id: "claim-write",
        target: {},
        kind: "lexical_identity",
        value: {type: "lexical_identity", value: {lexeme_id: "write"}},
        authority: "grammar_inference",
        provenance: {source: "grammar", method: "context-parse"},
        confidence: 0.44,
        calibration: null,
        lifecycle: "hypothesis",
        selected: false,
        conflicts_with_winner: true,
        supports: ["claim-context"],
        conflicts_with: ["claim-right"],
        rationale_code: "grammar.context",
        rationale: "The verb context supports write.",
        resolution_explanation: "Lower source priority than aligned acoustics.",
      }],
      option_total: 2,
      options_truncated: false,
      linked_claim_ids: ["claim-right", "claim-write", "claim-context"],
      linked_claims_truncated: false,
      acoustic_links: [{
        evidence_id: "audio-1",
        transcript: "right",
        span: {
          frame_start: 10,
          frame_end: 24,
          time_start: 0.2,
          time_end: 0.48,
          confidence: null,
        },
        alignment: "utterance_or_chunk_span",
      }],
      consequences: [{
        hypothesis_id: "right-branch",
        selected: true,
        statuses: ["selected", "committed"],
        output_text: "right",
        score: {
          acoustic_likelihood: 0.91,
          provider_prior: 0.55,
          lexical_evidence: 0,
          grammar_parse_rank: 0,
          prosody_compatibility: 0,
          user_markup: 0,
          direct_observation: 0,
          combined: 0.78,
          available_components: ["acoustic_likelihood", "provider_prior"],
          claim_attribution: {},
        },
        block_reasons: [],
        deliveries: [],
      }],
    }],
    backend_reports: [{
      hypothesis_id: "write-branch",
      status: "partial",
      diagnostic: "UDPipe projection retained unmatched input",
      report: {requested: "auto", selected: "tongues_rules", attempts: []},
      parse_alternatives: [],
    }],
    backend_reports_truncated: false,
    lifecycle: [{
      sequence: 1,
      claim_id: "claim-write",
      from: "hypothesis",
      to: "invalidated",
      reason: "Operator proposed a provenance-preserving correction.",
      superseded_by: "claim-right",
    }],
    lifecycle_truncated: false,
    projection_losses: [],
    warnings: [],
  },
  timeline: [],
};

test("evidence alternatives are keyboard-expandable, explicit, and narrow-safe", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto(`${baseUrl}/fixture`);
  await page.evaluate(fixture => {
    const root = document.querySelector("#speech-page");
    root.innerHTML = window.SpeechStudio.studioShell();
    document.querySelectorAll("[data-workflow]").forEach(section => {
      section.classList.toggle("hidden", section.dataset.workflow !== "operate");
    });
    window.SpeechStudio.renderDuplexProjection(fixture);
  }, projection);

  await page.locator(".operate-labs > summary").click();
  const section = page.getByRole("region", {name: "Evidence chain"});
  await expect(section).toContainText("Showing 1 of 1");
  const target = page.locator(".evidence-target");
  await expect(target).not.toHaveAttribute("open", "");
  await target.locator(":scope > summary").focus();
  await page.keyboard.press("Enter");
  await expect(target).toHaveAttribute("open", "");
  await expect(target).toContainText("Won · claim-right");
  await expect(target).toContainText("Alternative · claim-write");
  await expect(target).toContainText("Conflicts with winner");
  await expect(target).toContainText("0.440 · uncalibrated");
  await expect(target.getByRole("link", {name: "Permanent link to this target"}))
    .toHaveAttribute("href", /duplex_target=resolution%3Aword-right/);
  const diagnostics = page.locator(".evidence-diagnostics");
  await diagnostics.locator(":scope > summary").click();
  await expect(diagnostics).toContainText("write-branch · partial");
  await expect(diagnostics).toContainText("UDPipe projection retained unmatched input");
  await expect(diagnostics).toContainText("claim-write · hypothesis → invalidated");
  await expect(diagnostics).toContainText("provenance-preserving correction");
  await expect.poll(() => page.evaluate(
    () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
  )).toBe(true);
});
