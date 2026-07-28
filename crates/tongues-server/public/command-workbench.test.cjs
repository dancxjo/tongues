const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const app = fs.readFileSync(require.resolve('./app.js'), 'utf8');
const html = fs.readFileSync(require.resolve('./index.html'), 'utf8');
const studio = fs.readFileSync(require.resolve('./speech-studio.js'), 'utf8');
const graph = fs.readFileSync(require.resolve('./speech-dataflow.js'), 'utf8');
const tracks = fs.readFileSync(require.resolve('./run-tracks.js'), 'utf8');
const wavedeck = fs.readFileSync(require.resolve('./wavedeck.js'), 'utf8');
const navigationDoc = fs.readFileSync(require.resolve('../../../docs/speech-workspace-navigation.md'), 'utf8');
const server = fs.readFileSync(require.resolve('../src/main.rs'), 'utf8');

test('Command Workbench is a distinct deep-linkable schema-owned page', () => {
    assert.match(server, /\.route\("\/commands\/\{\*path\}", get\(serve_app_index\)\)/);
    assert.match(app, /fetch\('\/api\/cli\/schema'\)/);
    assert.match(app, /path === '\/commands' \|\| path\.startsWith\('\/commands\/'\)/);
    assert.match(html, /id="command-workbench-page"/);
    assert.match(html, /id="command-search"/);
    assert.match(html, /data-command-level="workflow"/);
    assert.doesNotMatch(studio, /\/api\/cli\/schema|command-workbench-page|command-results/);
});

test('Workbench runs, cancels, restores safe form state, and inspects structured streams', () => {
    assert.match(app, /history\.replaceState\(\{\}, '', `\$\{url\.pathname\}\$\{url\.search\}`\)/);
    assert.match(app, /tongues\.command-workbench\.recent\.v1/);
    assert.match(app, /new EventSource\(`\/api\/jobs\/\$\{encodeURIComponent\(jobId\)\}\/events`\)/);
    assert.match(app, /\/api\/jobs\/\$\{encodeURIComponent\(activeJobId\)\}\/cancel/);
    assert.match(app, /invocation_path: `\$\{window\.location\.pathname\}\$\{window\.location\.search\}`/);
    assert.match(app, /searchParams\.set\('job', data\.job_id\)/);
    assert.match(app, /new URLSearchParams\(window\.location\.search\)\.get\('job'\)/);
    assert.match(html, /id="workbench-job-link"/);
    assert.match(app, /form\.reportValidity\(\)/);
    assert.match(app, /activePage\.risk === 'destructive'.*window\.confirm/);
    assert.match(html, /id="command-risk"/);
    assert.match(app, /artifact\.download_url/);
    for (const mode of ['stream', 'jsonl', 'json', 'raw']) {
        assert.match(html, new RegExp(`<option value="${mode}">`));
    }
    assert.match(app, /navigator\.clipboard\.writeText\(byId\('command-preview'\)\.value\)/);
});

test('Command and graph navigation use server schema metadata and stable links', () => {
    assert.match(app, /page\.capability_href/);
    assert.match(app, /page\.model_href/);
    assert.match(app, /page\.studio_template/);
    assert.match(graph, /params\.get\("starter"\)/);
    assert.match(graph, /id="node-docs"|node-docs/);
});

test('speech workspaces expose durable multi-page routes and truthful identities', () => {
    for (const route of [
        '/studio/graphs/new',
        '/studio/graphs/{graph_id}',
        '/runs/{run_id}/tracks',
        '/sessions/{session_id}/correct',
    ]) {
        assert.match(server, new RegExp(route.replace(/[{}]/g, '\\$&')));
    }
    assert.match(app, /Editing a live conversation|Executing a live conversation/);
    assert.match(graph, /Editing a configuration draft/);
    assert.match(wavedeck, /original evidence remains unchanged|Original evidence remains immutable/i);
    assert.match(navigationDoc, /independently loadable browser workspaces/);
});

test('fresh loads restore durable graph, run, node, and session context with recovery', () => {
    assert.match(graph, /\/api\/pipeline\/graphs\/\$\{encodeURIComponent\(routeGraphId\)\}/);
    assert.match(graph, /params\.get\("node"\)/);
    assert.match(graph, /Start a new graph.*open recent runs/);
    assert.match(tracks, /\/api\/pipeline\/runs\/\$\{encodeURIComponent\(runId\)\}/);
    assert.match(tracks, /graphRoute\(state\.projected\.graph_id, provenance\.graph_node_id\)/);
    assert.match(tracks, /Run context could not be restored/);
    assert.match(wavedeck, /\/api\/timeline\/sessions\/\$\{encodeURIComponent\(sessionId\)\}/);
    assert.match(wavedeck, /Session context could not be restored/);
});

test('route transitions announce identity and move focus without hijacking modified clicks', () => {
    assert.match(app, /byId\('route-status'\)\.textContent/);
    assert.match(app, /byId\('page-title'\)\.focus\(\)/);
    assert.match(app, /window\.addEventListener\('popstate'/);
    for (const modifier of ['metaKey', 'ctrlKey', 'shiftKey', 'altKey']) {
        assert.match(app, new RegExp(`event\\.${modifier}`));
    }
    assert.match(tracks, /selection-heading.*focus/);
    assert.match(wavedeck, /page-title.*focus/);
});

test('the landing page names every v1 workflow and starts with the supported journey', () => {
    for (const [title, path] of [
        ['Speak', '/speech'],
        ['Compose', '/speech/compose'],
        ['Compare', '/speech/compare'],
        ['Catalog', '/speech/catalog'],
        ['Operate', '/speech/operate'],
        ['Advanced / Commands', '/commands'],
        ['Tracks / WaveDeck', '/runs'],
        ['Live', '/speech/live'],
    ]) {
        assert.match(app, new RegExp(`title: '${title.replace('/', '\\/')}'[\\s\\S]{0,80}path: '${path.replace('/', '\\/')}'`));
    }
    assert.match(app, /Start with Speak/);
    assert.match(app, /Compose or Compare → Operate → Tracks \/ WaveDeck/);
    assert.match(app, /V1_WORKFLOWS\.map\(workflowLink\)/);
    assert.match(app, /workflow\.clientRoute \? ` data-route/);
});
