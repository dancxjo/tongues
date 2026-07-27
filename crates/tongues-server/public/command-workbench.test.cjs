const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const app = fs.readFileSync(require.resolve('./app.js'), 'utf8');
const html = fs.readFileSync(require.resolve('./index.html'), 'utf8');
const studio = fs.readFileSync(require.resolve('./speech-studio.js'), 'utf8');
const graph = fs.readFileSync(require.resolve('./speech-dataflow.js'), 'utf8');
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
    for (const mode of ['stream', 'jsonl', 'json', 'raw']) {
        assert.match(html, new RegExp(`<option value="${mode}">`));
    }
    assert.match(app, /navigator\.clipboard\.writeText\(byId\('command-preview'\)\.value\)/);
});

test('Command and graph navigation use server schema metadata and stable links', () => {
    assert.match(app, /page\.capability_href/);
    assert.match(app, /page\.model_href/);
    assert.match(app, /page\.studio_template/);
    assert.match(graph, /new URLSearchParams\(location\.search\)\.get\("starter"\)/);
    assert.match(graph, /id="node-docs"|node-docs/);
});
