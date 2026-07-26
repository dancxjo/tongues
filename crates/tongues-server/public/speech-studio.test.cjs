const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const studio = require('./speech-studio.js');
const studioSource = fs.readFileSync(require.resolve('./speech-studio.js'), 'utf8');
const stylesSource = fs.readFileSync(require.resolve('./styles.css'), 'utf8');

function fixturePath(overrides = {}) {
    return {
        backend: 'fastpitch',
        model: 'fastpitch-ljspeech+hifigan-v2',
        display_name: 'FastPitch → HiFi-GAN',
        complete: true,
        runnable: true,
        selected: true,
        varieties: {
            support: 'listed',
            values: [
                { id: 'en-US-GA', label: 'General American English' },
                { id: 'en-GB-RP', label: 'Received Pronunciation' },
            ],
        },
        speakers: { required: false, values: { support: 'unsupported' } },
        controls: [
            {
                field: 'speed', label: 'Speed', kind: 'number', group: 'advanced', default: 1,
            },
            {
                field: 'pitch', label: 'Pitch', kind: 'number_array', group: 'expert',
            },
            {
                field: 'timings', label: 'Timing', kind: 'boolean', group: 'advanced', default: false,
            },
        ],
        ...overrides,
    };
}

test('selects a complete runnable path and keeps unavailable inventory visible', () => {
    const unavailable = fixturePath({
        backend: 'vits',
        model: 'vits-vctk',
        selected: false,
        runnable: false,
        unavailable_reason: 'speaker artifact missing',
    });
    const discovery = { paths: [unavailable, fixturePath()] };
    assert.equal(studio.selectInitialPath(discovery).backend, 'fastpitch');
    assert.equal(studio.availablePaths(discovery).length, 2);
});

test('deduplicates pending catalog verification calls', () => {
    assert.deepEqual(studio.pendingVerificationIds({
        verification_ids: [
            'fastpitch-ljspeech',
            'hifigan-v2-ljspeech',
            'fastpitch-ljspeech',
        ],
    }), ['fastpitch-ljspeech', 'hifigan-v2-ljspeech']);
    assert.deepEqual(studio.pendingVerificationIds({}), []);
});

test('background verification is serialized and uses explicit mutation requests', () => {
    assert.match(studioSource, /const VERIFICATION_CONCURRENCY = 1/);
    assert.match(
        studioSource,
        /models\/verify\/\$\{encodeURIComponent\(modelId\)\}`,\s*\{ method: 'POST'/,
    );
    assert.doesNotMatch(studioSource, />Verifying</);
});

test('rejects stale snapshots that would undo concurrent verification progress', () => {
    const current = { verification_ids: ['vits-vctk'] };
    assert.equal(studio.preservesVerificationProgress(current, {
        verification_ids: [],
    }), true);
    assert.equal(studio.preservesVerificationProgress(current, {
        verification_ids: ['vits-vctk', 'voice-amy-medium'],
    }), false);
});

test('merges cursor pages without duplicating shared inventory', () => {
    const first = {
        page: {
            cursor: 0, limit: 2, returned: 2, total: 3, next_cursor: 2,
        },
        paths: [fixturePath()],
        components: [{ id: 'text' }, { id: 'fastpitch-ljspeech' }],
        compositions: [{ id: 'pipeline/fastpitch' }],
        compatibility: [{
            from_component_id: 'projector/fastpitch',
            to_component_id: 'fastpitch-ljspeech',
            compatible: true,
        }],
        presets: [{ id: 'preset/fastpitch' }],
        verification_ids: ['fastpitch-ljspeech'],
    };
    const next = {
        page: {
            cursor: 2, limit: 2, returned: 1, total: 3,
        },
        paths: [fixturePath({ backend: 'fairseq', model: 'fairseq-mms-vits-eng' })],
        components: [{ id: 'text' }, { id: 'fairseq-mms-vits-eng' }],
        compositions: [{ id: 'pipeline/fairseq-eng' }],
        compatibility: [{
            from_component_id: 'projector/fairseq-mms-vits-eng',
            to_component_id: 'fairseq-mms-vits-eng',
            compatible: true,
        }],
        presets: [{ id: 'preset/fairseq-mms-vits-eng' }],
        verification_ids: ['fairseq-mms-vits-eng'],
    };
    const merged = studio.mergeDiscovery(first, next);
    assert.equal(merged.paths.length, 2);
    assert.equal(merged.components.length, 3);
    assert.equal(merged.compositions.length, 2);
    assert.equal(merged.compatibility.length, 2);
    assert.equal(merged.presets.length, 2);
    assert.deepEqual(
        merged.verification_ids,
        ['fastpitch-ljspeech', 'fairseq-mms-vits-eng'],
    );
    assert.equal(merged.page.cursor, 2);
});

test('runtime polling waits for each request instead of using an overlapping interval', () => {
    assert.doesNotMatch(studioSource, /setInterval\s*\(/);
    assert.match(
        studioSource,
        /await loadRuntime\(controller\.signal\)[\s\S]*window\.setTimeout\(poll, 750\)/,
    );
});

test('filters varieties from the selected path capability', () => {
    assert.deepEqual(
        studio.varietiesForPath(fixturePath()).map((item) => item.id),
        ['en-US-GA', 'en-GB-RP'],
    );
    assert.deepEqual(
        studio.varietiesForPath(fixturePath({ varieties: { support: 'unsupported' } })),
        [],
    );
});

test('selects runnable component compositions independently of legacy paths', () => {
    const pipeline = {
        input: 'text',
        projector: 'projector/fastpitch-ljspeech',
        acoustic_model: 'fastpitch-ljspeech',
        conditioners: [],
        vocoder: 'hifigan-v2-ljspeech',
        output: 'wav',
    };
    const unavailable = {
        id: 'unavailable',
        backend: 'vits',
        runnable: false,
        selected: true,
        pipeline: { input: 'text', projector: 'projector/vits', end_to_end: 'vits', output: 'wav' },
    };
    const ready = {
        id: 'ready',
        backend: 'fastpitch',
        runnable: true,
        selected: false,
        pipeline,
    };
    assert.equal(studio.selectInitialComposition({ compositions: [unavailable, ready] }).id, 'ready');
    assert.equal(studio.compositionGenerator(ready), 'fastpitch-ljspeech');
});

test('treats every MMS checkpoint as an end-to-end model with an integrated decoder', () => {
    const mms = {
        id: 'mms-tha',
        backend: 'fairseq',
        model: 'fairseq-mms-vits-tha',
        runnable: true,
        pipeline: {
            input: 'text',
            projector: 'projector/fairseq-mms-vits-tha',
            conditioners: [],
            end_to_end: 'fairseq-mms-vits-tha',
            output: 'wav',
        },
    };
    assert.equal(studio.compositionGenerator(mms), 'fairseq-mms-vits-tha');
    assert.equal(mms.pipeline.acoustic_model, undefined);
    assert.equal(mms.pipeline.vocoder, undefined);
    assert.match(studioSource, /Waveform decoding is integrated into this model/);
});

test('renders catalog language, script, preprocessing, license, and readiness metadata', () => {
    for (const label of ['Language', 'Script', 'Preprocessing', 'License', 'Readiness']) {
        assert.match(studioSource, new RegExp(`<dt>${label}</dt>`));
    }
});

test('catalog defaults to runnable pipelines and keeps installable and component views explicit', () => {
    assert.match(studioSource, /catalogView: 'ready'/);
    for (const view of ['ready', 'downloadable', 'components']) {
        assert.match(studioSource, new RegExp(`data-catalog-view="${view}"`));
    }
    assert.match(studioSource, /catalog-family/);
    assert.match(studioSource, /catalog-license/);
    assert.match(studioSource, /refreshCatalog/);
    assert.match(studioSource, /Load more models/);
});

test('uses calm metadata wording and clears stale duplex results during synthesis', () => {
    assert.match(studioSource, /No additional preprocessing required/);
    assert.doesNotMatch(studioSource, /None declared/);
    assert.doesNotMatch(studioSource, /License not asserted/);
    assert.match(studioSource, /hideDuplexResult\(\)/);
});

test('reports exact directed compatibility edges', () => {
    const discovery = {
        compatibility: [{
            from_component_id: 'fastpitch-ljspeech',
            to_component_id: 'hifigan-v2-ljspeech',
            compatible: true,
            reason: 'exact contract',
        }],
    };
    assert.equal(
        studio.compatibilityFor(
            discovery,
            'fastpitch-ljspeech',
            'hifigan-v2-ljspeech',
        ).reason,
        'exact contract',
    );
    assert.equal(studio.compatibilityFor(discovery, 'hifigan-v2-ljspeech', 'fastpitch-ljspeech'), undefined);
});

test('builds component-addressed payloads without legacy backend selection', () => {
    const pipeline = {
        input: 'text',
        projector: 'projector/fastpitch-ljspeech',
        acoustic_model: 'fastpitch-ljspeech',
        conditioners: [],
        vocoder: 'hifigan-v2-ljspeech',
        output: 'wav',
    };
    const path = fixturePath({ pipeline });
    const payload = studio.buildPayload(path, new Map(), {
        text: 'Composable speech.',
        variety: 'en-US-GA',
    });
    assert.deepEqual(payload.pipeline, pipeline);
    assert.equal(payload.backend, undefined);
    assert.equal(payload.model, undefined);
});

test('builds payloads only from declared controls', () => {
    const values = new Map([
        ['speed', '1.15'],
        ['pitch', '0.1, -0.2'],
        ['timings', true],
        ['noise_scale', '0.9'],
    ]);
    assert.deepEqual(studio.buildPayload(fixturePath(), values, {
        text: 'Capability driven.',
        variety: 'en-US-GA',
    }), {
        text: 'Capability driven.',
        backend: 'fastpitch',
        model: 'fastpitch-ljspeech+hifigan-v2',
        variety: 'en-US-GA',
        speed: 1.15,
        pitch: [0.1, -0.2],
        timings: true,
    });
});

test('builds duplex requests from prompt lines or saved journals', () => {
    assert.deepEqual(
        studio.duplexLines('Who shot John?\n\nKennedy?'),
        ['Who shot John?', 'Kennedy?'],
    );
    assert.deepEqual(studio.buildDuplexRequest({
        text: 'Who shot John?\nKennedy?',
        mockAcoustics: 'who shot\njohn kennedy',
        variety: 'en-US-GA',
    }), {
        chunks: ['Who shot John?', 'Kennedy?'],
        mock_acoustics: ['who shot', 'john kennedy'],
        variety: 'en-US-GA',
    });
    assert.deepEqual(studio.buildDuplexRequest({
        journalPath: ' target/duplex/oracle-chunks.journal.json ',
    }), {
        journal_path: 'target/duplex/oracle-chunks.journal.json',
    });
});

test('requires named speakers and rejects unavailable paths before submission', () => {
    const vits = fixturePath({
        backend: 'vits',
        model: 'vits-vctk',
        speakers: {
            required: true,
            values: {
                support: 'listed',
                values: [{ id: 'p225', label: 'p225', numeric_id: 0 }],
            },
        },
    });
    assert.throws(
        () => studio.buildPayload(vits, new Map(), { text: 'Hello.', variety: 'en-GB-RP' }),
        /requires a speaker/,
    );
    assert.equal(
        studio.buildPayload(vits, new Map(), {
            text: 'Hello.', variety: 'en-GB-RP', speaker: 'p225',
        }).speaker,
        'p225',
    );
    assert.throws(
        () => studio.buildPayload({ ...vits, runnable: false }, new Map(), { text: 'Hello.' }),
        /complete, ready synthesis path/,
    );
});

test('validates expert numeric arrays inline before network submission', () => {
    assert.deepEqual(studio.parseNumberArray('1, 2.5, -3'), [1, 2.5, -3]);
    assert.deepEqual(studio.parseNumberArray('4, 7, 3', true), [4, 7, 3]);
    assert.throws(() => studio.parseNumberArray('4, 0', true), /positive whole numbers/);
    assert.throws(() => studio.parseNumberArray('1, nope'), /finite numbers/);
});

test('renders predicted duplex tokens distinctly and loads the server-projected schema', () => {
    assert.match(studioSource, /\/api\/duplex\/project/);
    assert.match(stylesSource, /\.duplex-token-predicted/);
    assert.match(studioSource, /predicted[\s\S]*playable client audio/i);
});

test('routes every Speech Studio workflow to a stable deep link', () => {
    const routes = {
        '/speech': 'speak',
        '/speech/': 'speak',
        '/speech/compose': 'compose',
        '/speech/compare': 'compare',
        '/speech/catalog': 'catalog',
        '/speech/operate': 'operate',
    };
    for (const [path, workflow] of Object.entries(routes)) {
        assert.equal(studio.workflowForPath(path), workflow);
        assert.match(studio.studioShell(), new RegExp(`data-workflow="${workflow}"`));
    }
});

test('the initial shell has workflow structure but no expanded registry records', () => {
    const shell = studio.studioShell();
    assert.match(shell, /Turn text into speech/);
    assert.match(shell, /Compose a speech pipeline/);
    assert.match(shell, /Compare complete recipes/);
    assert.match(shell, /Capability discovery/);
    assert.match(shell, /Operate Speech Studio/);
    assert.doesNotMatch(shell, /fairseq-mms-vits-(eng|fra|deu)/);
    assert.equal((shell.match(/<option/g) || []).length < 30, true);
});

test('compose exposes checkpoint ownership, contracts, adapters, and exact CLI output', () => {
    for (const phrase of [
        'Ownership and compatibility',
        'Accepted contract',
        'Exact CLI representation',
        'Adapter + vocoder',
        'checkpoint-owned projector',
    ]) {
        assert.match(studioSource, new RegExp(phrase, 'i'));
    }
    const path = fixturePath({
        pipeline: {
            input: 'text',
            projector: 'projector/fastpitch-ljspeech',
            acoustic_model: 'fastpitch-ljspeech',
            conditioners: [],
            vocoder: 'hifigan-v2-ljspeech',
            output: 'wav',
        },
    });
    const command = studio.cliRepresentation(
        path,
        new Map([['speed', '1.2'], ['device', 'cpu']]),
        { text: 'Exact command.', variety: 'en-US-GA' },
    );
    assert.match(command, /^tongues speak --cpu 'Exact command\.'/);
    assert.match(command, /--backend fastpitch/);
    assert.match(command, /--speed 1\.2/);
});

test('compare preserves per-recipe results and permits partial failure', () => {
    assert.match(studioSource, /compareResults: new Map\(\)/);
    assert.match(studioSource, /Promise\.allSettled\(tasks\)/);
    assert.match(studioSource, /lane\.dataset\.state = 'failed'/);
    assert.match(studioSource, /Results remain playable without regeneration/);
    assert.match(studioSource, /Blind listening mode/);
});

test('recipes persist controls and navigation carries current state between workflows', () => {
    assert.match(studioSource, /tongues\.speech\.user-recipes\.v1/);
    assert.match(studioSource, /controls: controlSnapshot\(path\)/);
    assert.match(studioSource, /compositionId: state\.pathKey/);
    assert.match(studioSource, /show-pipeline/);
    assert.match(studioSource, /add-current-to-compare/);
    assert.match(studioSource, /open-pipeline-in-speak/);
});

test('operate keeps targeted verification, jobs, cancellation, and duplex evidence together', () => {
    assert.match(studioSource, /Verify changed models/);
    assert.match(studioSource, /\/api\/jobs/);
    assert.match(studioSource, /data-cancel-job/);
    assert.match(studioSource, /Labs: predictive duplex evidence/);
    assert.match(studioSource, /refreshOperateJobs/);
    assert.doesNotMatch(
        studioSource,
        /async function refreshDiscovery[\s\S]{0,600}verifyDiscovery\(generation, firstPage\);\s*if/,
    );
});

test('workflow controls are labeled, responsive, theme-aware, and reduced-motion safe', () => {
    for (const label of [
        'Text to speak',
        'Voice or language',
        'Recipe',
        'Shared prompt',
        'Search models and languages',
    ]) {
        assert.match(studioSource, new RegExp(`>${label}<`));
    }
    assert.match(stylesSource, /@media \(max-width: 720px\)/);
    assert.match(stylesSource, /@media \(prefers-reduced-motion: reduce\)/);
    assert.match(stylesSource, /@media \(prefers-color-scheme: dark\)/);
    assert.match(stylesSource, /\.compare-results/);
});
