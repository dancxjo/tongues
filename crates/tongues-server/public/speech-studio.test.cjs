const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const studio = require('./speech-studio.js');
const studioSource = fs.readFileSync(require.resolve('./speech-studio.js'), 'utf8');

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

test('rejects stale snapshots that would undo concurrent verification progress', () => {
    const current = { verification_ids: ['vits-vctk'] };
    assert.equal(studio.preservesVerificationProgress(current, {
        verification_ids: [],
    }), true);
    assert.equal(studio.preservesVerificationProgress(current, {
        verification_ids: ['vits-vctk', 'voice-amy-medium'],
    }), false);
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
