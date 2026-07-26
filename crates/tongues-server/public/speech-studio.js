(function speechStudioModule(root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.SpeechStudio = api;
}(typeof window !== 'undefined' ? window : null, function buildSpeechStudio() {
    'use strict';

    const state = {
        discovery: null,
        pathKey: '',
        presetId: '',
        values: new Map(),
        samples: [],
        emotions: [],
        audioUrl: null,
        runtimeTimer: null,
        runtimePollController: null,
        runtimePollGeneration: 0,
        verificationGeneration: 0,
    };
    const VERIFICATION_CONCURRENCY = 1;

    const pathKey = (path) => `${path.backend}::${path.model}`;
    const compositionGenerator = (composition) => (
        composition?.pipeline?.end_to_end || composition?.pipeline?.acoustic_model || ''
    );
    const selectedComposition = () => (
        state.discovery?.compositions?.find((composition) => composition.id === state.pathKey)
    );
    const selectedPath = () => {
        const composition = selectedComposition();
        if (!composition) {
            return state.discovery?.paths?.find((path) => pathKey(path) === state.pathKey);
        }
        const legacy = state.discovery?.paths?.find((path) => (
            path.backend === composition.backend && path.model === composition.model
        )) || {};
        return {
            ...legacy,
            ...(composition.capabilities || {}),
            ...composition,
            id: composition.model,
            complete: true,
            acoustic_model: composition.pipeline.acoustic_model,
            vocoder: composition.pipeline.vocoder,
            voice_model: composition.pipeline.end_to_end,
        };
    };
    const listedValues = (capability) => (
        capability?.support === 'listed' ? (capability.values || []) : []
    );
    const availablePaths = (discovery, includeTest = false) => (
        (discovery?.paths || []).filter((path) => includeTest || path.backend !== 'mock')
    );
    const selectInitialPath = (discovery) => {
        const paths = availablePaths(discovery);
        return paths.find((path) => path.selected && path.runnable)
            || paths.find((path) => path.runnable)
            || paths.find((path) => path.selected)
            || paths[0]
            || null;
    };
    const availableCompositions = (discovery, includeTest = false) => (
        (discovery?.compositions || []).filter((composition) => (
            includeTest || composition.backend !== 'mock'
        ))
    );
    const selectInitialComposition = (discovery) => {
        const compositions = availableCompositions(discovery);
        return compositions.find((composition) => composition.selected && composition.runnable)
            || compositions.find((composition) => composition.runnable)
            || compositions.find((composition) => composition.selected)
            || compositions[0]
            || null;
    };
    const compatibilityFor = (discovery, from, to) => (
        (discovery?.compatibility || []).find((edge) => (
            edge.from_component_id === from && edge.to_component_id === to
        ))
    );
    const varietiesForPath = (path) => listedValues(path?.varieties);
    const controlsForPath = (path, group) => (
        (path?.controls || []).filter((control) => control.group === group)
    );
    const pendingVerificationIds = (discovery) => (
        [...new Set((discovery?.verification_ids || []).filter(Boolean))]
    );
    const preservesVerificationProgress = (current, updated) => {
        if (!current) return true;
        const currentPending = new Set(pendingVerificationIds(current));
        return pendingVerificationIds(updated).every((id) => currentPending.has(id));
    };

    function parseNumberArray(source, positiveIntegers = false) {
        if (!String(source || '').trim()) return null;
        const parts = String(source).split(',').map((part) => part.trim());
        const values = parts.map(Number);
        const valid = parts.every(Boolean) && (positiveIntegers
            ? values.every((value) => Number.isSafeInteger(value) && value > 0)
            : values.every(Number.isFinite));
        if (!valid) {
            throw new Error(positiveIntegers
                ? 'Use comma-separated positive whole numbers.'
                : 'Use comma-separated finite numbers.');
        }
        return values;
    }

    function buildPayload(path, values, context = {}) {
        if (!path?.complete || !path?.runnable) {
            throw new Error(path?.unavailable_reason || 'Select a complete, ready synthesis path.');
        }
        const payload = {
            text: String(context.text || ''),
            variety: context.variety || varietiesForPath(path)[0]?.id || null,
        };
        if (path.pipeline) payload.pipeline = path.pipeline;
        else {
            payload.backend = path.backend;
            payload.model = path.model;
        }
        if (!payload.text.trim()) throw new Error('Enter text to synthesize.');
        if (path.speakers?.required && !context.speaker) {
            throw new Error('This model requires a speaker.');
        }
        if (context.speaker) payload.speaker = context.speaker;

        const declared = new Set((path.controls || []).map((control) => control.field));
        for (const control of path.controls || []) {
            const value = values.get(control.field);
            if (value == null || value === '') continue;
            if (control.field === 'device') {
                if (value === 'cpu') payload.cpu = true;
                if (String(value).startsWith('cuda:')) {
                    payload.cuda_device = Number(String(value).split(':')[1]);
                }
                continue;
            }
            if (control.field === 'blend_mode') continue;
            if (
                ['style_alpha', 'style_beta'].includes(control.field)
                && values.get('blend_mode') !== 'raw'
            ) continue;
            if (
                ['speaker_reference_strength', 'style_reference_strength'].includes(control.field)
                && values.get('blend_mode') === 'raw'
            ) continue;
            if (control.kind === 'number') {
                const numeric = Number(value);
                if (!Number.isFinite(numeric)) throw new Error(`${control.label} must be a number.`);
                payload[control.field] = numeric;
            } else if (control.kind === 'number_array') {
                const parsed = parseNumberArray(value);
                if (parsed) payload[control.field] = parsed;
            } else if (control.kind === 'positive_integer_array') {
                const parsed = parseNumberArray(value, true);
                if (parsed) payload[control.field] = parsed;
            } else if (control.kind === 'boolean') {
                payload[control.field] = Boolean(value);
            } else {
                payload[control.field] = value;
            }
        }
        if (payload.emotion && context.emotionVector) {
            payload.emotion_vector = context.emotionVector;
        }
        for (const field of Object.keys(payload)) {
            if (
                !['text', 'pipeline', 'backend', 'model', 'variety', 'speaker', 'emotion_vector'].includes(field)
                && !declared.has(field)
                && !['cpu', 'cuda_device'].includes(field)
            ) {
                delete payload[field];
            }
        }
        return payload;
    }

    function studioShell() {
        return `
            <main class="glass-panel speech-studio">
                <div class="page-doc">
                    <p>Assemble a text-to-audio pipeline. Every connection is checked against
                    the component contracts reported by the resident speech runtime.</p>
                </div>
                <section class="speech-runtime-panel" aria-labelledby="speech-runtime-heading">
                    <div class="speech-section-heading">
                        <div>
                            <span id="speech-runtime-state" class="runtime-badge" data-state="loading">Checking</span>
                            <h2 id="speech-runtime-heading">Resident runtime</h2>
                        </div>
                        <button id="reload-speech-runtime" type="button" class="secondary-button">Reload models</button>
                    </div>
                    <dl id="speech-runtime-grid" class="metadata-grid" aria-live="polite"></dl>
                    <div id="speech-runtime-errors" class="inline-error hidden" role="alert"></div>
                </section>

                <form id="synth-form" novalidate>
                    <div id="speech-error" class="inline-error hidden" role="alert" tabindex="-1"></div>
                    <div class="form-group">
                        <label for="text">Prompt</label>
                        <textarea id="text" required>Wow, the magic wand actually worked!</textarea>
                        <small>Text is planned and synthesized through the selected component pipeline.</small>
                    </div>
                    <div class="pipeline-toolbar">
                        <div class="form-group">
                            <label for="speech-preset">Pipeline preset</label>
                            <select id="speech-preset"></select>
                            <small id="synthesis-path-detail">Loading component discovery…</small>
                        </div>
                    </div>
                    <section class="pipeline-workbench" aria-label="Speech synthesis pipeline">
                        <div class="pipeline-stage" data-stage="input">
                            <span class="pipeline-stage-label">Input</span>
                            <strong>Text</strong>
                            <small>Tongues linguistic plan</small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" data-stage="projector">
                            <label class="pipeline-stage-label" for="pipeline-projector">Projector</label>
                            <select id="pipeline-projector"></select>
                            <small id="pipeline-projector-detail"></small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-generator-stack">
                            <div class="pipeline-stage pipeline-conditioning" data-stage="conditioner">
                                <span class="pipeline-stage-label">Conditioning</span>
                                <strong id="pipeline-conditioning-name">Model controls</strong>
                                <small id="pipeline-conditioning-detail">Speaker, language, style, and prosody</small>
                            </div>
                            <span class="pipeline-branch" aria-hidden="true">↓</span>
                            <div class="pipeline-stage" data-stage="generator">
                                <label class="pipeline-stage-label" for="pipeline-generator">Acoustic model</label>
                                <select id="pipeline-generator"></select>
                                <small id="pipeline-generator-detail"></small>
                            </div>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" id="pipeline-vocoder-stage" data-stage="vocoder">
                            <label class="pipeline-stage-label" for="pipeline-vocoder">Vocoder</label>
                            <select id="pipeline-vocoder"></select>
                            <small id="pipeline-vocoder-detail"></small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" data-stage="output">
                            <span class="pipeline-stage-label">Output</span>
                            <strong>WAV audio</strong>
                            <small>Playback, download, and details</small>
                        </div>
                    </section>
                    <div class="controls-grid">
                        <div class="form-group" id="variety-control">
                            <label for="variety">Linguistic variety</label>
                            <select id="variety"></select>
                            <div id="fixed-variety" class="fixed-value hidden"></div>
                            <small id="variety-detail"></small>
                        </div>
                        <div class="form-group hidden" id="speaker-control">
                            <label for="speaker">Speaker</label>
                            <input id="speaker" type="search" list="speaker-options"
                                autocomplete="off" placeholder="Search p### speakers">
                            <datalist id="speaker-options"></datalist>
                            <small id="speaker-detail"></small>
                        </div>
                    </div>

                    <section id="speech-model-card" class="model-card" aria-live="polite"></section>
                    <div id="speech-controls-basic" class="controls-grid"></div>
                    <details class="advanced-section">
                        <summary>Advanced synthesis controls</summary>
                        <div id="speech-controls-advanced" class="controls-grid"></div>
                    </details>
                    <details class="advanced-section">
                        <summary>Expert token controls</summary>
                        <p class="expert-warning">Token arrays are checked for valid numeric values.
                        They are rejected by the server if they do not match the model projection.</p>
                        <div id="speech-controls-expert" class="controls-grid"></div>
                    </details>
                    <details id="speech-developer" class="advanced-section">
                        <summary>Developer and testing</summary>
                        <p>The deterministic mock generates test audio and is not a voice engine.</p>
                        <div id="speech-controls-developer" class="controls-grid"></div>
                        <button id="select-mock-path" type="button" class="secondary-button hidden">
                            Use deterministic test path
                        </button>
                    </details>
                    <div class="action-bar">
                        <button type="submit" id="submit-btn" disabled>
                            <span class="btn-text">Generate speech</span>
                            <div class="spinner"></div>
                        </button>
                    </div>
                    <div id="speech-submit-status" class="sr-status" role="status" aria-live="polite"></div>
                </form>

                <section id="result-container" class="result-panel hidden" aria-labelledby="speech-result-heading">
                    <div class="speech-section-heading">
                        <h2 id="speech-result-heading">Synthesis result</h2>
                        <a id="speech-download" class="secondary-button" download="tongues-speech.wav">Download WAV</a>
                    </div>
                    <audio id="audio-player" controls></audio>
                    <dl id="speech-result-metadata" class="metadata-grid"></dl>
                    <details id="speech-result-diagnostics" class="advanced-section hidden">
                        <summary>Diagnostics</summary>
                        <pre id="speech-diagnostics-output" class="source-preview"></pre>
                    </details>
                </section>

                <section class="duplex-panel" aria-labelledby="duplex-heading">
                    <div class="speech-section-heading">
                        <div>
                            <h2 id="duplex-heading">Predictive duplex timeline</h2>
                            <p class="page-doc">
                                The server projects replayable duplex belief snapshots so predicted
                                text stays visible without ever becoming playable client audio.
                            </p>
                        </div>
                        <div class="duplex-actions">
                            <button id="run-duplex-preview" type="button" class="secondary-button">Preview timeline</button>
                            <button id="replay-duplex-journal" type="button" class="secondary-button">Replay journal</button>
                        </div>
                    </div>
                    <div class="controls-grid">
                        <div class="form-group">
                            <label for="duplex-mock-acoustics">Mock acoustic chunks</label>
                            <textarea id="duplex-mock-acoustics"
                                placeholder="Optional transcript chunks, one line per chunk"></textarea>
                            <small>Leave blank to preview the prompt lines as direct text chunks.</small>
                        </div>
                        <div class="form-group">
                            <label for="duplex-journal-path">Saved journal path</label>
                            <input id="duplex-journal-path" type="text"
                                placeholder="target/duplex/oracle-chunks.journal.json">
                            <small>Replay a saved journal relative to the repository root.</small>
                        </div>
                    </div>
                    <div id="duplex-error" class="inline-error hidden" role="alert"></div>
                    <div id="duplex-summary" class="duplex-summary hidden"></div>
                    <ol id="duplex-timeline" class="duplex-timeline hidden"></ol>
                </section>

                <details class="engine-inventory">
                    <summary>Engines and Models <span id="component-count"></span></summary>
                    <p>All native, catalog, compatibility, and test components are listed even when
                    they cannot independently produce audio.</p>
                    <button id="verify-all-models" type="button" class="secondary-button">Verify all changed models</button>
                    <div id="component-inventory" class="component-inventory"></div>
                </details>
            </main>`;
    }

    const byId = (id) => document.getElementById(id);

    function showError(message, target = byId('speech-error')) {
        target.textContent = message;
        target.classList.remove('hidden');
        if (target.id === 'speech-error') target.focus();
    }

    function clearError(target = byId('speech-error')) {
        target.textContent = '';
        target.classList.add('hidden');
    }

    function setStoredValue(field, value) {
        state.values.set(field, value);
    }

    function controlInput(control) {
        const wrapper = document.createElement('div');
        wrapper.className = 'form-group speech-control';
        wrapper.dataset.control = control.field;
        const label = document.createElement('label');
        label.htmlFor = `speech-control-${control.field}`;
        label.textContent = control.label;
        const id = label.htmlFor;
        let input;

        if (control.kind === 'select') {
            input = document.createElement('select');
            for (const choice of control.options || []) {
                const option = document.createElement('option');
                option.value = choice.value;
                option.textContent = choice.label;
                input.appendChild(option);
            }
        } else if (control.kind === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            wrapper.classList.add('control-checkbox');
        } else if (control.kind === 'reference_audio' || control.kind === 'emotion') {
            input = document.createElement('select');
            const empty = document.createElement('option');
            empty.value = '';
            empty.textContent = control.kind === 'emotion' ? 'None / neutral' : 'Default reference';
            input.appendChild(empty);
            const source = control.kind === 'emotion' ? state.emotions : state.samples;
            for (const item of source) {
                const option = document.createElement('option');
                option.value = item.id || item.name;
                option.textContent = item.label || item.name || item.id;
                input.appendChild(option);
            }
        } else {
            input = document.createElement('input');
            if (control.kind === 'number') {
                input.type = 'number';
                if (control.min != null) input.min = control.min;
                if (control.max != null) input.max = control.max;
                if (control.step != null) input.step = control.step;
            } else {
                input.type = 'text';
                input.placeholder = control.kind === 'positive_integer_array'
                    ? '4, 7, 3'
                    : '0.1, -0.2, 0.0';
            }
        }
        input.id = id;
        input.name = control.field;
        const stored = state.values.get(control.field);
        const initial = stored ?? control.default;
        const optionValues = input.options ? Array.from(input.options, (item) => item.value) : [];
        if (control.kind === 'boolean') {
            input.checked = Boolean(initial);
        } else if (initial != null && optionValues.includes(String(initial))) {
            input.value = String(initial);
        } else if (initial != null && control.kind !== 'select') {
            input.value = String(initial);
        } else if (control.kind === 'select' && input.options.length) {
            input.value = input.options[0].value;
        }
        setStoredValue(control.field, control.kind === 'boolean' ? input.checked : input.value);
        input.addEventListener('input', () => {
            setStoredValue(control.field, control.kind === 'boolean' ? input.checked : input.value);
            syncBlendControls();
        });
        const help = document.createElement('small');
        help.textContent = [control.help, control.unit ? `Unit: ${control.unit}.` : '']
            .filter(Boolean).join(' ');
        wrapper.append(label, input, help);
        if (control.kind === 'reference_audio') {
            const preview = document.createElement('audio');
            preview.controls = true;
            preview.className = 'preview-audio empty';
            input.addEventListener('change', () => {
                const sample = state.samples.find((item) => item.id === input.value);
                if (sample) {
                    preview.src = sample.audio_url;
                    preview.classList.remove('empty');
                } else {
                    preview.removeAttribute('src');
                    preview.classList.add('empty');
                }
            });
            wrapper.appendChild(preview);
        }
        return wrapper;
    }

    function syncBlendControls() {
        const raw = state.values.get('blend_mode') === 'raw';
        for (const field of ['style_alpha', 'style_beta']) {
            document.querySelector(`[data-control="${field}"]`)?.classList.toggle('hidden', !raw);
        }
        for (const field of ['speaker_reference_strength', 'style_reference_strength']) {
            document.querySelector(`[data-control="${field}"]`)?.classList.toggle('hidden', raw);
        }
    }

    function renderControls(path) {
        for (const group of ['basic', 'advanced', 'expert', 'developer']) {
            const target = byId(`speech-controls-${group}`);
            target.replaceChildren(...controlsForPath(path, group).map(controlInput));
        }
        syncBlendControls();
    }

    function renderPathSelector() {
        const select = byId('speech-preset');
        select.replaceChildren();
        select.appendChild(new Option('Custom pipeline', 'custom'));
        for (const preset of state.discovery.presets || []) {
            if (preset.developer) continue;
            const composition = state.discovery.compositions.find(
                (candidate) => candidate.id === preset.composition_id,
            );
            const option = new Option(
                `${preset.display_name}${composition?.runnable ? '' : ' — unavailable'}`,
                preset.id,
            );
            option.disabled = !composition;
            select.appendChild(option);
        }
        const matchingPreset = (state.discovery.presets || []).find((preset) => (
            preset.composition_id === state.pathKey && preset.id === state.presetId
        ));
        select.value = matchingPreset?.id || 'custom';
        const mock = (state.discovery.compositions || []).find((path) => path.backend === 'mock');
        byId('select-mock-path').classList.toggle('hidden', !mock);
    }

    function componentById(id) {
        return (state.discovery.components || []).find((component) => component.id === id);
    }

    function componentOptions(select, stage, selectedId, compatibleWith = null) {
        select.replaceChildren();
        const candidates = (state.discovery.components || []).filter((component) => (
            component.stage === stage
        ));
        for (const component of candidates) {
            const option = new Option(component.display_name, component.id);
            let reason = '';
            if (compatibleWith) {
                const edge = compatibilityFor(
                    state.discovery,
                    compatibleWith.from === 'selected' ? selectedId : compatibleWith.from,
                    compatibleWith.to === 'component' ? component.id : compatibleWith.to,
                );
                if (!edge?.compatible) reason = edge?.reason || 'No compatible executable connection is registered.';
            }
            if (!component.runnable && component.id !== selectedId) {
                reason ||= component.explanation || 'Component is not ready for runtime synthesis.';
            }
            option.disabled = Boolean(reason);
            option.title = reason;
            if (reason) option.textContent += ` — ${reason}`;
            select.appendChild(option);
        }
        if (![...select.options].some((option) => option.value === selectedId)) {
            select.appendChild(new Option(selectedId, selectedId));
        }
        select.value = selectedId;
    }

    function renderComposition(path) {
        const pipeline = path.pipeline;
        if (!pipeline) return;
        const generatorId = pipeline.end_to_end || pipeline.acoustic_model;
        const generator = componentById(generatorId);
        const projector = componentById(pipeline.projector);
        const generatorSelect = byId('pipeline-generator');
        generatorSelect.replaceChildren();
        const generatorIds = new Set((state.discovery.compositions || []).map(compositionGenerator));
        for (const component of state.discovery.components || []) {
            if (!['acoustic_model', 'end_to_end'].includes(component.stage)) continue;
            const option = new Option(component.display_name, component.id);
            const executable = generatorIds.has(component.id);
            option.disabled = !executable;
            if (!executable) {
                option.title = component.explanation || 'No executable composition is registered.';
                option.textContent += ` — ${option.title}`;
            }
            generatorSelect.appendChild(option);
        }
        if (![...generatorSelect.options].some((option) => option.value === generatorId)) {
            generatorSelect.appendChild(new Option(generatorId, generatorId));
        }
        generatorSelect.value = generatorId;

        const projectorSelect = byId('pipeline-projector');
        projectorSelect.replaceChildren();
        for (const component of (state.discovery.components || []).filter(
            (candidate) => candidate.stage === 'projector',
        )) {
            const edge = compatibilityFor(state.discovery, component.id, generatorId);
            const option = new Option(component.display_name, component.id);
            option.disabled = !edge?.compatible;
            option.title = edge?.reason || 'No compatible projector contract is registered.';
            if (option.disabled) option.textContent += ` — ${option.title}`;
            projectorSelect.appendChild(option);
        }
        projectorSelect.value = pipeline.projector;
        byId('pipeline-projector-detail').textContent = (
            compatibilityFor(state.discovery, pipeline.projector, generatorId)?.reason
            || projector?.explanation
            || 'Checkpoint-bound linguistic projection.'
        );

        const endToEnd = Boolean(pipeline.end_to_end);
        const vocoderStage = byId('pipeline-vocoder-stage');
        vocoderStage.classList.toggle('pipeline-stage-spanned', endToEnd);
        byId('pipeline-generator').previousElementSibling.textContent = endToEnd
            ? 'End-to-end model'
            : 'Acoustic model';
        byId('pipeline-generator-detail').textContent = endToEnd
            ? 'This block spans acoustic generation and waveform decoding.'
            : (generator?.produces?.[0]?.summary || 'Produces features for a compatible vocoder.');
        const vocoder = byId('pipeline-vocoder');
        if (endToEnd) {
            vocoder.replaceChildren(new Option(`Integrated in ${generator?.display_name || generatorId}`, 'integrated'));
            vocoder.disabled = true;
            byId('pipeline-vocoder-detail').textContent = 'Waveform decoding is integrated into this model.';
        } else {
            vocoder.disabled = false;
            componentOptions(vocoder, 'vocoder', pipeline.vocoder, {
                from: generatorId,
                to: 'component',
            });
            byId('pipeline-vocoder-detail').textContent = (
                compatibilityFor(state.discovery, generatorId, pipeline.vocoder)?.reason
                || 'Only exact contract matches can be selected.'
            );
        }
        const conditioners = (pipeline.conditioners || []).map((id) => (
            componentById(id)?.display_name || id
        ));
        byId('pipeline-conditioning-name').textContent = conditioners.join(', ')
            || 'Built-in model conditioning';
        byId('pipeline-conditioning-detail').textContent = [
            path.speakers?.required ? 'speaker required' : null,
            path.languages?.required ? 'model language required' : null,
            path.reference_audio?.speaker ? 'reference voice' : null,
            path.styles?.reference_audio ? 'reference style' : null,
        ].filter(Boolean).join(' · ') || 'Variety and synthesis controls';
    }

    function renderVarieties(path) {
        const select = byId('variety');
        const fixed = byId('fixed-variety');
        const varieties = varietiesForPath(path);
        select.replaceChildren();
        for (const variety of varieties) {
            select.appendChild(new Option(variety.label, variety.id));
        }
        const previous = state.values.get(`variety:${path.id}`);
        if (previous && varieties.some((item) => item.id === previous)) select.value = previous;
        select.classList.toggle('hidden', varieties.length === 1);
        fixed.classList.toggle('hidden', varieties.length !== 1);
        fixed.textContent = varieties[0]?.label || 'No supported variety declared';
        select.disabled = varieties.length < 2;
        byId('variety-detail').textContent = varieties.length === 1
            ? 'This path is fixed to its declared linguistic variety.'
            : `${varieties.length} model-supported varieties.`;
    }

    function renderSpeakers(path) {
        const control = byId('speaker-control');
        const input = byId('speaker');
        const datalist = byId('speaker-options');
        const speakers = listedValues(path.speakers?.values);
        control.classList.toggle('hidden', speakers.length === 0 && !path.speakers?.required);
        datalist.replaceChildren(...speakers.map((speaker) => {
            const option = document.createElement('option');
            option.value = speaker.id;
            option.label = speaker.numeric_id == null
                ? speaker.label
                : `${speaker.label} · embedding ${speaker.numeric_id}`;
            return option;
        }));
        const previous = state.values.get(`speaker:${path.id}`);
        input.value = previous && speakers.some((item) => item.id === previous)
            ? previous
            : (speakers.find((item) => item.id === 'p225')?.id || '');
        input.required = Boolean(path.speakers?.required);
        input.placeholder = speakers.length > 20 ? `Search ${speakers.length} named speakers` : 'Speaker name';
        byId('speaker-detail').textContent = `${speakers.length} named embeddings${path.speakers?.required ? ' · selection required' : ''}.`;
    }

    function renderModelCard(path) {
        const card = byId('speech-model-card');
        const catalog = path.catalog || [];
        const licenses = [...new Set(catalog.map((entry) => entry.license.expression))];
        const provenance = [...new Set(catalog.map((entry) => (
            `${entry.provenance.format} · ${entry.provenance.source}`
        )))];
        const rate = path.output?.sample_rate_hz;
        const languages = [...new Set(catalog.flatMap((entry) => entry.languages || []))];
        const scripts = [...new Set(catalog.map((entry) => entry.script).filter(Boolean))];
        const preprocessing = [...new Set(catalog.flatMap((entry) => entry.preprocessing || []))];
        const statuses = path.statuses.map((status) => `<span class="status-badge">${escapeHtml(status)}</span>`).join('');
        card.innerHTML = `
            <div class="speech-section-heading">
                <div>
                    <p class="eyebrow">${escapeHtml(path.family || 'speech')}</p>
                    <h2>${escapeHtml(path.display_name)}</h2>
                </div>
                <div class="status-badges">${statuses}</div>
            </div>
            <dl class="metadata-grid">
                <div><dt>Architecture</dt><dd>${escapeHtml(catalog.map((entry) => entry.architecture).join(' → ') || path.model)}</dd></div>
                <div><dt>Audio</dt><dd>${rate ? `${rate} Hz · ${path.output.channels} channel` : 'Not declared'}</dd></div>
                <div><dt>Devices</dt><dd>${escapeHtml((path.controls.find((item) => item.field === 'device')?.options || []).map((item) => item.label).join(', '))}</dd></div>
                <div><dt>Language</dt><dd>${escapeHtml(languages.join(', ') || 'Not declared')}</dd></div>
                <div><dt>Script</dt><dd>${escapeHtml(scripts.join(', ') || 'Source default')}</dd></div>
                <div><dt>Preprocessing</dt><dd>${escapeHtml(preprocessing.join(', ') || 'None declared')}</dd></div>
                <div><dt>Speakers</dt><dd>${listedValues(path.speakers?.values).length || catalog.reduce((sum, entry) => sum + (entry.speakers?.count || 0), 0)}</dd></div>
                <div><dt>License</dt><dd>${escapeHtml(licenses.join(', ') || 'Not asserted')}</dd></div>
                <div><dt>Readiness</dt><dd>${escapeHtml(path.runnable ? 'Ready' : path.statuses.join(', '))}</dd></div>
                <div><dt>Resident</dt><dd>${escapeHtml(path.load_state)}</dd></div>
                <div><dt>Provenance</dt><dd>${escapeHtml(provenance.join(' | ') || path.provenance.join(', ') || 'Not declared')}</dd></div>
            </dl>
            ${path.unavailable_reason ? `<p class="inline-error">${escapeHtml(path.unavailable_reason)}</p>` : ''}
            <div class="model-actions">
                ${path.install_command ? `<button type="button" class="secondary-button copy-install-command" data-command="${escapeAttribute(path.install_command)}">Copy install command</button>` : ''}
                ${catalog.length && path.verification_status !== 'verified' && path.verification_status !== 'unavailable' ? '<button type="button" class="secondary-button verify-speech-pipeline">Verify pipeline</button>' : ''}
                ${path.load_state === 'loaded' ? '<button type="button" class="secondary-button unload-speech-path">Unload model</button>' : ''}
            </div>`;
        card.querySelector('.copy-install-command')?.addEventListener('click', async (event) => {
            await navigator.clipboard.writeText(event.currentTarget.dataset.command);
            event.currentTarget.textContent = 'Install command copied';
        });
        card.querySelector('.unload-speech-path')?.addEventListener('click', async (event) => {
            event.currentTarget.disabled = true;
            try {
                const response = await fetch('/api/speech/runtime/unload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(path.pipeline
                        ? { pipeline: path.pipeline }
                        : { backend: path.backend, model: path.model }),
                });
                if (!response.ok) throw new Error(await response.text());
                renderRuntime(await response.json());
                await refreshDiscovery();
            } catch (error) {
                showError(`Unload failed: ${error.message}`);
            }
        });
        card.querySelector('.verify-speech-pipeline')?.addEventListener('click', async (event) => {
            event.currentTarget.disabled = true;
            try {
                await verifyModelIds(catalog.map((entry) => entry.id));
            } catch (error) {
                showError(`Pipeline verification failed: ${error.message}`);
            } finally {
                event.currentTarget.disabled = false;
            }
        });
    }

    function renderSelectedPath() {
        const path = selectedPath();
        if (!path) {
            showError(state.discovery?.error || 'No synthesis paths were discovered.');
            byId('submit-btn').disabled = true;
            return;
        }
        clearError();
        byId('synthesis-path-detail').textContent = path.runnable
            ? `${path.display_name} is a complete, contract-valid pipeline.`
            : 'Pipeline unavailable. Inspect its model card for installation or verification details.';
        renderComposition(path);
        renderVarieties(path);
        renderSpeakers(path);
        renderModelCard(path);
        renderControls(path);
        byId('submit-btn').disabled = !path.complete || !path.runnable;
    }

    function renderInventory() {
        const target = byId('component-inventory');
        const groups = new Map();
        for (const component of state.discovery.components || []) {
            const values = groups.get(component.kind) || [];
            values.push(component);
            groups.set(component.kind, values);
        }
        target.replaceChildren();
        for (const [kind, components] of groups) {
            const section = document.createElement('section');
            section.className = 'component-kind';
            const heading = document.createElement('h3');
            heading.textContent = kind.replaceAll('_', ' ');
            section.appendChild(heading);
            for (const component of components) {
                const details = document.createElement('details');
                details.className = 'component-card';
                const summary = document.createElement('summary');
                const name = document.createElement('strong');
                name.textContent = component.display_name;
                const badges = document.createElement('span');
                badges.className = 'status-badges';
                for (const status of component.statuses || []) {
                    const badge = document.createElement('span');
                    badge.className = 'status-badge';
                    badge.textContent = status;
                    badges.appendChild(badge);
                }
                summary.append(name, badges);
                const body = document.createElement('div');
                body.className = 'component-detail';
                const explanation = document.createElement('p');
                explanation.textContent = component.explanation;
                const facts = document.createElement('p');
                const catalog = component.catalog || [];
                facts.textContent = [
                    `Architecture: ${component.architecture}`,
                    `State: ${component.readiness}`,
                    `Load: ${component.load_state}`,
                    component.control_fields?.length
                        ? `Controls: ${component.control_fields.join(', ')}`
                        : null,
                    catalog.length ? `Language: ${[...new Set(catalog.flatMap((entry) => entry.languages || []))].join(', ')}` : null,
                    catalog.some((entry) => entry.script)
                        ? `Script: ${[...new Set(catalog.map((entry) => entry.script).filter(Boolean))].join(', ')}`
                        : null,
                    catalog.some((entry) => (entry.preprocessing || []).length)
                        ? `Preprocessing: ${[...new Set(catalog.flatMap((entry) => entry.preprocessing || []))].join(', ')}`
                        : null,
                    catalog.length ? `License: ${[...new Set(catalog.map((entry) => entry.license.expression))].join(', ')}` : null,
                    component.compatible_paths.length ? `Paths: ${component.compatible_paths.join(', ')}` : null,
                ].filter(Boolean).join(' · ');
                body.append(explanation, facts);
                if (component.install_command) {
                    const code = document.createElement('code');
                    code.textContent = component.install_command;
                    body.appendChild(code);
                }
                if (
                    catalog.length
                    && component.verification_status !== 'verified'
                    && component.verification_status !== 'unavailable'
                ) {
                    const verify = document.createElement('button');
                    verify.type = 'button';
                    verify.className = 'secondary-button';
                    verify.textContent = 'Verify model';
                    verify.addEventListener('click', async () => {
                        verify.disabled = true;
                        try {
                            await verifyModelIds(catalog.map((entry) => entry.id));
                        } catch (error) {
                            showError(`Model verification failed: ${error.message}`);
                        } finally {
                            verify.disabled = false;
                        }
                    });
                    body.appendChild(verify);
                }
                details.append(summary, body);
                section.appendChild(details);
            }
            target.appendChild(section);
        }
        byId('component-count').textContent = `(${state.discovery.components.length})`;
        const verifyAll = byId('verify-all-models');
        const pending = pendingVerificationIds(state.discovery);
        verifyAll.disabled = pending.length === 0;
        verifyAll.textContent = pending.length
            ? `Verify all changed models (${pending.length})`
            : 'All installed models verified';
    }

    function renderRuntime(runtime) {
        const badge = byId('speech-runtime-state');
        const stateName = runtime.state || 'unknown';
        badge.dataset.state = stateName;
        badge.textContent = stateName;
        const values = [
            ['Device', `${runtime.device || 'unknown'}${Number.isInteger(runtime.device_index) ? ` ${runtime.device_index}` : ''}`],
            ['Requests', `${runtime.active ?? 0} active · ${runtime.queued ?? 0} queued`],
            ['Concurrency', `${runtime.capacity ?? 0} maximum · ${runtime.concurrency || 'unknown'}`],
            ['Warm paths', (runtime.loaded || []).join(', ') || 'None; models load on first synthesis'],
        ];
        const grid = byId('speech-runtime-grid');
        grid.replaceChildren(...values.map(([label, value]) => metadataItem(label, value)));
        const failures = Object.entries(runtime.failed || {});
        const error = byId('speech-runtime-errors');
        if (failures.length) {
            showError(failures.map(([engine, message]) => `${engine}: ${message}`).join(' · '), error);
        } else {
            clearError(error);
        }
    }

    async function loadRuntime(signal) {
        const response = await fetch('/api/speech/runtime', { cache: 'no-store', signal });
        if (!response.ok) throw new Error(await response.text());
        const runtime = await response.json();
        renderRuntime(runtime);
        return runtime;
    }

    function stopRuntimePolling() {
        state.runtimePollGeneration += 1;
        if (state.runtimeTimer != null) {
            window.clearTimeout(state.runtimeTimer);
            state.runtimeTimer = null;
        }
        state.runtimePollController?.abort();
        state.runtimePollController = null;
    }

    function startRuntimePolling() {
        stopRuntimePolling();
        const generation = state.runtimePollGeneration;
        const poll = async () => {
            if (generation !== state.runtimePollGeneration) return;
            const controller = new AbortController();
            state.runtimePollController = controller;
            try {
                await loadRuntime(controller.signal);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    showError(
                        `Runtime status unavailable: ${error.message}`,
                        byId('speech-runtime-errors'),
                    );
                }
            } finally {
                if (state.runtimePollController === controller) {
                    state.runtimePollController = null;
                }
                if (generation === state.runtimePollGeneration) {
                    // Delay from completion so slow runtime requests never overlap.
                    state.runtimeTimer = window.setTimeout(poll, 750);
                }
            }
        };
        poll();
    }

    function metadataItem(label, value) {
        const wrapper = document.createElement('div');
        const term = document.createElement('dt');
        term.textContent = label;
        const detail = document.createElement('dd');
        detail.textContent = String(value ?? '—');
        wrapper.append(term, detail);
        return wrapper;
    }

    function renderResult(metadata, url) {
        byId('audio-player').src = url;
        const download = byId('speech-download');
        download.href = url;
        download.download = `tongues-${metadata.backend || 'speech'}.wav`;
        const fields = [
            ['Pipeline', metadata.pipeline_id || metadata.path],
            ['Backend', metadata.backend],
            ['Projector', metadata.projector],
            ['Acoustic model', metadata.acoustic_model],
            ['Conditioners', (metadata.conditioners || []).join(', ')],
            ['Vocoder', metadata.vocoder],
            ['Voice / model', metadata.voice_model],
            ['Speaker / reference', metadata.speaker || metadata.reference_voice],
            ['Variety', metadata.variety],
            ['Device', `${metadata.device || 'unknown'}${metadata.device_index == null ? '' : ` ${metadata.device_index}`}`],
            ['Audio', `${metadata.sample_rate_hz} Hz · ${metadata.channels} channel · ${metadata.sample_count} samples`],
            ['Duration', `${Number(metadata.duration_seconds || 0).toFixed(3)} s`],
            ['Queue time', `${Number(metadata.queue_ms || 0).toFixed(2)} ms`],
            ['Model load', `${Number(metadata.model_load_ms || 0).toFixed(2)} ms`],
            ['Synthesis', `${Number(metadata.synthesis_ms || 0).toFixed(2)} ms`],
            ['Real-time factor', Number(metadata.real_time_factor || 0).toFixed(3)],
            ['Resident model', metadata.resident_model_reused ? 'Reused' : 'Loaded for this request'],
        ].filter(([, value]) => value != null && value !== '');
        byId('speech-result-metadata').replaceChildren(
            ...fields.map(([label, value]) => metadataItem(label, value)),
        );
        const diagnostics = metadata.diagnostics || {};
        const hasDiagnostics = Object.values(diagnostics).some((value) => (
            Array.isArray(value) ? value.length : value != null
        ));
        byId('speech-result-diagnostics').classList.toggle('hidden', !hasDiagnostics);
        byId('speech-diagnostics-output').textContent = JSON.stringify(diagnostics, null, 2);
        byId('result-container').classList.remove('hidden');
    }

    function duplexLines(source) {
        return String(source || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean);
    }

    function buildDuplexRequest({
        text = '',
        mockAcoustics = '',
        variety = null,
        journalPath = '',
    } = {}) {
        const replayPath = String(journalPath || '').trim();
        if (replayPath) return { journal_path: replayPath };
        const chunks = duplexLines(text);
        const mockChunks = duplexLines(mockAcoustics);
        if (!chunks.length && !mockChunks.length) {
            throw new Error('Enter prompt text, mock acoustic chunks, or a saved journal path.');
        }
        return {
            chunks,
            mock_acoustics: mockChunks,
            variety,
        };
    }

    function duplexTokenMarkup(tokens, kind) {
        if (!tokens?.length) return '<span class="duplex-empty">—</span>';
        return tokens.map((token) => (
            `<span class="duplex-token duplex-token-${kind}">${escapeHtml(token)}</span>`
        )).join('');
    }

    function duplexSurfaceList(items) {
        return (items || []).map((item) => (
            typeof item === 'string' ? item : (item?.surface || item?.key || '')
        )).filter(Boolean);
    }

    function formatDuplexTranscriptEvent(event) {
        if (!event) return 'No transcript delta';
        if (event.type === 'append') return `Append: ${duplexSurfaceList(event.data?.morphemes).join(' ') || '—'}`;
        if (event.type === 'withdraw') return `Withdraw: ${duplexSurfaceList(event.data?.morphemes).join(' ') || '—'}`;
        if (event.type === 'commit') return `Commit: ${duplexSurfaceList(event.data?.morphemes).join(' ') || '—'}`;
        if (event.type === 'replace') {
            const previous = duplexSurfaceList(event.data?.previous).join(' ');
            const replacement = duplexSurfaceList(event.data?.replacement).join(' ');
            return `Replace: ${previous || '—'} → ${replacement || '—'}`;
        }
        return event.type;
    }

    function renderDuplexProjection(projection) {
        const timeline = byId('duplex-timeline');
        const summary = byId('duplex-summary');
        const finalCommitted = projection?.final_state?.committed || [];
        summary.innerHTML = `
            <dl class="metadata-grid">
                <div><dt>Run</dt><dd>${escapeHtml(projection.run_id)}</dd></div>
                <div><dt>Replay</dt><dd>${projection.replay_verified ? 'Verified' : 'Mismatch'}</dd></div>
                <div><dt>Events</dt><dd>${projection.timeline?.length || 0}</dd></div>
                <div><dt>Final commit</dt><dd>${escapeHtml(finalCommitted.map((item) => item.surface).join(' ') || '—')}</dd></div>
            </dl>`;
        summary.classList.remove('hidden');
        timeline.innerHTML = (projection.timeline || []).map((snapshot) => `
            <li class="duplex-step">
                <div class="duplex-step-heading">
                    <strong>#${snapshot.sequence} ${escapeHtml(snapshot.layer)}</strong>
                    <span>${escapeHtml(snapshot.event_type)}</span>
                </div>
                <p class="duplex-step-message">${escapeHtml(snapshot.message)}</p>
                <div class="duplex-track"><span>Observed</span>${duplexTokenMarkup(snapshot.observed, 'observed')}</div>
                <div class="duplex-track"><span>Shared</span>${duplexTokenMarkup(snapshot.shared_prefix, 'inferred')}</div>
                <div class="duplex-track"><span>Committed</span>${duplexTokenMarkup(snapshot.committed, 'committed')}</div>
                <div class="duplex-track"><span>Predicted</span>${duplexTokenMarkup(snapshot.predicted, 'predicted')}</div>
                <div class="duplex-step-meta">
                    <span>Frontier ${snapshot.commit_frontier}</span>
                    <span>${snapshot.first_divergent_morpheme_index == null ? 'No branch divergence' : `First divergent stage ${snapshot.first_divergent_morpheme_index}`}</span>
                    <span>${escapeHtml(formatDuplexTranscriptEvent(snapshot.transcript_event))}</span>
                </div>
                <details class="duplex-branches">
                    <summary>Branches (${snapshot.branches.length})</summary>
                    <ul>
                        ${(snapshot.branches || []).map((branch) => `
                            <li>
                                <strong>${escapeHtml(branch.hypothesis_id)}</strong>
                                <span>${branch.selected ? 'selected' : 'standby'} · p=${Number(branch.probability || 0).toFixed(3)}</span>
                                <span>${escapeHtml(branch.provenance)}</span>
                                <div class="duplex-track">${duplexTokenMarkup(branch.morphemes, branch.selected ? 'predicted' : 'observed')}</div>
                            </li>
                        `).join('')}
                    </ul>
                </details>
            </li>
        `).join('');
        timeline.classList.remove('hidden');
    }

    async function loadDuplexProjection(payload) {
        const response = await fetch('/api/duplex/project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        return response.json();
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (character) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[character]));
    }

    function escapeAttribute(value) {
        return escapeHtml(value).replace(/`/g, '&#96;');
    }

    async function loadAuxiliaryDiscovery() {
        const [sampleResponse, emotionResponse] = await Promise.allSettled([
            fetch('/api/styletts2-samples').then((response) => response.json()),
            fetch('/api/emotions').then((response) => response.json()),
        ]);
        state.samples = sampleResponse.status === 'fulfilled'
            ? (sampleResponse.value.samples || [])
            : [];
        state.emotions = emotionResponse.status === 'fulfilled'
            ? (emotionResponse.value.emotions || [])
            : [];
    }

    function acceptDiscovery(discovery, allowVerificationReset = false) {
        if (
            !allowVerificationReset
            && !preservesVerificationProgress(state.discovery, discovery)
        ) return false;
        const previous = state.pathKey;
        state.discovery = discovery;
        if (
            !(state.discovery.compositions || []).some((composition) => composition.id === previous)
            && !(state.discovery.paths || []).some((path) => pathKey(path) === previous)
        ) {
            const initial = selectInitialComposition(state.discovery)
                || selectInitialPath(state.discovery);
            state.pathKey = initial?.pipeline ? initial.id : (initial ? pathKey(initial) : '');
            const preset = (state.discovery.presets || []).find(
                (candidate) => candidate.composition_id === state.pathKey,
            );
            state.presetId = preset?.id || '';
        }
        renderPathSelector();
        renderInventory();
        renderSelectedPath();
        return true;
    }

    async function verifyModelIds(modelIds, generation = null) {
        const ids = [...new Set((modelIds || []).filter(Boolean))];
        const activeGeneration = generation ?? (state.verificationGeneration + 1);
        if (generation == null) state.verificationGeneration = activeGeneration;
        let cursor = 0;
        const failures = [];
        const verifyNext = async () => {
            while (activeGeneration === state.verificationGeneration && cursor < ids.length) {
                const modelId = ids[cursor];
                cursor += 1;
                try {
                    const response = await fetch(
                        `/api/speech/models/verify/${encodeURIComponent(modelId)}`,
                        { method: 'POST', cache: 'no-store' },
                    );
                    if (!response.ok) throw new Error(await response.text());
                    const updated = await response.json();
                    if (activeGeneration === state.verificationGeneration) {
                        acceptDiscovery(updated);
                    }
                } catch (error) {
                    failures.push(`${modelId}: ${error.message}`);
                }
            }
        };
        await Promise.all(Array.from(
            { length: Math.min(VERIFICATION_CONCURRENCY, ids.length) },
            verifyNext,
        ));
        if (activeGeneration === state.verificationGeneration && ids.length) {
            try {
                const response = await fetch('/api/speech/models', { cache: 'no-store' });
                if (!response.ok) throw new Error(await response.text());
                acceptDiscovery(await response.json());
            } catch (error) {
                failures.push(`final refresh: ${error.message}`);
            }
        }
        if (activeGeneration === state.verificationGeneration && failures.length) {
            throw new Error(failures.join(' · '));
        }
    }

    async function verifyDiscovery(generation, discovery) {
        try {
            await verifyModelIds(pendingVerificationIds(discovery), generation);
        } catch (error) {
            showError(`Background model verification failed: ${error.message}`);
        }
    }

    async function refreshDiscovery(waitForVerification = true) {
        const generation = state.verificationGeneration + 1;
        state.verificationGeneration = generation;
        const response = await fetch('/api/speech/models', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        const discovery = await response.json();
        if (discovery.error && !(discovery.paths || []).length) {
            throw new Error(discovery.error);
        }
        acceptDiscovery(discovery, true);
        const verification = verifyDiscovery(generation, discovery);
        if (waitForVerification) await verification;
        else verification.catch((error) => showError(`Speech verification failed: ${error.message}`));
    }

    async function init() {
        const page = byId('speech-page');
        if (!page) return;
        page.innerHTML = studioShell();
        const submit = byId('submit-btn');
        try {
            await loadAuxiliaryDiscovery();
            await refreshDiscovery(false);
        } catch (error) {
            showError(`Speech discovery failed: ${error.message}`);
            submit.disabled = true;
            byId('speech-runtime-state').dataset.state = 'failed';
            byId('speech-runtime-state').textContent = 'failed';
        }

        byId('speech-preset').addEventListener('change', (event) => {
            const preset = state.discovery.presets.find(
                (candidate) => candidate.id === event.target.value,
            );
            if (!preset) return;
            state.pathKey = preset.composition_id;
            state.presetId = preset.id;
            renderSelectedPath();
        });
        const selectCompositionForStage = (stage, componentId) => {
            const current = selectedComposition();
            const candidates = (state.discovery.compositions || []).filter((composition) => {
                if (stage === 'generator') return compositionGenerator(composition) === componentId;
                if (stage === 'projector') return composition.pipeline.projector === componentId;
                if (stage === 'vocoder') return composition.pipeline.vocoder === componentId;
                return false;
            });
            const next = candidates.find((composition) => (
                composition.runnable
                && (!current || stage === 'generator'
                    || compositionGenerator(composition) === compositionGenerator(current))
            )) || candidates.find((composition) => composition.runnable) || candidates[0];
            if (!next) return;
            state.pathKey = next.id;
            state.presetId = '';
            renderPathSelector();
            renderSelectedPath();
        };
        byId('pipeline-generator').addEventListener('change', (event) => {
            selectCompositionForStage('generator', event.target.value);
        });
        byId('pipeline-projector').addEventListener('change', (event) => {
            selectCompositionForStage('projector', event.target.value);
        });
        byId('pipeline-vocoder').addEventListener('change', (event) => {
            selectCompositionForStage('vocoder', event.target.value);
        });
        byId('variety').addEventListener('change', (event) => {
            const path = selectedPath();
            if (path) state.values.set(`variety:${path.id}`, event.target.value);
        });
        byId('speaker').addEventListener('input', (event) => {
            const path = selectedPath();
            if (path) state.values.set(`speaker:${path.id}`, event.target.value);
        });
        byId('select-mock-path').addEventListener('click', () => {
            const mock = state.discovery.compositions.find((path) => path.backend === 'mock');
            if (!mock) return;
            state.pathKey = mock.id;
            state.presetId = '';
            renderPathSelector();
            renderSelectedPath();
        });
        byId('verify-all-models').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            try {
                await verifyModelIds(pendingVerificationIds(state.discovery));
            } catch (error) {
                showError(`Model verification failed: ${error.message}`);
            } finally {
                renderInventory();
            }
        });
        byId('reload-speech-runtime').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            clearError(byId('speech-runtime-errors'));
            try {
                const response = await fetch('/api/speech/runtime/reload', { method: 'POST' });
                if (!response.ok) throw new Error(await response.text());
                renderRuntime(await response.json());
                await refreshDiscovery();
            } catch (error) {
                showError(`Reload failed: ${error.message}`, byId('speech-runtime-errors'));
                byId('speech-runtime-state').dataset.state = 'failed';
                byId('speech-runtime-state').textContent = 'failed';
            } finally {
                button.disabled = false;
            }
        });
        byId('run-duplex-preview').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            clearError(byId('duplex-error'));
            button.disabled = true;
            try {
                const path = selectedPath();
                const variety = varietiesForPath(path).length === 1
                    ? varietiesForPath(path)[0].id
                    : byId('variety').value;
                const projection = await loadDuplexProjection(buildDuplexRequest({
                    text: byId('text').value,
                    mockAcoustics: byId('duplex-mock-acoustics').value,
                    variety,
                }));
                renderDuplexProjection(projection);
            } catch (error) {
                showError(`Duplex preview failed: ${error.message}`, byId('duplex-error'));
            } finally {
                button.disabled = false;
            }
        });
        byId('replay-duplex-journal').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            clearError(byId('duplex-error'));
            button.disabled = true;
            try {
                const projection = await loadDuplexProjection(buildDuplexRequest({
                    journalPath: byId('duplex-journal-path').value,
                }));
                renderDuplexProjection(projection);
            } catch (error) {
                showError(`Duplex replay failed: ${error.message}`, byId('duplex-error'));
            } finally {
                button.disabled = false;
            }
        });

        byId('synth-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            clearError();
            const path = selectedPath();
            const variety = varietiesForPath(path).length === 1
                ? varietiesForPath(path)[0].id
                : byId('variety').value;
            const emotionName = state.values.get('emotion');
            const emotionVector = state.emotions.find((item) => item.name === emotionName)?.vector;
            let payload;
            let projection = null;
            try {
                payload = buildPayload(path, state.values, {
                    text: byId('text').value,
                    variety,
                    speaker: byId('speaker-control').classList.contains('hidden')
                        ? null
                        : byId('speaker').value.trim(),
                    emotionVector,
                });
                const tokenFields = ['pitch', 'energy', 'durations']
                    .filter((field) => Array.isArray(payload[field]));
                if (tokenFields.length) {
                    const projectionResponse = await fetch('/api/speech/project', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: payload.text,
                            variety: payload.variety,
                            ...(payload.pipeline
                                ? { pipeline: payload.pipeline }
                                : { backend: payload.backend }),
                        }),
                    });
                    if (!projectionResponse.ok) {
                        throw new Error(await projectionResponse.text());
                    }
                    projection = await projectionResponse.json();
                    for (const field of tokenFields) {
                        if (payload[field].length !== projection.projected_token_count) {
                            throw new Error(
                                `${field} has ${payload[field].length} values, but the selected model projects ${projection.projected_token_count} tokens.`,
                            );
                        }
                    }
                }
            } catch (error) {
                showError(error.message);
                return;
            }
            submit.disabled = true;
            submit.classList.add('loading');
            byId('speech-submit-status').textContent = 'Synthesis in progress.';
            byId('result-container').classList.add('hidden');
            startRuntimePolling();
            try {
                const response = await fetch('/api/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!response.ok) throw new Error(await response.text());
                const metadata = JSON.parse(response.headers.get('X-Tongues-Speech-Metadata') || '{}');
                if (projection) {
                    metadata.diagnostics = {
                        ...(metadata.diagnostics || {}),
                        projection,
                    };
                }
                const blob = await response.blob();
                if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
                state.audioUrl = URL.createObjectURL(blob);
                renderResult(metadata, state.audioUrl);
                byId('speech-submit-status').textContent = 'Speech synthesis complete.';
                byId('audio-player').play().catch(() => {});
            } catch (error) {
                showError(`Synthesis failed: ${error.message}`);
                byId('speech-submit-status').textContent = 'Speech synthesis failed.';
            } finally {
                stopRuntimePolling();
                submit.classList.remove('loading');
                submit.disabled = !selectedPath()?.runnable;
                loadRuntime().catch((error) => {
                    showError(`Runtime status unavailable: ${error.message}`, byId('speech-runtime-errors'));
                });
            }
        });

        loadRuntime().catch((error) => {
            byId('speech-runtime-state').dataset.state = 'failed';
            byId('speech-runtime-state').textContent = 'failed';
            showError(`Runtime status unavailable: ${error.message}`, byId('speech-runtime-errors'));
        });
    }

    return {
        availablePaths,
        availableCompositions,
        buildPayload,
        buildDuplexRequest,
        compatibilityFor,
        compositionGenerator,
        controlsForPath,
        duplexLines,
        init,
        parseNumberArray,
        pathKey,
        pendingVerificationIds,
        preservesVerificationProgress,
        selectInitialPath,
        selectInitialComposition,
        varietiesForPath,
    };
}));
