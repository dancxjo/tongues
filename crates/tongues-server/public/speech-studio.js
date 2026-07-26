(function speechStudioModule(root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.SpeechStudio = api;
}(typeof window !== 'undefined' ? window : null, function buildSpeechStudio() {
    'use strict';

    const state = {
        discovery: null,
        pathKey: '',
        values: new Map(),
        samples: [],
        emotions: [],
        audioUrl: null,
        runtimeTimer: null,
    };

    const pathKey = (path) => `${path.backend}::${path.model}`;
    const selectedPath = () => state.discovery?.paths.find((path) => pathKey(path) === state.pathKey);
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
    const varietiesForPath = (path) => listedValues(path?.varieties);
    const controlsForPath = (path, group) => (
        (path?.controls || []).filter((control) => control.group === group)
    );

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
            backend: path.backend,
            model: path.model,
            variety: context.variety || varietiesForPath(path)[0]?.id || null,
        };
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
                !['text', 'backend', 'model', 'variety', 'speaker', 'emotion_vector'].includes(field)
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
                    <p>Choose a complete text-to-audio path. Tongues reports acoustic models,
                    vocoders, voices, speakers, varieties, and controls from server discovery.</p>
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
                        <small>Text is planned and synthesized through the selected complete path.</small>
                    </div>
                    <div class="controls-grid">
                        <div class="form-group">
                            <label for="synthesis-path">Synthesis path</label>
                            <select id="synthesis-path"></select>
                            <small id="synthesis-path-detail">Loading server discovery…</small>
                        </div>
                        <div class="form-group" id="variety-control">
                            <label for="variety">Linguistic variety</label>
                            <select id="variety"></select>
                            <div id="fixed-variety" class="fixed-value hidden"></div>
                            <small id="variety-detail"></small>
                        </div>
                        <div class="form-group hidden" id="acoustic-model-control">
                            <label for="acoustic-model">Acoustic model</label>
                            <select id="acoustic-model" disabled></select>
                            <small>Produces acoustic features; it cannot produce a WAV alone.</small>
                        </div>
                        <div class="form-group hidden" id="vocoder-control">
                            <label for="vocoder">Compatible vocoder</label>
                            <select id="vocoder"></select>
                            <small id="vocoder-detail"></small>
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

                <details class="engine-inventory">
                    <summary>Engines and Models <span id="component-count"></span></summary>
                    <p>All native, catalog, compatibility, and test components are listed even when
                    they cannot independently produce audio.</p>
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
        const select = byId('synthesis-path');
        select.replaceChildren();
        const ready = document.createElement('optgroup');
        ready.label = 'Ready synthesis paths';
        const unavailable = document.createElement('optgroup');
        unavailable.label = 'Unavailable synthesis paths';
        for (const path of availablePaths(state.discovery)) {
            const option = document.createElement('option');
            option.value = pathKey(path);
            option.textContent = `${path.display_name}${path.runnable ? '' : ' — unavailable'}`;
            option.disabled = !path.complete;
            (path.runnable ? ready : unavailable).appendChild(option);
        }
        if (ready.children.length) select.appendChild(ready);
        if (unavailable.children.length) select.appendChild(unavailable);
        select.value = state.pathKey;
        const mock = (state.discovery.paths || []).find((path) => path.backend === 'mock');
        byId('select-mock-path').classList.toggle('hidden', !mock);
    }

    function renderComposition(path) {
        const acousticGroup = byId('acoustic-model-control');
        const vocoderGroup = byId('vocoder-control');
        acousticGroup.classList.toggle('hidden', !path.acoustic_model);
        vocoderGroup.classList.toggle('hidden', !path.vocoder);
        if (!path.acoustic_model) return;
        const acoustic = byId('acoustic-model');
        acoustic.replaceChildren(new Option(
            path.catalog.find((item) => item.id === path.acoustic_model)?.display_name
                || path.acoustic_model,
            path.acoustic_model,
        ));
        const vocoder = byId('vocoder');
        vocoder.replaceChildren();
        for (const match of path.compatible_vocoders || []) {
            const component = state.discovery.components.find((item) => (
                item.id === match.component_id
                || item.catalog.some((entry) => entry.id === match.component_id)
            ));
            const option = new Option(component?.display_name || match.component_id, match.component_id);
            option.disabled = !match.compatible;
            option.title = match.reason;
            vocoder.appendChild(option);
        }
        vocoder.value = path.vocoder;
        const selected = (path.compatible_vocoders || []).find((item) => item.component_id === path.vocoder);
        byId('vocoder-detail').textContent = selected?.reason || 'Only compatible vocoders are selectable.';
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
                <div><dt>Speakers</dt><dd>${listedValues(path.speakers?.values).length || catalog.reduce((sum, entry) => sum + (entry.speakers?.count || 0), 0)}</dd></div>
                <div><dt>License</dt><dd>${escapeHtml(licenses.join(', ') || 'Not asserted')}</dd></div>
                <div><dt>Resident</dt><dd>${escapeHtml(path.load_state)}</dd></div>
                <div><dt>Provenance</dt><dd>${escapeHtml(provenance.join(' | ') || path.provenance.join(', ') || 'Not declared')}</dd></div>
            </dl>
            ${path.unavailable_reason ? `<p class="inline-error">${escapeHtml(path.unavailable_reason)}</p>` : ''}
            <div class="model-actions">
                ${path.install_command ? `<button type="button" class="secondary-button copy-install-command" data-command="${escapeAttribute(path.install_command)}">Copy install command</button>` : ''}
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
                    body: JSON.stringify({ backend: path.backend, model: path.model }),
                });
                if (!response.ok) throw new Error(await response.text());
                renderRuntime(await response.json());
                await refreshDiscovery();
            } catch (error) {
                showError(`Unload failed: ${error.message}`);
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
        byId('synthesis-path').value = state.pathKey;
        byId('synthesis-path-detail').textContent = path.runnable
            ? `${path.display_name} is complete and ready.`
            : (path.unavailable_reason || 'This path is not runnable.');
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
                    catalog.length ? `License: ${[...new Set(catalog.map((entry) => entry.license.expression))].join(', ')}` : null,
                    component.compatible_paths.length ? `Paths: ${component.compatible_paths.join(', ')}` : null,
                ].filter(Boolean).join(' · ');
                body.append(explanation, facts);
                if (component.install_command) {
                    const code = document.createElement('code');
                    code.textContent = component.install_command;
                    body.appendChild(code);
                }
                details.append(summary, body);
                section.appendChild(details);
            }
            target.appendChild(section);
        }
        byId('component-count').textContent = `(${state.discovery.components.length})`;
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

    async function loadRuntime() {
        const response = await fetch('/api/speech/runtime', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        const runtime = await response.json();
        renderRuntime(runtime);
        return runtime;
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
            ['Complete path', metadata.path],
            ['Backend', metadata.backend],
            ['Acoustic model', metadata.acoustic_model],
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

    async function refreshDiscovery() {
        const response = await fetch('/api/speech/models', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        const previous = state.pathKey;
        state.discovery = await response.json();
        if (!(state.discovery.paths || []).some((path) => pathKey(path) === previous)) {
            const initial = selectInitialPath(state.discovery);
            state.pathKey = initial ? pathKey(initial) : '';
        }
        renderPathSelector();
        renderInventory();
        renderSelectedPath();
    }

    async function init() {
        const page = byId('speech-page');
        if (!page) return;
        page.innerHTML = studioShell();
        const submit = byId('submit-btn');
        try {
            await loadAuxiliaryDiscovery();
            const response = await fetch('/api/speech/models', { cache: 'no-store' });
            if (!response.ok) throw new Error(await response.text());
            state.discovery = await response.json();
            if (state.discovery.error && !state.discovery.paths.length) {
                throw new Error(state.discovery.error);
            }
            const initial = selectInitialPath(state.discovery);
            state.pathKey = initial ? pathKey(initial) : '';
            renderPathSelector();
            renderInventory();
            renderSelectedPath();
        } catch (error) {
            showError(`Speech discovery failed: ${error.message}`);
            submit.disabled = true;
            byId('speech-runtime-state').dataset.state = 'failed';
            byId('speech-runtime-state').textContent = 'failed';
        }

        byId('synthesis-path').addEventListener('change', (event) => {
            state.pathKey = event.target.value;
            renderSelectedPath();
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
            const mock = state.discovery.paths.find((path) => path.backend === 'mock');
            if (!mock) return;
            state.pathKey = pathKey(mock);
            const select = byId('synthesis-path');
            if (![...select.options].some((option) => option.value === state.pathKey)) {
                const group = document.createElement('optgroup');
                group.label = 'Developer / testing';
                group.appendChild(new Option(mock.display_name, state.pathKey));
                select.appendChild(group);
            }
            renderSelectedPath();
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
                            backend: payload.backend,
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
            state.runtimeTimer = window.setInterval(() => loadRuntime().catch(() => {}), 750);
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
                window.clearInterval(state.runtimeTimer);
                state.runtimeTimer = null;
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
        buildPayload,
        controlsForPath,
        init,
        parseNumberArray,
        pathKey,
        selectInitialPath,
        varietiesForPath,
    };
}));
