const commandPages = [
    {
        title: 'StyleTTS2 Speak',
        path: '/styletts2',
        command: 'tongues speak',
        group: 'Speech',
        summary: 'Generate speech with StyleTTS2 voice, style, and emotion controls.',
        implemented: true,
    },
    {
        title: 'Speak',
        path: '/cli/speak',
        command: 'tongues speak',
        group: 'Speech',
        summary: 'Synthesize text into WAV output using the selected speech backend.',
        fields: ['text', '--variety', '--backend', '--output', '--voice-wav', '--style-wav'],
    },
    {
        title: 'Phonemes',
        path: '/cli/phonemes',
        command: 'tongues phonemes',
        group: 'Speech',
        summary: 'Convert text into a broad IPA phoneme sequence.',
        fields: ['text'],
    },
    {
        title: 'Phones',
        path: '/cli/phones',
        command: 'tongues phones',
        group: 'Speech',
        summary: 'Convert text into a narrow IPA phone sequence.',
        fields: ['text'],
    },
    {
        title: 'G2P2G',
        path: '/g2p2g/prepare',
        command: 'tongues g2p2g prepare',
        group: 'G2P2G',
        summary: 'Prepare OpenEPD train, validation, and test splits.',
        fields: ['--config', '--input', '--out', '--train-frac', '--valid-frac', '--seed'],
    },
    {
        title: 'G2P2G Clean',
        path: '/g2p2g/clean',
        command: 'tongues g2p2g clean',
        group: 'G2P2G',
        summary: 'Archive selected G2P2G artifacts and recreate default directories.',
        fields: ['--data', '--model', '--all', '--archive-dir', '--run-id', '--no-create'],
    },
    {
        title: 'G2P2G Train',
        path: '/g2p2g/train',
        command: 'tongues g2p2g train',
        group: 'G2P2G',
        summary: 'Train the G2P2G seq2seq model.',
        fields: ['--config', '--data', '--out', '--task', '--epochs', '--batch-size'],
    },
    {
        title: 'G2P2G Infer',
        path: '/g2p2g/infer',
        command: 'tongues g2p2g infer',
        group: 'G2P2G',
        summary: 'Run grapheme-to-phoneme or phoneme-to-grapheme inference.',
        fields: ['input', '--task', '--model', '--data'],
    },
    {
        title: 'G2P2G Eval',
        path: '/g2p2g/eval',
        command: 'tongues g2p2g eval',
        group: 'G2P2G',
        summary: 'Evaluate a trained G2P2G model on a prepared split.',
        fields: ['--model', '--split', '--data', '--task'],
    },
    {
        title: 'G2P2G Refine',
        path: '/g2p2g/refine',
        command: 'tongues g2p2g refine',
        group: 'G2P2G',
        summary: 'Fine-tune a G2P2G model on held-out discrepancies or sight words.',
        fields: ['--model', '--data', '--out', '--splits', '--source', '--task'],
    },
    {
        title: 'G2P2G Repl',
        path: '/g2p2g/repl',
        command: 'tongues g2p2g repl',
        group: 'G2P2G',
        summary: 'Run an interactive G2P2G translation session.',
        fields: ['--task', '--model', '--data'],
    },
    {
        title: 'Sentence Parser',
        path: '/sentence-parser/prepare',
        command: 'tongues sentence-parser prepare',
        group: 'Sentence Parser',
        summary: 'Prepare sentence parser data from text files or directories.',
        fields: ['--config', '--input', '--out'],
    },
    {
        title: 'Sentence Parser Clean',
        path: '/sentence-parser/clean',
        command: 'tongues sentence-parser clean',
        group: 'Sentence Parser',
        summary: 'Archive selected sentence parser artifacts and recreate default directories.',
        fields: ['--data', '--model', '--all', '--archive-dir', '--run-id', '--no-create'],
    },
    {
        title: 'Sentence Parser Train',
        path: '/sentence-parser/train',
        command: 'tongues sentence-parser train',
        group: 'Sentence Parser',
        summary: 'Train or scaffold the sentence parser model.',
        fields: ['--config', '--data', '--input', '--out', '--prepare', '--training-set'],
    },
    {
        title: 'Sentence Parser Eval',
        path: '/sentence-parser/eval',
        command: 'tongues sentence-parser eval',
        group: 'Sentence Parser',
        summary: 'Validate a sentence parser artifact scaffold.',
        fields: ['--model', '--split'],
    },
    {
        title: 'Sentence Parser Parse',
        path: '/sentence-parser/parse',
        command: 'tongues sentence-parser parse',
        group: 'Sentence Parser',
        summary: 'Parse a sentence into the speech syntax analysis shape.',
        fields: ['text', '--model'],
    },
    {
        title: 'Sentence Parser Infer',
        path: '/sentence-parser/infer',
        command: 'tongues sentence-parser infer',
        group: 'Sentence Parser',
        summary: 'Run cursor-time sentence-boundary seq2seq inference.',
        fields: ['cursor', '--model', '--previous'],
    },
    {
        title: 'Sentence Parser Stream',
        path: '/sentence-parser/stream',
        command: 'tongues sentence-parser stream',
        group: 'Sentence Parser',
        summary: 'Stream stdin through the cursor-time sentence parser.',
        fields: ['--model', '--repair-control'],
    },
    {
        title: 'Head2Phones',
        path: '/head2phones/prepare',
        command: 'tongues head2phones prepare',
        group: 'Head2Phones',
        summary: 'Prepare rolling head-chunk-to-phones training data.',
        fields: ['--config', '--input', '--out', '--verify-ollama'],
    },
    {
        title: 'Head2Phones Clean',
        path: '/head2phones/clean',
        command: 'tongues head2phones clean',
        group: 'Head2Phones',
        summary: 'Archive selected head2phones artifacts and recreate default directories.',
        fields: ['--data', '--model', '--all', '--archive-dir', '--run-id', '--no-create'],
    },
    {
        title: 'Head2Phones Verify',
        path: '/head2phones/verify',
        command: 'tongues head2phones verify',
        group: 'Head2Phones',
        summary: 'Passively verify prepared head2phones training rows with Ollama.',
        fields: ['--config', '--data', '--ollama-model', '--ollama-url', '--strict'],
    },
    {
        title: 'Head2Phones Train',
        path: '/head2phones/train',
        command: 'tongues head2phones train',
        group: 'Head2Phones',
        summary: 'Train the rolling head-chunk-to-phones seq2seq model.',
        fields: ['--config', '--data', '--input', '--out', '--prepare', '--epochs'],
    },
    {
        title: 'Head2Phones Infer',
        path: '/head2phones/infer',
        command: 'tongues head2phones infer',
        group: 'Head2Phones',
        summary: 'Run rolling-buffer head2phones inference.',
        fields: ['buffer', '--model', '--variety'],
    },
    {
        title: 'Interpretation',
        path: '/interpretation/prepare',
        command: 'tongues interpretation prepare',
        group: 'Interpretation',
        summary: 'Prepare LibriSpeech audio supervision data.',
        fields: ['--subset', '--out', '--max-utterances', '--whisper-model'],
    },
    {
        title: 'Interpretation Clean',
        path: '/interpretation/clean',
        command: 'tongues interpretation clean',
        group: 'Interpretation',
        summary: 'Archive selected interpretation artifacts and recreate default directories.',
        fields: ['--data', '--model', '--all', '--archive-dir', '--run-id', '--no-create'],
    },
    {
        title: 'Interpretation Train',
        path: '/interpretation/train',
        command: 'tongues interpretation train',
        group: 'Interpretation',
        summary: 'Train the LibriSpeech ASR model.',
        fields: ['--data', '--out', '--wait-for-prepare', '--epochs', '--batch-size', '--seed'],
    },
    {
        title: 'Interpretation Eval',
        path: '/interpretation/eval',
        command: 'tongues interpretation eval',
        group: 'Interpretation',
        summary: 'Evaluate a LibriSpeech ASR model.',
        fields: ['--model', '--data', '--split'],
    },
    {
        title: 'Interpretation Stream',
        path: '/interpretation/stream',
        command: 'tongues interpretation stream',
        group: 'Interpretation',
        summary: 'Stream a WAV file through the ASR model.',
        fields: ['--model', '--wav'],
    },
    {
        title: 'Wiktionary',
        path: '/wiktionary/prepare',
        command: 'tongues wiktionary prepare',
        group: 'Wiktionary',
        summary: 'Download and prepare Wiktionary pronunciation data.',
        fields: ['--config', '--dump', '--out', '--cache-dir', '--lang'],
    },
    {
        title: 'Wiktionary Clean',
        path: '/wiktionary/clean',
        command: 'tongues wiktionary clean',
        group: 'Wiktionary',
        summary: 'Archive selected Wiktionary artifacts and recreate default directories.',
        fields: ['--data', '--model', '--all', '--archive-dir', '--run-id', '--no-create'],
    },
    {
        title: 'Wiktionary Train',
        path: '/wiktionary/train',
        command: 'tongues wiktionary train',
        group: 'Wiktionary',
        summary: 'Train a Wiktionary pronunciation seq2seq model.',
        fields: ['--config', '--data', '--out', '--lang', '--notation', '--task'],
    },
    {
        title: 'Wiktionary Infer',
        path: '/wiktionary/infer',
        command: 'tongues wiktionary infer',
        group: 'Wiktionary',
        summary: 'Run pronunciation and normalization tasks with a trained Wiktionary model.',
        fields: ['input', '--model', '--task', '--lang', '--notation', '--variety'],
    },
    {
        title: 'Models Menu',
        path: '/models/menu',
        command: 'tongues models menu',
        group: 'Utilities',
        summary: 'Choose the active model through the CLI menu.',
        fields: ['model category', 'bundle'],
    },
    {
        title: 'Models List',
        path: '/models/list',
        command: 'tongues models list',
        group: 'Utilities',
        summary: 'List known model bundles.',
        fields: ['kind', 'id', 'display name', 'presence'],
    },
    {
        title: 'Models Path',
        path: '/models/path',
        command: 'tongues models path',
        group: 'Utilities',
        summary: 'Print model paths and current selection.',
        fields: ['model'],
    },
    {
        title: 'Models Status',
        path: '/models/status',
        command: 'tongues models status',
        group: 'Utilities',
        summary: 'Show selected models and local file presence.',
        fields: ['selected model', 'selected Piper voice', 'file presence'],
    },
    {
        title: 'Models Use',
        path: '/models/use',
        command: 'tongues models use',
        group: 'Utilities',
        summary: 'Select the active LLM model.',
        fields: ['model'],
    },
    {
        title: 'Models Fetch',
        path: '/models/fetch',
        command: 'tongues models fetch',
        group: 'Utilities',
        summary: 'Fetch default runtime models or a named model.',
        fields: ['model', '--force'],
    },
    {
        title: 'Fetch Corpora',
        path: '/cli/fetch-corpora',
        command: 'tongues fetch-corpora',
        group: 'Utilities',
        summary: 'Download public emotion corpora for StyleTTS2 signatures.',
        fields: ['--out-dir', '--corpus', '--list'],
    },
    {
        title: 'Fetch CMUdict',
        path: '/cli/fetch-cmudict',
        command: 'tongues fetch-cmudict',
        group: 'Utilities',
        summary: 'Download CMUdict from GitHub.',
        fields: ['--out'],
    },
    {
        title: 'Discrepancies',
        path: '/cli/discrepancies',
        command: 'tongues discrepancies',
        group: 'Utilities',
        summary: 'Compare pronunciations from lexicons, rules, and trained models.',
        fields: ['--out', '--limit', '--max-rarity', '--word', '--words-file'],
    },
    {
        title: 'StyleTTS2 Discover',
        path: '/cli/styletts2/discover',
        command: 'tongues styletts2 discover',
        group: 'StyleTTS2',
        summary: 'Sample diffusion parameters and synthesize StyleTTS2 variants.',
        fields: ['text', '--out-dir', '--num-samples', '--head2phones-model', '--tier'],
    },
    {
        title: 'StyleTTS2 Encode Style',
        path: '/cli/styletts2/encode-style',
        command: 'tongues styletts2 encode-style',
        group: 'StyleTTS2',
        summary: 'Batch-encode reference WAV files into StyleTTS2 style vectors.',
        fields: ['refs', '--out', '--labels'],
    },
    {
        title: 'StyleTTS2 Emotion Signatures',
        path: '/cli/styletts2/emotion-signatures',
        command: 'tongues styletts2 emotion-signatures',
        group: 'StyleTTS2',
        summary: 'Compute emotion delta signatures from encoded style vectors.',
        fields: ['style-vectors', '--method', '--out'],
    },
    {
        title: 'Legacy Prepare',
        path: '/cli/prepare',
        command: 'tongues prepare',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G prepare.',
        fields: ['--input', '--out', '--train-frac', '--valid-frac', '--seed'],
    },
    {
        title: 'Legacy Train',
        path: '/cli/train',
        command: 'tongues train',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G train.',
        fields: ['--data', '--out', '--task', '--epochs', '--batch-size'],
    },
    {
        title: 'Legacy Eval',
        path: '/cli/eval',
        command: 'tongues eval',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G eval.',
        fields: ['--model', '--split', '--data', '--task'],
    },
    {
        title: 'Legacy Refine',
        path: '/cli/refine',
        command: 'tongues refine',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G refine.',
        fields: ['--model', '--data', '--out', '--splits', '--source', '--task'],
    },
    {
        title: 'Legacy Repl',
        path: '/cli/repl',
        command: 'tongues repl',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G repl.',
        fields: ['--task', '--model', '--data'],
    },
    {
        title: 'Legacy Predict',
        path: '/cli/predict',
        command: 'tongues predict',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G infer.',
        fields: ['input', '--task', '--model', '--data'],
    },
];

const byId = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', async () => {
    renderNavigation();
    renderRoute();
    window.addEventListener('popstate', renderRoute);
    await initStyleTts2();
});

function renderNavigation() {
    const nav = byId('primary-nav');
    const groups = [...new Set(commandPages.map((page) => page.group))];
    nav.innerHTML = groups.map((group) => {
        const links = commandPages
            .filter((page) => page.group === group)
            .map((page) => `<a href="${page.path}" data-route="${page.path}">${page.title}</a>`)
            .join('');
        return `<div class="nav-group"><div class="nav-heading">${group}</div>${links}</div>`;
    }).join('');

    nav.addEventListener('click', (event) => {
        const link = event.target.closest('a[data-route]');
        if (!link) return;
        event.preventDefault();
        history.pushState({}, '', link.getAttribute('href'));
        renderRoute();
    });
}

function renderRoute() {
    const path = normalizePath(window.location.pathname);
    const page = commandPages.find((candidate) => path === candidate.path)
        || commandPages.find((candidate) => path.startsWith(candidate.path + '/'));

    byId('styletts2-page').classList.toggle('hidden', !page?.implemented);
    byId('dashboard-page').classList.toggle('hidden', Boolean(page));
    byId('skeleton-page').classList.toggle('hidden', !page || page.implemented);

    document.querySelectorAll('[data-route]').forEach((link) => {
        link.classList.toggle('active', page && link.dataset.route === page.path);
    });

    if (!page) {
        renderDashboard();
        byId('page-kicker').textContent = 'Command surface';
        byId('page-title').textContent = 'Tongues Web';
        byId('page-summary').textContent = 'Every CLI command has a web route reserved here.';
        byId('page-command').textContent = 'tongues';
        return;
    }

    byId('page-kicker').textContent = page.group;
    byId('page-title').textContent = page.title;
    byId('page-summary').textContent = page.summary;
    byId('page-command').textContent = page.command;

    if (!page.implemented) {
        renderSkeleton(page);
    }
}

function renderDashboard() {
    const grid = byId('dashboard-grid');
    grid.innerHTML = commandPages.map((page) => `
        <a class="command-card" href="${page.path}" data-dashboard-route="${page.path}">
            <span>${page.group}</span>
            <strong>${page.title}</strong>
            <small>${page.command}</small>
        </a>
    `).join('');

    grid.querySelectorAll('[data-dashboard-route]').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            history.pushState({}, '', link.getAttribute('href'));
            renderRoute();
        });
    });
}

function renderSkeleton(page) {
    byId('command-preview').value = page.command;
    byId('skeleton-fields').innerHTML = (page.fields || []).map((field) => `
        <div class="form-group">
            <label>${field}</label>
            <input type="text" placeholder="${field}">
        </div>
    `).join('');
    byId('skeleton-notes').value = `${page.command}\n\n${page.summary}`;
}

function normalizePath(path) {
    if (path.length > 1 && path.endsWith('/')) return path.slice(0, -1);
    return path;
}

async function initStyleTts2() {
    const emotionSelect = byId('emotion');
    const strengthInput = byId('emotion_strength');
    const strengthVal = byId('strength-val');
    const emotionDetail = byId('emotion-detail');
    const strengthPresets = byId('strength-presets');
    const voiceSelect = byId('voice_sample');
    const styleSelect = byId('style_sample');
    const voicePreview = byId('voice-preview');
    const stylePreview = byId('style-preview');
    const blendMode = byId('blend_mode');
    const form = byId('synth-form');
    const btn = byId('submit-btn');
    const resultContainer = byId('result-container');
    const audioPlayer = byId('audio-player');

    if (!form) return;

    const emotions = new Map();
    const samples = new Map();

    const numericControls = [
        ['diffusion_steps', 'diffusion-steps-val', 0],
        ['speed', 'speed-val', 2],
        ['speaker_reference_strength', 'speaker-strength-val', 2],
        ['style_reference_strength', 'style-strength-val', 2],
        ['style_alpha', 'alpha-val', 2],
        ['style_beta', 'beta-val', 2],
        ['embedding_scale', 'embedding-scale-val', 2],
    ];

    const setStrength = (value) => {
        strengthInput.value = value.toFixed(2);
        strengthVal.textContent = value.toFixed(2);
    };

    const formatEmotionName = (name) => name.charAt(0).toUpperCase() + name.slice(1);

    const formatDuration = (durationMs) => {
        if (!durationMs) return '';
        const seconds = durationMs / 1000;
        return ` (${seconds.toFixed(1)} s)`;
    };

    const updateEmotionDetail = () => {
        const selected = emotions.get(emotionSelect.value);
        if (!selected) {
            emotionDetail.textContent = 'No emotion signature selected';
            return;
        }

        const stats = selected.stats || {};
        const sampleCount = stats.sample_count || 0;
        const speakerCount = stats.n_speakers || 0;
        emotionDetail.textContent = `${selected.dims || selected.vector.length} dims · ${speakerCount} speakers · ${sampleCount} samples`;
    };

    const updatePreview = (select, audio) => {
        const sample = samples.get(select.value);
        if (!sample) {
            audio.removeAttribute('src');
            audio.classList.add('empty');
            return;
        }
        audio.src = sample.audio_url;
        audio.classList.remove('empty');
    };

    const syncBlendMode = () => {
        const raw = blendMode.value === 'raw';
        document.querySelectorAll('.blend-strength').forEach((node) => node.classList.toggle('hidden', raw));
        document.querySelectorAll('.blend-raw').forEach((node) => node.classList.toggle('hidden', !raw));
    };

    numericControls.forEach(([inputId, outputId, precision]) => {
        const input = byId(inputId);
        const output = byId(outputId);
        const sync = () => {
            const value = Number(input.value);
            output.textContent = precision === 0 ? String(value) : value.toFixed(precision);
        };
        input.addEventListener('input', sync);
        sync();
    });

    strengthInput.addEventListener('input', (event) => {
        setStrength(parseFloat(event.target.value));
    });

    emotionSelect.addEventListener('change', () => {
        const selected = emotions.get(emotionSelect.value);
        if (selected && selected.recommended_strength) {
            setStrength(selected.recommended_strength.normal || 0.65);
        }
        updateEmotionDetail();
    });

    strengthPresets.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-preset]');
        const selected = emotions.get(emotionSelect.value);
        if (!button || !selected || !selected.recommended_strength) return;
        const value = selected.recommended_strength[button.dataset.preset];
        if (typeof value === 'number') {
            setStrength(value);
        }
    });

    voiceSelect.addEventListener('change', () => updatePreview(voiceSelect, voicePreview));
    styleSelect.addEventListener('change', () => updatePreview(styleSelect, stylePreview));
    blendMode.addEventListener('change', syncBlendMode);
    syncBlendMode();

    const loadEmotions = async () => {
        const res = await fetch('/api/emotions');
        const data = await res.json();

        if (data.error) {
            emotionDetail.textContent = data.error;
            return;
        }

        if (data.emotions && data.emotions.length > 0) {
            data.emotions.forEach((em) => {
                if (!Array.isArray(em.vector)) return;
                emotions.set(em.name, em);
                const option = document.createElement('option');
                option.value = em.name;
                option.textContent = formatEmotionName(em.name);
                emotionSelect.appendChild(option);
            });
            emotionDetail.textContent = `${data.emotions.length} emotion signatures loaded`;
        } else {
            emotionDetail.textContent = 'No emotion signatures found';
        }
    };

    const loadSamples = async () => {
        const res = await fetch('/api/styletts2-samples');
        const data = await res.json();
        if (data.error) {
            const option = document.createElement('option');
            option.textContent = data.error;
            option.disabled = true;
            voiceSelect.appendChild(option.cloneNode(true));
            styleSelect.appendChild(option);
            return;
        }

        data.samples.forEach((sample) => {
            samples.set(sample.id, sample);
            const voiceOption = document.createElement('option');
            voiceOption.value = sample.id;
            voiceOption.textContent = `${sample.label}${formatDuration(sample.duration_ms)}`;
            const styleOption = voiceOption.cloneNode(true);
            voiceSelect.appendChild(voiceOption);
            styleSelect.appendChild(styleOption);
        });

        const defaults = data.defaults || {};
        if (samples.has(defaults.voice)) voiceSelect.value = defaults.voice;
        if (samples.has(defaults.style)) styleSelect.value = defaults.style;
        updatePreview(voiceSelect, voicePreview);
        updatePreview(styleSelect, stylePreview);
    };

    await Promise.all([
        loadEmotions().catch((err) => {
            console.error('Failed to load emotions', err);
            emotionDetail.textContent = 'Failed to load emotion signatures';
        }),
        loadSamples().catch((err) => {
            console.error('Failed to load StyleTTS2 samples', err);
        }),
    ]);

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const text = byId('text').value;
        const emotion = emotionSelect.value;
        const selectedEmotion = emotions.get(emotion);
        const strength = parseFloat(strengthInput.value);

        if (!text.trim()) return;

        btn.classList.add('loading');
        btn.disabled = true;
        resultContainer.classList.add('hidden');

        try {
            const payload = {
                text,
                voice_sample: voiceSelect.value || null,
                style_sample: styleSelect.value || null,
                emotion: emotion || null,
                emotion_vector: selectedEmotion ? selectedEmotion.vector : null,
                emotion_strength: emotion ? strength : null,
                quality: byId('quality').value,
                diffusion_steps: Number(byId('diffusion_steps').value),
                embedding_scale: Number(byId('embedding_scale').value),
                style_seed: Number(byId('style_seed').value || 0),
                speed: Number(byId('speed').value),
                sample_rate_hz: Number(byId('sample_rate_hz').value),
                max_tts_symbols: Number(byId('max_tts_symbols').value),
                no_tts_chunking: byId('no_tts_chunking').checked,
            };

            if (blendMode.value === 'raw') {
                payload.style_alpha = Number(byId('style_alpha').value);
                payload.style_beta = Number(byId('style_beta').value);
            } else {
                payload.speaker_reference_strength = Number(byId('speaker_reference_strength').value);
                payload.style_reference_strength = Number(byId('style_reference_strength').value);
            }

            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const textErr = await response.text();
                throw new Error(textErr);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            audioPlayer.src = url;
            resultContainer.classList.remove('hidden');
            audioPlayer.play().catch((error) => console.log('Auto-play prevented', error));
        } catch (err) {
            alert(`Synthesis Error: ${err.message}`);
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    });
}
