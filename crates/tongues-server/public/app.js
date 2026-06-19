document.addEventListener('DOMContentLoaded', async () => {
    const byId = (id) => document.getElementById(id);

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

    strengthInput.addEventListener('input', (e) => {
        setStrength(parseFloat(e.target.value));
    });

    emotionSelect.addEventListener('change', () => {
        const selected = emotions.get(emotionSelect.value);
        if (selected && selected.recommended_strength) {
            setStrength(selected.recommended_strength.normal || 0.65);
        }
        updateEmotionDetail();
    });

    strengthPresets.addEventListener('click', (e) => {
        const button = e.target.closest('button[data-preset]');
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

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

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
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
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
});
