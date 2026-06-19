document.addEventListener('DOMContentLoaded', async () => {
    const emotionSelect = document.getElementById('emotion');
    const strengthInput = document.getElementById('emotion_strength');
    const strengthVal = document.getElementById('strength-val');
    const emotionDetail = document.getElementById('emotion-detail');
    const strengthPresets = document.getElementById('strength-presets');
    const form = document.getElementById('synth-form');
    const btn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const audioPlayer = document.getElementById('audio-player');
    const emotions = new Map();

    const setStrength = (value) => {
        strengthInput.value = value.toFixed(2);
        strengthVal.textContent = value.toFixed(2);
    };

    const formatEmotionName = (name) => name.charAt(0).toUpperCase() + name.slice(1);

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

    try {
        const res = await fetch('/api/emotions');
        const data = await res.json();

        if (data.error) {
            emotionDetail.textContent = data.error;
            return;
        }

        if (data.emotions && data.emotions.length > 0) {
            data.emotions.forEach(em => {
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
    } catch (err) {
        console.error("Failed to load emotions", err);
        emotionDetail.textContent = 'Failed to load emotion signatures';
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = document.getElementById('text').value;
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
                emotion: emotion || null,
                emotion_vector: selectedEmotion ? selectedEmotion.vector : null,
                emotion_strength: emotion ? strength : null
            };

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
            
            // Auto play
            audioPlayer.play().catch(e => console.log("Auto-play prevented", e));

        } catch (err) {
            alert(`Synthesis Error: ${err.message}`);
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    });
});
