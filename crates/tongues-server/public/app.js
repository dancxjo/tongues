document.addEventListener('DOMContentLoaded', async () => {
    const emotionSelect = document.getElementById('emotion');
    const strengthInput = document.getElementById('emotion_strength');
    const strengthVal = document.getElementById('strength-val');
    const form = document.getElementById('synth-form');
    const btn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const audioPlayer = document.getElementById('audio-player');

    // Update strength value display
    strengthInput.addEventListener('input', (e) => {
        strengthVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Fetch emotions
    try {
        const res = await fetch('/api/emotions');
        const data = await res.json();
        
        if (data.emotions && data.emotions.length > 0) {
            data.emotions.forEach(em => {
                const option = document.createElement('option');
                option.value = em;
                option.textContent = em.charAt(0).toUpperCase() + em.slice(1);
                emotionSelect.appendChild(option);
            });
        }
    } catch (err) {
        console.error("Failed to load emotions", err);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = document.getElementById('text').value;
        const emotion = emotionSelect.value;
        const strength = parseFloat(strengthInput.value);

        if (!text.trim()) return;

        btn.classList.add('loading');
        btn.disabled = true;
        resultContainer.classList.add('hidden');

        try {
            const payload = {
                text,
                emotion: emotion || null,
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
