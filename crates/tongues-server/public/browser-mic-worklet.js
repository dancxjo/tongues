class TonguesBrowserMicCapture extends AudioWorkletProcessor {
    process(inputs) {
        const channels = inputs[0];
        if (!channels?.length || !channels[0]?.length) return true;
        const mono = new Float32Array(channels[0].length);
        for (const channel of channels) {
            for (let index = 0; index < mono.length; index += 1) {
                mono[index] += channel[index] / channels.length;
            }
        }
        this.port.postMessage(mono.buffer, [mono.buffer]);
        return true;
    }
}

registerProcessor('tongues-browser-mic-capture', TonguesBrowserMicCapture);
