class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.cursor = 0;
    this.held = [];
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let index = 0; index < channel.length; index += 1) {
      this.cursor += 1;
      if (this.cursor < this.ratio) continue;
      this.cursor -= this.ratio;
      const sample = Math.max(-1, Math.min(1, channel[index]));
      this.held.push(sample < 0 ? sample * 0x8000 : sample * 0x7fff);
    }
    if (this.held.length >= 1600) {
      const pcm = new Int16Array(this.held);
      this.held = [];
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
