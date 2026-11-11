const recordButton = document.querySelector("#record");
const recordLabel = document.querySelector("#record-label");
const statusLine = document.querySelector("#status");
const fileInput = document.querySelector("#file");
const drop = document.querySelector("#drop");
const dropLabel = document.querySelector("#drop-label");
const promptField = document.querySelector("#prompt");
const preview = document.querySelector("#preview");
const previewCount = document.querySelector("#preview-count");
const speakersToggle = document.querySelector("#speakers");
const speakerOutputToggle = document.querySelector("#speaker-output");
const sharedPanel = document.querySelector("#shared-panel");
const addTabButton = document.querySelector("#add-tab");
const sharedList = document.querySelector("#shared-list");
const sharedStreams = [];
const copyButton = document.querySelector("#copy");
const gpuName = document.querySelector("#gpu-name");
const gpuReadout = document.querySelector("#gpu-readout");
const gpuMeter = document.querySelector("#gpu-meter");
const gpuMemoryMeter = document.querySelector("#gpu-memory-meter");
const gpuMemory = document.querySelector("#gpu-memory");
const library = document.querySelector("#library");
const libraryEmpty = document.querySelector("#library-empty");

const MAX_SECONDS = 10 * 60;
let recorder = null;
let timer = null;
let startedAt = 0;
let savedRecordings = [];
let selectedRecording = null;

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("is-error", isError);
}

function setTranscript(text, isEmpty = false) {
  preview.replaceChildren();
  const paragraph = document.createElement("p");
  paragraph.className = isEmpty ? "transcript is-empty" : "transcript";
  paragraph.textContent = text;
  preview.append(paragraph);
  previewCount.textContent = "";
  copyButton.disabled = isEmpty;
}

function formatClock(seconds) {
  if (seconds == null) return "";
  const total = Math.max(0, Number(seconds));
  const minutes = Math.floor(total / 60);
  const rest = (total % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${rest}`;
}

function appendTurn(list, turn) {
  const item = document.createElement("li");
  const speaker = turn.speaker == null ? "Speaker" : `Speaker ${turn.speaker}`;
  item.className = `turn turn-${turn.speaker == null ? 0 : Number(turn.speaker) % 4}`;
  const timing = turn.start == null ? "" : `${formatClock(turn.start)}–${formatClock(turn.end)}`;
  const header = document.createElement("header");
  const who = document.createElement("span");
  who.textContent = speaker;
  const when = document.createElement("span");
  when.textContent = timing;
  const spoken = document.createElement("p");
  spoken.textContent = turn.text;
  header.append(who, when);
  item.append(header, spoken);
  list.append(item);
}

function recordingText(item) {
  if (item.turns && item.turns.length) {
    return item.turns.map((turn) => {
      const speaker = turn.speaker == null ? "Speaker" : `Speaker ${turn.speaker}`;
      const timing = turn.start == null ? "" : `${formatClock(turn.start)}–${formatClock(turn.end)}`;
      return timing ? `${speaker} (${timing}) ${turn.text}` : `${speaker}: ${turn.text}`;
    }).join("\n");
  }
  return item.text || "";
}

function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function sharedName(stream, index) {
  return stream.getVideoTracks()[0]?.label || stream.getAudioTracks()[0]?.label || `Tab ${index + 1}`;
}

function stopStream(stream) {
  stream.getTracks().forEach((track) => track.stop());
}

function renderShared() {
  const busy = Boolean(recorder) || recordButton.disabled;
  sharedPanel.hidden = !speakerOutputToggle.checked;
  addTabButton.disabled = busy || !speakerOutputToggle.checked;
  sharedList.replaceChildren();
  sharedStreams.forEach((stream, index) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = sharedName(stream, index);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.disabled = busy;
    remove.addEventListener("click", () => {
      const at = sharedStreams.indexOf(stream);
      if (at >= 0) sharedStreams.splice(at, 1);
      stopStream(stream);
      renderShared();
    });
    item.append(name, remove);
    sharedList.append(item);
  });
}

async function captureSharedTab() {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
  if (stream.getAudioTracks().length === 0) {
    stopStream(stream);
    throw new Error("No shared audio was selected. Choose a tab and enable Share audio.");
  }
  const label = stream.getVideoTracks()[0]?.label;
  if (label && sharedStreams.some((existing) => existing.getVideoTracks()[0]?.label === label)) {
    stopStream(stream);
    throw new Error("That tab is already added.");
  }
  stream.getTracks().forEach((track) => {
    track.addEventListener("ended", () => {
      const at = sharedStreams.indexOf(stream);
      if (at >= 0) sharedStreams.splice(at, 1);
      renderShared();
    });
  });
  sharedStreams.push(stream);
  renderShared();
}

async function transcribe(blob, filename) {
  recordButton.disabled = true;
  fileInput.disabled = true;
  speakerOutputToggle.disabled = true;
  renderShared();
  setStatus("Transcribing…");
  const body = new FormData();
  body.append("file", blob, filename);
  body.append("prompt", promptField.value);
  body.append("speakers", speakersToggle.checked ? "1" : "0");
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `The server returned ${response.status}.`);
    }
    if (payload.recording) {
      selectedRecording = payload.recording.id;
      await loadLibrary();
    } else if (payload.turns && payload.turns.length) {
      setTranscript(payload.turns.map((turn) => turn.text).join("\n"));
    } else {
      setTranscript(payload.text || "(empty transcript)");
    }
    setStatus(payload.recording ? "Saved." : "Done.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    recordButton.disabled = false;
    fileInput.disabled = false;
    speakerOutputToggle.disabled = false;
    renderShared();
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function downsample(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = sum / (end - start);
  }
  return output;
}

function recordingLabel(elapsed) {
  const clock = formatElapsed(elapsed);
  const count = recorder.sources.length;
  const tabs = count === 1 ? "1 tab" : `${count} tabs`;
  if (count && recorder.stream) return `Recording you and ${tabs} ${clock}`;
  if (count) return `Recording ${tabs} ${clock}`;
  return `Recording ${clock}`;
}

async function startRecording() {
  const includeSpeakers = speakerOutputToggle.checked;
  if (includeSpeakers && sharedStreams.length === 0) await captureSharedTab();
  const sources = includeSpeakers ? sharedStreams.slice() : [];
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    if (sources.length === 0) throw error;
  }
  let context = null;
  let processor = null;
  const chunks = [];
  if (stream || sources.length) {
    context = new AudioContext();
    processor = context.createScriptProcessor(4096, 1, 1);
    const sink = context.createGain();
    sink.gain.value = 0;
    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    if (stream) context.createMediaStreamSource(stream).connect(processor);
    sources.forEach((source) => context.createMediaStreamSource(source).connect(processor));
    processor.connect(sink);
    sink.connect(context.destination);
  }
  recorder = { stream, sources, context, processor, chunks };
  startedAt = performance.now();
  fileInput.disabled = true;
  speakerOutputToggle.disabled = true;
  recordButton.classList.add("is-recording");
  recordLabel.textContent = "Stop";
  renderShared();
  setStatus(recordingLabel(0));
  timer = window.setInterval(() => {
    const elapsed = performance.now() - startedAt;
    setStatus(recordingLabel(elapsed));
    if (elapsed >= MAX_SECONDS * 1000) stopRecording();
  }, 250);
}

async function stopRecording() {
  if (!recorder) return;
  window.clearInterval(timer);
  const { stream, context, processor, chunks } = recorder;
  recorder = null;
  if (processor) processor.disconnect();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  recordButton.classList.remove("is-recording");
  recordLabel.textContent = "Record";
  fileInput.disabled = false;
  speakerOutputToggle.disabled = false;
  renderShared();

  let micSamples = null;
  if (context) {
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    chunks.forEach((chunk) => {
      merged.set(chunk, offset);
      offset += chunk.length;
    });
    micSamples = downsample(merged, context.sampleRate, 16000);
    await context.close();
  }

  if (!micSamples || micSamples.length === 0) {
    setStatus("The recording was empty.", true);
    return;
  }
  await transcribe(encodeWav(micSamples, 16000), "recording.wav");
}

recordButton.addEventListener("click", async () => {
  if (recorder) {
    await stopRecording();
    return;
  }
  try {
    await startRecording();
  } catch (error) {
    setStatus(error.message || "The microphone could not be opened.", true);
  }
});

async function takeFile(file) {
  if (!file) return;
  dropLabel.textContent = file.name;
  await transcribe(file, file.name);
}

fileInput.addEventListener("change", () => takeFile(fileInput.files[0]));

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("is-hot");
});

drop.addEventListener("dragleave", () => drop.classList.remove("is-hot"));

drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("is-hot");
  takeFile(event.dataTransfer.files[0]);
});

copyButton.addEventListener("click", async () => {
  const item = savedRecordings.find((entry) => entry.id === selectedRecording);
  const text = item ? recordingText(item) : preview.textContent;
  await navigator.clipboard.writeText(text);
  setStatus("Copied.");
});

speakersToggle.addEventListener("change", () => {
  promptField.disabled = speakersToggle.checked;
  promptField.labels[0].classList.toggle("is-disabled", speakersToggle.checked);
});

speakerOutputToggle.addEventListener("change", () => {
  if (!speakerOutputToggle.checked) {
    sharedStreams.splice(0).forEach(stopStream);
  }
  renderShared();
});

addTabButton.addEventListener("click", async () => {
  try {
    await captureSharedTab();
    const count = sharedStreams.length;
    setStatus(count === 1 ? "1 tab added." : `${count} tabs added.`);
  } catch (error) {
    if (error.name === "NotAllowedError") {
      setStatus("Tab sharing was cancelled.", true);
      return;
    }
    setStatus(error.message || "The tab could not be added.", true);
  }
});

function formatWhen(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function previewText(item) {
  const text = (item.text || "").replace(/\s+/g, " ").trim();
  if (text.length <= 90) return text || "Empty transcript";
  return `${text.slice(0, 90)}…`;
}

function markSelected() {
  const index = savedRecordings.findIndex((item) => item.id === selectedRecording);
  library.querySelectorAll("li").forEach((row) => {
    row.classList.toggle("is-selected", row.dataset.id === selectedRecording);
  });
  preview.querySelectorAll(".preview-item").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.id === selectedRecording);
  });
  const item = index >= 0 ? savedRecordings[index] : null;
  copyButton.disabled = !item || !recordingText(item);
  previewCount.textContent = index >= 0 ? `${index + 1} of ${savedRecordings.length}` : "";
}

function scrollPreviewTo(id) {
  const card = preview.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!card) return;
  preview.scrollTo({ top: card.offsetTop, behavior: "smooth" });
}

function selectRecording(item) {
  selectedRecording = item.id;
  markSelected();
  scrollPreviewTo(item.id);
}

function renderPreview() {
  preview.replaceChildren();
  if (!savedRecordings.length) {
    setTranscript("The transcript will appear here.", true);
    return;
  }
  savedRecordings.forEach((item) => {
    const card = document.createElement("article");
    card.className = "preview-item";
    card.dataset.id = item.id;
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = item.name || "Recording";
    const when = document.createElement("span");
    when.textContent = formatWhen(item.created);
    header.append(title, when);
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/recordings/${item.id}/audio`;
    audio.addEventListener("play", () => {
      preview.querySelectorAll("audio").forEach((other) => {
        if (other !== audio) other.pause();
      });
    });
    const fold = document.createElement("details");
    fold.className = "transcript-fold";
    const summary = document.createElement("summary");
    summary.textContent = "Transcript";
    fold.append(summary);
    if (item.turns && item.turns.length) {
      const turns = document.createElement("ol");
      turns.className = "speakers";
      item.turns.forEach((turn) => appendTurn(turns, turn));
      fold.append(turns);
    } else {
      const text = document.createElement("p");
      text.className = "transcript";
      text.textContent = item.text || "(empty transcript)";
      fold.append(text);
    }
    card.append(header, audio, fold);
    preview.append(card);
  });
  markSelected();
}

function renderLibrary() {
  library.replaceChildren();
  libraryEmpty.hidden = savedRecordings.length > 0;
  savedRecordings.forEach((item) => {
    const row = document.createElement("li");
    row.dataset.id = item.id;
    row.classList.toggle("is-selected", item.id === selectedRecording);
    const open = document.createElement("button");
    open.type = "button";
    open.className = "library-item";
    const title = document.createElement("strong");
    title.textContent = item.name || "Recording";
    const when = document.createElement("span");
    when.textContent = formatWhen(item.created);
    const preview = document.createElement("em");
    preview.textContent = previewText(item);
    open.append(title, when, preview);
    open.addEventListener("click", () => selectRecording(item));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      const response = await fetch(`/api/recordings/${item.id}`, { method: "DELETE" });
      if (!response.ok) {
        setStatus("The recording could not be deleted.", true);
        return;
      }
      if (selectedRecording === item.id) {
        const remaining = savedRecordings.filter((entry) => entry.id !== item.id);
        selectedRecording = remaining.length ? remaining[0].id : null;
      }
      await loadLibrary();
    });
    row.append(open, remove);
    library.append(row);
  });
}

async function loadLibrary() {
  const response = await fetch("/api/recordings");
  if (!response.ok) return;
  savedRecordings = await response.json();
  if (!savedRecordings.some((item) => item.id === selectedRecording)) {
    selectedRecording = savedRecordings[0] ? savedRecordings[0].id : null;
  }
  renderLibrary();
  renderPreview();
  if (selectedRecording) scrollPreviewTo(selectedRecording);
}

preview.addEventListener("scroll", () => {
  const top = preview.scrollTop;
  let nearest = null;
  let distance = Infinity;
  preview.querySelectorAll(".preview-item").forEach((card) => {
    const delta = Math.abs(card.offsetTop - top);
    if (delta < distance) {
      nearest = card;
      distance = delta;
    }
  });
  if (!nearest || nearest.dataset.id === selectedRecording) return;
  selectedRecording = nearest.dataset.id;
  markSelected();
}, { passive: true });

function formatStat(value, suffix) {
  return value == null ? "—" : `${Math.round(value)}${suffix}`;
}

async function refreshGpu() {
  try {
    const response = await fetch("/api/stats");
    if (!response.ok) return;
    const stats = await response.json();
    gpuName.textContent = stats.name || "GPU";
    const util = stats.utilization == null ? 0 : stats.utilization;
    gpuMeter.style.width = `${Math.max(0, Math.min(100, util))}%`;
    const total = stats.memory_total_gb || 0;
    const used = stats.memory_used_gb || 0;
    const memoryPercent = total ? (used / total) * 100 : 0;
    gpuMemoryMeter.style.width = `${Math.max(0, Math.min(100, memoryPercent))}%`;
    gpuReadout.textContent = [
      formatStat(stats.utilization, "%"),
      formatStat(stats.temperature_c, "°C"),
      stats.power_watts == null ? "—" : `${stats.power_watts.toFixed(0)} W`,
    ].join(" · ");
    if (total) {
      gpuMemory.textContent = `${used.toFixed(0)} GB of ${total.toFixed(0)} GB shared memory in use`;
    }
  } catch (_error) {
    gpuReadout.textContent = "GPU stats unavailable";
  }
}

refreshGpu();
window.setInterval(refreshGpu, 1000);
loadLibrary();
setTranscript("The transcript will appear here.", true);
