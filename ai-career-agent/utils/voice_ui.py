"""Browser-native voice controls for the AI Career Chat only."""
import json
import re
import unicodedata

import streamlit as st


_DEFAULT_SETTINGS = {
    "voice_uri": "",
    "rate": 1.0,
    "pitch": 1.0,
    "volume": 1.0,
}


def _json_to_speech(value) -> str:
    """Turn a JSON value into short natural phrases instead of reading syntax."""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            label = re.sub(r"[_-]+", " ", str(key)).strip()
            detail = _json_to_speech(item)
            if detail:
                parts.append(f"{label}: {detail}" if label else detail)
        return ". ".join(parts)
    if isinstance(value, list):
        return _join_naturally([_json_to_speech(item) for item in value if item is not None])
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return ""
    return str(value)


def _join_naturally(items: list[str]) -> str:
    items = [item.strip().rstrip(".,;:") for item in items if item and item.strip()]
    if len(items) < 2:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _replace_json_values(text: str) -> str:
    """Replace complete JSON objects/arrays (including inline ones) with prose."""
    decoder = json.JSONDecoder()
    out, cursor = [], 0
    while cursor < len(text):
        candidate = next((i for i in range(cursor, len(text)) if text[i] in "{["), None)
        if candidate is None:
            out.append(text[cursor:])
            break
        out.append(text[cursor:candidate])
        try:
            value, end = decoder.raw_decode(text, candidate)
        except json.JSONDecodeError:
            out.append(text[candidate])
            cursor = candidate + 1
            continue
        out.append(_json_to_speech(value))
        cursor = end
    return "".join(out)


def clean_text_for_speech(text: str, *, include_urls: bool = False) -> str:
    """Convert formatted AI text to natural speech without changing on-screen text.

    Markdown lists are spoken as sentences, fenced JSON is summarized as prose,
    URLs are omitted by default, and punctuation is retained for natural pauses.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines = []
    code_lines = []
    in_code = False
    code_is_json = False
    table_headers = None
    for raw_line in lines:
        fence = re.match(r"^\s*```\s*([\w+-]*)", raw_line)
        if fence:
            if not in_code:
                in_code = True
                code_is_json = fence.group(1).lower() in {"json", "jsonc"}
                code_lines = []
            else:
                block = "\n".join(code_lines).strip()
                if code_is_json or block.startswith(("{", "[")):
                    try:
                        cleaned_lines.append(_json_to_speech(json.loads(block)))
                    except (json.JSONDecodeError, TypeError):
                        pass
                # Non-JSON code is intentionally not read aloud.
                in_code = False
                code_lines = []
            continue
        if in_code:
            code_lines.append(raw_line)
            continue

        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        is_ordered_item = bool(re.match(r"^\s*\d+[.)]\s+", raw_line))
        is_bullet_item = bool(re.match(r"^\s*[-+*•▪◦‣]\s+", raw_line))
        # Markdown table separator rows contain no user-facing content.
        if "|" in line and re.fullmatch(r"[\s|:=-]+", line):
            continue
        # Convert tables to labeled phrases rather than reading their grid syntax.
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if table_headers is None:
                table_headers = cells
                continue
            line = ", ".join(
                f"{table_headers[index]}: {cell}" if index < len(table_headers) else cell
                for index, cell in enumerate(cells)
            )
        else:
            table_headers = None

        line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"<https?://[^>]+>", " ", line, flags=re.IGNORECASE)
        if not include_urls:
            had_url = bool(re.search(r"\b(?:https?://|www\.)\S+", line, flags=re.IGNORECASE))
            line = re.sub(r"\b(?:visit|see|read|open|click)\s+(?=(?:https?://|www\.))", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\b(?:https?://|www\.)\S+", " ", line, flags=re.IGNORECASE)
            if had_url:
                # Avoid leaving a dangling instruction such as "Visit." after URL removal.
                line = re.sub(r"\b(?:visit|see|read|open|click|link)\s*[.!?]?\s*$", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"<[^>]+>", " ", line)
        # Emojis can precede Markdown headings, so allow decorative symbols before #.
        line = re.sub(r"^\s*[^\w#]*#{1,6}\s*", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = _replace_json_values(line)

        # Preserve the semantic text in emphasis/code marks, dropping only markup.
        line = re.sub(r"(\*\*|__|~~|`{1,3})", "", line)
        line = line.replace("*", "").replace("_", " ")
        line = line.replace("/", " or ").replace("|", ", ").replace("=", " is ")
        line = re.sub(r'[{}\[\]()<>"“”]', " ", line)
        # Remove pictographs/decorative symbols without removing ordinary letters.
        line = "".join(
            char for char in line
            if unicodedata.category(char) != "So" and char not in "\u200d\ufe0f"
        )
        line = re.sub(r"\s+", " ", line).strip()
        if is_bullet_item and not is_ordered_item:
            line = re.sub(r"^\s*[-+*•▪◦‣]\s+", "", line)
            line = "• " + line
        cleaned_lines.append(line)

    result = []
    numbered_title = None
    numbered_items = []

    def flush_numbered():
        nonlocal numbered_title, numbered_items
        if numbered_title:
            ordinal = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
            lead = ordinal.get(numbered_title[0], f"Item {numbered_title[0]}")
            phrase = f"{lead}, {numbered_title[1]}"
            if numbered_items:
                phrase += f". You should focus on {_join_naturally(numbered_items)}"
            result.append(phrase + ".")
        numbered_title, numbered_items = None, []

    for raw_line in cleaned_lines:
        line = raw_line.strip()
        if not line:
            flush_numbered()
            continue
        numbered = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if numbered:
            flush_numbered()
            numbered_title = (int(numbered.group(1)), numbered.group(2).strip(" .,:;"))
            continue
        bullet = re.match(r"^(?:[-+•▪◦‣]|\d+[.)])\s+(.+)$", line)
        if bullet:
            item = bullet.group(1).strip(" .,:;")
            if numbered_title:
                numbered_items.append(item)
            elif item:
                result.append(item.rstrip(".!?") + ".")
            continue
        flush_numbered()
        line = re.sub(r"\s*([,;:])\s*", r"\1 ", line)
        line = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", line)
        line = re.sub(r"\s+([,.!?;:])", r"\1", line)
        line = re.sub(r"\s+", " ", line).strip(" \t,;:-")
        if line:
            result.append(line.rstrip(".!?") + ".")
    flush_numbered()

    speech = " ".join(part for part in result if part)
    speech = re.sub(r"\s+", " ", speech)
    speech = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", speech)
    return speech.strip()

_TOOLBAR_HTML = """
<section class="voice-panel" aria-label="Voice controls for AI Career Chat">
  <div class="voice-heading"><strong>Speech settings</strong><span>Choose a voice available in this browser or device.</span></div>
  <div class="voice-grid">
    <label>Voice<select id="voice-choice" aria-label="Voice"></select></label>
    <label class="range-control">Speed <output id="rate-value">1.00×</output><input id="voice-rate" aria-label="Speed" type="range" min="0.65" max="1.50" step="0.05"></label>
    <label class="range-control">Pitch <output id="pitch-value">1.00</output><input id="voice-pitch" aria-label="Pitch" type="range" min="0.50" max="1.80" step="0.05"></label>
    <label class="range-control">Volume <output id="volume-value">100%</output><input id="voice-volume" aria-label="Volume" type="range" min="0" max="1" step="0.05"></label>
  </div>
  <p class="voice-note">Voice choices depend on your browser and operating system.</p>
</section>
"""

_TOOLBAR_CSS = """
.voice-panel { border: 1px solid var(--st-border-color, #d8dee8); border-radius: 12px; padding: 14px 16px; margin: 4px 0 12px; background: var(--st-secondary-background-color, #f7f9fc); color: var(--st-text-color, #1f2937); font: inherit; }
.voice-heading { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:12px; }
.voice-heading strong { font-size: .98rem; }
.voice-heading span,.voice-note { color: var(--st-text-color, #536174); opacity:.78; font-size:.78rem; }
.voice-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px 14px; }
.voice-grid label { display:flex; flex-direction:column; gap:5px; font-size:.82rem; font-weight:600; }
.voice-grid select { width:100%; border:1px solid var(--st-border-color,#cbd5e1); border-radius:7px; padding:7px 9px; background:var(--st-background-color,#fff); color:inherit; font:inherit; }
.range-control { display:grid !important; grid-template-columns:1fr auto; align-items:center; }
.range-control input { grid-column:1 / -1; width:100%; accent-color:var(--st-primary-color,#2563eb); }
.range-control output { font-variant-numeric:tabular-nums; font-weight:500; }
.voice-actions { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-top:12px; }
.voice-actions button,.speech-actions button { border:1px solid var(--st-border-color,#cbd5e1); border-radius:8px; padding:7px 12px; background:var(--st-background-color,#fff); color:inherit; font:inherit; cursor:pointer; }
.voice-actions button:hover,.speech-actions button:hover { border-color:var(--st-primary-color,#2563eb); }
.voice-actions button:disabled { opacity:.55; cursor:not-allowed; }
.auto-read { display:flex; align-items:center; gap:7px; font-size:.84rem; }
.voice-note { margin:9px 0 2px; }
#voice-status { min-height:1.2em; margin:5px 0 0; font-size:.84rem; }
@media(max-width:560px) { .voice-heading { align-items:flex-start; flex-direction:column; gap:3px; } }
"""

_TOOLBAR_JS = r"""
export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement;
  if (!root.__careerVoice) {
    root.innerHTML = `__HTML__`;
    const ui = {
      voice: root.querySelector('#voice-choice'),
      rate: root.querySelector('#voice-rate'), pitch: root.querySelector('#voice-pitch'),
      volume: root.querySelector('#voice-volume')
    };
    const controller = root.__careerVoice = { ui, data, setStateValue, voices: [] };
    const settingsFromUI = () => ({
      voice_uri: ui.voice.value || '',
      rate: Number(ui.rate.value), pitch: Number(ui.pitch.value),
      volume: Number(ui.volume.value)
    });
    const updateState = () => {
      controller.setStateValue('settings', settingsFromUI());
      root.querySelector('#rate-value').value = `${Number(ui.rate.value).toFixed(2)}×`;
      root.querySelector('#pitch-value').value = Number(ui.pitch.value).toFixed(2);
      root.querySelector('#volume-value').value = `${Math.round(Number(ui.volume.value) * 100)}%`;
    };
    const loadVoices = () => {
      const prior = ui.voice.value || controller.data.settings.voice_uri || '';
      controller.voices = window.speechSynthesis?.getVoices?.() || [];
      ui.voice.replaceChildren();
      controller.voices.forEach((voice) => {
        const option = document.createElement('option'); option.value = voice.voiceURI;
        option.textContent = `${voice.name} (${voice.lang})`; ui.voice.append(option);
      });
      const preferred = controller.voices.find(v => v.voiceURI === prior)
        || controller.voices.find(v => v.default)
        || controller.voices.find(v => /^en/i.test(v.lang) && /natural|neural|premium/i.test(v.name))
        || controller.voices.find(v => /^en/i.test(v.lang))
        || controller.voices[0];
      if (!prior && preferred) ui.voice.value = preferred.voiceURI;
      else ui.voice.value = controller.voices.some(v => v.voiceURI === prior) ? prior : (preferred?.voiceURI || '');
      updateState();
    };
    controller.loadVoices = loadVoices;
    ui.voice.addEventListener('change', updateState);
    [ui.rate, ui.pitch, ui.volume].forEach((input) => {
      input.addEventListener('input', () => {
        root.querySelector('#rate-value').value = `${Number(ui.rate.value).toFixed(2)}×`;
        root.querySelector('#pitch-value').value = Number(ui.pitch.value).toFixed(2);
        root.querySelector('#volume-value').value = `${Math.round(Number(ui.volume.value) * 100)}%`;
      });
      input.addEventListener('change', updateState);
    });
    loadVoices();
    window.speechSynthesis?.addEventListener?.('voiceschanged', loadVoices);
  }
  const controller = root.__careerVoice;
  controller.data = data;
  controller.setStateValue = setStateValue;
  const settings = data.settings || {};
  const ui = controller.ui;
  if (ui.rate.value !== String(settings.rate ?? 1)) ui.rate.value = String(settings.rate ?? 1);
  if (ui.pitch.value !== String(settings.pitch ?? 1)) ui.pitch.value = String(settings.pitch ?? 1);
  if (ui.volume.value !== String(settings.volume ?? 1)) ui.volume.value = String(settings.volume ?? 1);
  const chosen = settings.voice_uri || '';
  if (Array.from(ui.voice.options).some((option) => option.value === chosen)) ui.voice.value = chosen;
  root.querySelector('#rate-value').value = `${Number(ui.rate.value).toFixed(2)}×`;
  root.querySelector('#pitch-value').value = Number(ui.pitch.value).toFixed(2);
  root.querySelector('#volume-value').value = `${Math.round(Number(ui.volume.value) * 100)}%`;
}
""".replace("`__HTML__`", repr(_TOOLBAR_HTML))

_INPUT_HTML = """
<div class="voice-input" aria-label="Speak your message">
  <button type="button" id="input-mic" aria-label="Speak your message" title="Speak your message">🎤</button>
  <span id="input-status" role="status" aria-live="polite">Speak instead of typing, if you prefer.</span>
</div>
"""

_INPUT_CSS = """
.voice-input { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0; min-height:52px; font:inherit; color:var(--st-text-color,#1f2937); }
.voice-input button { width:40px; height:40px; flex:0 0 40px; border:1px solid var(--st-border-color,#cbd5e1); border-radius:50%; background:var(--st-background-color,#fff); color:inherit; font-size:1.15rem; cursor:pointer; }
.voice-input button:hover { border-color:var(--st-primary-color,#2563eb); }
.voice-input button:disabled { opacity:.55; cursor:not-allowed; }
.voice-input span { max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.66rem; line-height:1.1; opacity:.78; }
"""

_INPUT_JS = r"""
export default function(component) {
  const { parentElement, setStateValue } = component;
  const root = parentElement;
  if (!root.__careerVoiceInput) {
    root.innerHTML = `__INPUT_HTML__`;
    const mic = root.querySelector('#input-mic');
    const status = root.querySelector('#input-status');
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    if (!Recognition) {
      mic.disabled = true;
      status.textContent = 'Voice input is unsupported here. You can still type and send.';
    }
    mic.addEventListener('click', () => {
      if (!Recognition) return;
      if (recognition) { recognition.stop(); recognition = null; mic.textContent = '🎤'; status.textContent = 'Listening stopped.'; return; }
      const beginRecognition = async () => {
        recognition = new Recognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.lang = document.documentElement.lang || navigator.language || 'en-US';
        mic.textContent = '⏹';
        status.textContent = 'Listening...';
        // Use browser on-device recognition when available; this avoids sending
        // microphone audio to a remote speech service. Older browsers fall back
        // to their native Web Speech implementation.
        if ('processLocally' in recognition && typeof Recognition.available === 'function') {
          try {
            let availability = await Recognition.available({ langs: [recognition.lang], processLocally: true });
            if (availability !== 'available' && typeof Recognition.install === 'function'
                && (availability === 'downloadable' || availability === 'downloading')) {
              status.textContent = 'Preparing on-device speech recognition…';
              const installed = await Recognition.install({ langs: [recognition.lang], processLocally: true });
              if (installed) availability = 'available';
            }
            if (availability === 'available') recognition.processLocally = true;
          } catch (error) {
            // Keep browser speech recognition as a compatibility fallback.
          }
        }
        recognition.onresult = (event) => {
          const text = Array.from(event.results || []).map((row) => row[0]?.transcript || '').join(' ').trim();
          if (text) {
            setStateValue('transcript_event', { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, text });
            status.textContent = 'Transcript ready. Review or edit it, then Send.';
          } else status.textContent = 'No speech detected. Click the microphone to try again.';
        };
        recognition.onerror = (event) => {
          const error = event.error || 'unknown';
          if (error === 'not-allowed' || error === 'service-not-allowed') status.textContent = 'Microphone access was denied. Allow it in browser settings, or type your message.';
          else if (error === 'audio-capture') status.textContent = 'No microphone was found. Connect one, or type your message.';
          else if (error === 'no-speech') status.textContent = 'No speech detected. Click the microphone to try again.';
          else if (error === 'network') status.textContent = 'The browser speech service could not be reached. You can still type and send.';
          else status.textContent = `Voice input could not start (${error}). You can still type and send.`;
        };
        recognition.onend = () => { recognition = null; mic.textContent = '🎤'; };
        recognition.start();
      };
      try {
        beginRecognition().catch(() => {
          recognition = null; mic.textContent = '🎤';
          status.textContent = 'Voice input could not start. Check microphone permission, or type your message.';
        });
      } catch (error) {
        recognition = null; mic.textContent = '🎤';
        status.textContent = 'Voice input could not start. Check microphone permission, or type your message.';
      }
    });
    root.__careerVoiceInput = true;
  }
}
""".replace("`__INPUT_HTML__`", repr(_INPUT_HTML))

_COMPOSER_HTML = """
<section class="chat-composer" aria-label="Career chat message composer">
  <div class="composer-shell">
    <div class="composer-field">
      <textarea id="composer-text" rows="2" aria-label="Ask about your resume or career" placeholder="Ask your career question..."></textarea>
      <div class="composer-actions">
        <button id="composer-send" type="button" aria-label="Send message" title="Send message">Send</button>
        <button id="composer-mic" type="button" aria-label="Speak your message" title="Speak your message">🎤</button>
      </div>
    </div>
    <span id="composer-status" role="status" aria-live="polite">Type a message and press Enter, or click the microphone.</span>
  </div>
</section>
"""

_COMPOSER_CSS = """
:host { display:block; width:100%; min-width:0; }
.chat-composer { width:100%; max-width:none; min-width:0; margin:0 auto; padding:3px 0 5px; font:inherit; color:var(--st-text-color,#1f2937); }
.composer-shell { width:100%; min-width:0; }
.composer-field { position:relative; width:100%; min-width:0; min-height:74px; border:1px solid var(--st-border-color,#cbd5e1); border-radius:20px; background:var(--st-background-color,#fff); box-shadow:0 3px 12px #1112; transition:border-color .15s ease,box-shadow .15s ease; }
.composer-field:focus-within { border-color:var(--st-primary-color,#2563eb); box-shadow:0 0 0 2px color-mix(in srgb,var(--st-primary-color,#2563eb) 18%,transparent); }
.composer-field textarea { display:block; width:100%; height:74px; min-height:74px; max-height:142px; overflow-y:hidden; resize:none; box-sizing:border-box; padding:15px 68px 14px 20px; border:0; outline:0; border-radius:20px; background:transparent; color:inherit; font:inherit; font-size:1rem; line-height:1.5; }
.composer-field textarea::placeholder { color:var(--st-text-color,#64748b); opacity:.65; }
.composer-actions { position:absolute; top:50%; right:10px; display:flex; align-items:center; gap:6px; transform:translateY(-50%); }
.composer-actions button { display:grid; place-items:center; flex:0 0 auto; min-width:40px; height:40px; padding:0 9px; border:0; border-radius:9px; color:var(--st-text-color,#1f2937); background:transparent; font:inherit; font-size:.9rem; cursor:pointer; transition:background .15s ease,transform .15s ease; }
.composer-actions button:hover { background:var(--st-secondary-background-color,#eef2f7); }
.composer-actions #composer-send { color:var(--st-primary-color,#2563eb); font-weight:650; }
.composer-actions #composer-send:disabled { opacity:.5; cursor:not-allowed; }
.composer-actions #composer-mic { min-width:40px; width:40px; padding:0; border-radius:50%; font-size:1.08rem; }
.composer-actions #composer-mic.recording { background:#fee2e2; color:#b91c1c; animation:pulse 1.2s infinite; }
#composer-status { display:block; min-height:15px; margin:4px 4px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--st-text-color,#64748b); opacity:.72; font-size:.72rem; line-height:1.25; }
@keyframes pulse { 50% { box-shadow:0 0 0 5px #ef444433; } }
@media(prefers-reduced-motion:reduce) { .composer-field #composer-mic.recording { animation:none; } }
@media(max-width:640px) { .chat-composer { width:100%; } .composer-field { border-radius:16px; } .composer-field textarea { padding-left:15px; padding-right:62px; border-radius:16px; } .composer-actions { right:7px; gap:3px; } .composer-actions button { flex-basis:36px; width:36px; height:36px; } #composer-status { margin-left:3px; } }
"""

_COMPOSER_JS = r"""
export default function(component) {
  const { parentElement: root, data, setTriggerValue } = component;
  if (!root.__careerChatComposer) {
    const field = root.querySelector('#composer-text');
    const mic = root.querySelector('#composer-mic');
    const send = root.querySelector('#composer-send');
    const status = root.querySelector('#composer-status');
    let recorder = null;
    let stream = null;
    let chunks = [];
    let audioContext = null;
    let analyser = null;
    let meterFrame = 0;
    let speechMs = 0;
    let priorMeterAt = 0;
    let silenceSince = 0;
    let recordingStartedAt = 0;
    let submitId = 0;
    let discardCaptureOnStop = false;
    const resizeField = () => {
      field.style.height = '74px';
      const nextHeight = Math.min(field.scrollHeight, 142);
      field.style.height = `${nextHeight}px`;
      field.style.overflowY = field.scrollHeight > 142 ? 'auto' : 'hidden';
    };
    field.addEventListener('input', () => {
      resizeField();
      if (root.__careerChatComposer?.isTranscribing) {
        root.__careerChatComposer.transcriptionEdited = true;
      }
      send.disabled = !field.value.trim() || Boolean(root.__careerChatComposer?.isSubmitting);
    });
    const submit = () => {
      if (send.disabled) return;
      const text = field.value.trim();
      if (!text) { status.textContent = 'Type or speak a message first.'; field.focus(); return; }
      window.speechSynthesis?.cancel();
      if (recorder?.state === 'recording') {
        discardCaptureOnStop = true;
        stopCapture();
      }
      submitId += 1;
      if (root.__careerChatComposer) root.__careerChatComposer.isSubmitting = true;
      setTriggerValue('submit_event', { id: `${data.instance_id || 'legacy'}:${Date.now()}:${submitId}`, text });
      if (root.__careerChatComposer?.isTranscribing) {
        root.__careerChatComposer.typedSubmitDuringTranscription = true;
        root.__careerChatComposer.ignoreTranscriptId = root.__careerChatComposer.pendingTranscriptId;
      }
      send.disabled = true;
      status.textContent = 'Thinking…';
    };
    send.addEventListener('click', submit);
    field.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit(); }
    });
    const stopMeter = () => {
      if (meterFrame) cancelAnimationFrame(meterFrame);
      meterFrame = 0;
      try { analyser?.disconnect(); } catch (error) {}
      analyser = null;
      if (audioContext && audioContext.state !== 'closed') audioContext.close().catch(() => {});
      audioContext = null;
    };
    const stopCapture = () => {
      if (recorder?.state === 'recording') recorder.stop();
    };
    const beginMeter = () => {
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        audioContext = new AudioContextClass();
        analyser = audioContext.createAnalyser(); analyser.fftSize = 1024;
        audioContext.createMediaStreamSource(stream).connect(analyser);
        const samples = new Uint8Array(analyser.fftSize);
        const sample = (now) => {
          if (!analyser || recorder?.state !== 'recording') return;
          analyser.getByteTimeDomainData(samples);
          let energy = 0;
          for (const value of samples) { const centered = (value - 128) / 128; energy += centered * centered; }
          const rms = Math.sqrt(energy / samples.length);
          if (priorMeterAt && rms > 0.012) {
            speechMs += Math.min(80, now - priorMeterAt);
            silenceSince = 0;
          } else if (speechMs >= 300) {
            if (!silenceSince) silenceSince = now;
            if (now - silenceSince >= 1200) {
              status.textContent = 'Speech captured. Transcribing…';
              stopCapture();
              return;
            }
          } else if (recordingStartedAt && now - recordingStartedAt >= 8000) {
            stopCapture();
            return;
          }
          priorMeterAt = now;
          meterFrame = requestAnimationFrame(sample);
        };
        meterFrame = requestAnimationFrame(sample);
      } catch (error) { stopMeter(); }
    };
    mic.addEventListener('click', async () => {
      if (recorder?.state === 'recording') { stopCapture(); status.textContent = 'Finishing recording…'; return; }
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        status.textContent = 'Microphone access is unavailable. Please allow microphone access in your browser.'; return;
      }
      try {
        window.speechSynthesis?.cancel();
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        if (root.__careerChatComposer) root.__careerChatComposer.recordingDraftAtStart = field.value;
        chunks = []; speechMs = 0; priorMeterAt = 0; silenceSince = 0; recordingStartedAt = performance.now();
        const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
        recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
        recorder.ondataavailable = event => { if (event.data?.size) chunks.push(event.data); };
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: recorder?.mimeType || 'audio/webm' });
          stopMeter(); stream?.getTracks().forEach(track => track.stop()); stream = null;
          mic.classList.remove('recording'); mic.textContent = '🎤'; mic.disabled = false;
          recorder = null;
          if (root.__careerChatComposer) root.__careerChatComposer.isRecording = false;
          if (discardCaptureOnStop) {
            discardCaptureOnStop = false;
            chunks = [];
            if (root.__careerChatComposer) root.__careerChatComposer.isTranscribing = false;
            return;
          }
          if (root.__careerChatComposer) {
            root.__careerChatComposer.isTranscribing = true;
            root.__careerChatComposer.transcriptionEdited = (
              field.value !== (root.__careerChatComposer.recordingDraftAtStart || '')
            );
            root.__careerChatComposer.typedSubmitDuringTranscription = false;
          }
          if (!blob.size || speechMs < 300) {
            if (root.__careerChatComposer) root.__careerChatComposer.isTranscribing = false;
            chunks = []; status.textContent = 'No speech detected. Your input is still empty; click the microphone to try again.'; return;
          }
          status.textContent = 'Transcribing… You can still type and send.'; mic.disabled = true;
          const transcriptId = `${data.instance_id || 'legacy'}:${Date.now()}:${Math.random().toString(36).slice(2)}`;
          if (root.__careerChatComposer) root.__careerChatComposer.pendingTranscriptId = transcriptId;
          const reader = new FileReader();
          reader.onload = () => {
            const audio_b64 = String(reader.result).split(',')[1] || '';
            if (!audio_b64) {
              if (root.__careerChatComposer) root.__careerChatComposer.isTranscribing = false;
              mic.disabled = false; send.disabled = false; status.textContent = 'Could not read the recording. Your input is still empty.'; return;
            }
            const event = {
              id: transcriptId,
              text: '', audio_b64, mime_type: blob.type || 'audio/webm'
            };
            if (root.__careerChatComposer) root.__careerChatComposer.pendingTranscriptId = event.id;
            setTriggerValue('transcript_event', event);
          };
          reader.onerror = () => {
            if (root.__careerChatComposer) root.__careerChatComposer.isTranscribing = false;
            mic.disabled = false; send.disabled = false; status.textContent = 'Could not read the recording. Your input is still empty.';
          };
          reader.readAsDataURL(blob);
          chunks = [];
        };
        recorder.onerror = () => {
          stopMeter(); stream?.getTracks().forEach(track => track.stop()); stream = null; recorder = null;
          if (root.__careerChatComposer) {
            root.__careerChatComposer.isRecording = false;
            root.__careerChatComposer.isTranscribing = false;
          }
          mic.disabled = false; send.disabled = false; mic.classList.remove('recording'); mic.textContent = '🎤';
          if (discardCaptureOnStop) { discardCaptureOnStop = false; return; }
          status.textContent = 'Microphone recording failed. Please allow microphone access in your browser.';
        };
        recorder.start(250); beginMeter();
        if (root.__careerChatComposer) root.__careerChatComposer.isRecording = true;
        mic.classList.add('recording'); mic.textContent = '⏹'; status.textContent = 'Recording… speak, then click the microphone to stop.';
        window.clearTimeout(root.__recordLimit);
        root.__recordLimit = window.setTimeout(stopCapture, 60000);
      } catch (error) {
        stopMeter(); stream?.getTracks().forEach(track => track.stop()); stream = null; recorder = null;
        if (root.__careerChatComposer) {
          root.__careerChatComposer.isRecording = false;
          root.__careerChatComposer.isTranscribing = false;
        }
        mic.classList.remove('recording'); mic.textContent = '🎤'; mic.disabled = false; send.disabled = false;
        const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError';
        status.textContent = denied
          ? 'Microphone access is unavailable. Please allow microphone access in your browser.'
          : 'Microphone access is unavailable. Please allow microphone access in your browser.';
      }
    });
    root.__careerChatComposer = {
      field, mic, send, status, resizeField, lastClearId: '', lastFailedId: '', lastTranscriptKey: '',
      lastTranscriptText: '', isRecording: false, isTranscribing: false, isSubmitting: false, transcriptionEdited: false,
      typedSubmitDuringTranscription: false, pendingTranscriptId: '', ignoreTranscriptId: ''
    };
    send.disabled = !field.value.trim();
  }
  const ui = root.__careerChatComposer;
  // Python-to-browser data is only allowed to clear stale state. A transcript
  // request is started exclusively by a fresh microphone trigger event.
  if (data.is_transcribing !== true && ui.isTranscribing) {
    ui.isTranscribing = false;
    ui.pendingTranscriptId = '';
    ui.mic.disabled = false;
    ui.send.disabled = !ui.field.value.trim() || ui.isSubmitting;
    ui.status.textContent = 'Type a message and click Send, or click the microphone.';
  }
  const transcriptEventKey = data.transcript_event?.id
    ? `${data.transcript_event.id}:text`
    : '';
  if (transcriptEventKey && transcriptEventKey !== ui.lastTranscriptKey) {
    ui.lastTranscriptKey = transcriptEventKey;
    const transcriptWasIgnored = ui.ignoreTranscriptId === data.transcript_event.id;
    const typedQuestionWasSent = ui.typedSubmitDuringTranscription;
    const keepTypedText = ui.transcriptionEdited || transcriptWasIgnored;
    ui.isTranscribing = false;
    ui.pendingTranscriptId = '';
    if (data.transcript_event.text) {
        if (keepTypedText) {
          ui.mic.disabled = false;
          if (!typedQuestionWasSent) ui.send.disabled = !ui.field.value.trim() || ui.isSubmitting;
          ui.status.textContent = typedQuestionWasSent
            ? 'Thinking…'
            : transcriptWasIgnored && !ui.transcriptionEdited
              ? 'Type a message and click Send, or click the microphone.'
              : 'Your typed message was kept. Review it, then click Send.';
        } else {
          ui.lastTranscriptText = data.transcript_event.text;
          ui.field.value = data.transcript_event.text;
          ui.resizeField();
          ui.mic.disabled = false; ui.send.disabled = ui.isSubmitting;
          ui.status.textContent = 'Transcript ready. Review or edit it, then click Send.';
        }
        if (ui.ignoreTranscriptId === data.transcript_event.id) ui.ignoreTranscriptId = '';
      } else {
        if (!keepTypedText) {
          ui.field.value = '';
          ui.resizeField();
        }
        ui.mic.disabled = false; ui.send.disabled = !ui.field.value.trim() || ui.isSubmitting;
        ui.status.textContent = keepTypedText
          ? 'Your typed message was kept. Review it, then click Send.'
          : 'No speech could be transcribed. Your input is still empty; please try again or type your question.';
    }
  }
  if (data.clear_id && data.clear_id !== ui.lastClearId) {
    ui.lastClearId = data.clear_id;
    ui.field.value = '';
    ui.resizeField();
    ui.status.textContent = 'Type a message and click Send, or click the microphone.';
    ui.typedSubmitDuringTranscription = false;
    ui.isSubmitting = false;
    if (ui.ignoreTranscriptId && !ui.isTranscribing) ui.transcriptionEdited = false;
    ui.send.disabled = !ui.field.value.trim(); ui.mic.disabled = false;
  } else if (data.failed_id && data.failed_id !== ui.lastFailedId) {
    ui.lastFailedId = data.failed_id;
    ui.isSubmitting = false;
    ui.send.disabled = !ui.field.value.trim();
    ui.status.textContent = 'The message could not be sent. Review it and press Enter to retry.';
  }
}
"""

_PLAYER_HTML = """
<div class="speech-actions" role="group" aria-label="Speak this AI response">
  <button type="button" data-action="speak" aria-label="Listen to AI response">🔊 Listen</button>
  <button type="button" data-action="pause" aria-label="Pause speech">⏸ Pause</button>
  <button type="button" data-action="resume" aria-label="Resume speech">▶ Resume</button>
  <button type="button" data-action="stop" aria-label="Stop speech">⏹ Stop</button>
  <span data-speech-status role="status" aria-live="polite"></span>
</div>
"""

_PLAYER_CSS = """
.speech-actions { display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin:8px 0 4px; font:inherit; }
.speech-actions button { border:1px solid var(--st-border-color,#cbd5e1); border-radius:7px; padding:5px 10px; background:var(--st-background-color,#fff); color:var(--st-text-color,#1f2937); font:inherit; font-size:.82rem; cursor:pointer; }
.speech-actions button:hover { border-color:var(--st-primary-color,#2563eb); }
.speech-actions span { font-size:.78rem; opacity:.75; }
"""

_PLAYER_JS = r"""
export default function(component) {
  const { parentElement: root, data, setStateValue } = component;
  if (!root.__careerSpeechPlayer) {
    root.innerHTML = `__PLAYER_HTML__`;
    root.__careerSpeechPlayer = { data, utterance: null, active: false, settings: {} };
    const status = root.querySelector('[data-speech-status]');
    const controller = root.__careerSpeechPlayer;
    const speak = () => {
      const current = controller.data;
      if (!('speechSynthesis' in window) || !window.SpeechSynthesisUtterance) { status.textContent = 'Speech playback is unsupported in this browser.'; return; }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(current.speech_text || '');
      utterance.rate = Number(current.settings?.rate ?? 1); utterance.pitch = Number(current.settings?.pitch ?? 1); utterance.volume = Number(current.settings?.volume ?? 1);
      const selected = window.speechSynthesis.getVoices().find(v => v.voiceURI === current.settings?.voice_uri);
      if (selected) utterance.voice = selected;
      controller.utterance = utterance; controller.active = false;
      status.textContent = '';
      utterance.onstart = () => { controller.active = true; status.textContent = 'Speaking…'; };
      utterance.onend = () => { controller.active = false; status.textContent = ''; };
      utterance.onerror = (event) => { controller.active = false; status.textContent = event.error === 'canceled' || event.error === 'interrupted' ? '' : 'Speech could not be played. Check browser audio settings.'; };
      window.speechSynthesis.speak(utterance);
    };
    controller.speak = speak;
    root.querySelector('[data-action="speak"]').addEventListener('click', speak);
    root.querySelector('[data-action="pause"]').addEventListener('click', () => {
      if (window.speechSynthesis?.speaking && !window.speechSynthesis.paused) { window.speechSynthesis.pause(); status.textContent = 'Speech paused.'; }
      else status.textContent = 'There is no active speech to pause.';
    });
    root.querySelector('[data-action="resume"]').addEventListener('click', () => {
      if (window.speechSynthesis?.paused) { window.speechSynthesis.resume(); status.textContent = 'Speaking…'; }
      else status.textContent = 'There is no paused speech to resume.';
    });
    root.querySelector('[data-action="stop"]').addEventListener('click', () => {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      controller.active = false; status.textContent = '';
    });
  }
  const controller = root.__careerSpeechPlayer;
  controller.data = data;
}
""".replace("`__PLAYER_HTML__`", repr(_PLAYER_HTML))


voice_toolbar = st.components.v2.component(
    "career_chat_voice_toolbar",
    html=_TOOLBAR_HTML,
    css=_TOOLBAR_CSS,
    js=_TOOLBAR_JS,
)

voice_input = st.components.v2.component(
    "career_chat_voice_input",
    html=_INPUT_HTML,
    css=_INPUT_CSS,
    js=_INPUT_JS,
)

chat_composer = st.components.v2.component(
    "career_chat_composer",
    html=_COMPOSER_HTML,
    css=_COMPOSER_CSS,
    js=_COMPOSER_JS,
)

voice_player = st.components.v2.component(
    "career_chat_voice_player",
    html=_PLAYER_HTML,
    css=_PLAYER_CSS,
    js=_PLAYER_JS,
)


def voice_defaults() -> dict:
    """Return fresh default component state to avoid sharing mutable settings."""
    return {"settings": dict(_DEFAULT_SETTINGS), "transcript_event": {"id": "", "text": ""}}


def voice_input_defaults() -> dict:
    """Return fresh state for the compact chat composer microphone."""
    return {"transcript_event": {"id": "", "text": ""}}


def voice_settings(state) -> dict:
    """Return valid stored settings with safe defaults for incomplete state."""
    settings = state.get("settings", {}) if state else {}
    result = dict(_DEFAULT_SETTINGS)
    if isinstance(settings, dict):
        result.update({key: settings[key] for key in result if key in settings})
    for key, low, high in (("rate", 0.65, 1.5), ("pitch", 0.5, 1.8), ("volume", 0.0, 1.0)):
        try:
            result[key] = min(high, max(low, float(result[key])))
        except (TypeError, ValueError):
            result[key] = _DEFAULT_SETTINGS[key]
    result["voice_uri"] = str(result["voice_uri"] or "")
    return result
