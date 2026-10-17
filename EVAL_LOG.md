# Eval log

Speech clips are listed in `eval/fixtures/manifest.json` and are blocked until real sermon recordings exist. Do not treat a blocked clip as a score.

chrF is reported only when a clip has both a reference translation and a hypothesis. Silence and tone have neither, so chrF is not applicable. WER against an empty reference is 0 when the pipeline also produced no words, and 1 when it invented some.

## 2026-10-18 — harness, synthetic clips, fake recognizer

Command: `python -m eval.harness --engine fake`

The fake recognizer returns no words. It checks the voice-activity gate, not Whisper and not Qwen. No sermon WER or translation score is claimed.

| Clip | Result |
| --- | --- |
| silence-5s | ran. WER 0. Recognizer calls 0. No caption. chrF not applicable. |
| tone-music | ran. WER 0. Recognizer calls 1. No caption. Banner `Worship`. chrF not applicable. |
| pending-en-fast | blocked. No audio. |
| pending-en-runon | blocked. No audio. |
| pending-en-correction | blocked. No audio. |
| pending-en-verse | blocked. No audio. |
| pending-id-glossary | blocked. No audio. |
| pending-name-grace | blocked. No audio. |
| pending-codeswitch | blocked. No audio. |

Glossary check, not a speech score: `Grace Church is full of grace` with Grace Church on the do-not-translate list became `Grace Church is full of anugerah`. No glossary token was dropped.

Latency against the 3 second / 5 second targets was not measured. That number has to come from a GPU run with real speech.
