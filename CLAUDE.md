# Fish-Audio — Higgs Audio v3 TTS FastAPI wrapper

FastAPI wrapper for Higgs Audio v3 TTS served by SGLang-Omni (see `README.md`).
The API process does not load the model — it handles auth, reference-audio caching,
async jobs, chunk progress, and proxies each chunk to SGLang's `/v1/audio/speech`.

The whole service lives in one large `app.py` (~1.5k lines). This makes scope
discipline (principles 2 & 3 below) the single most important thing here.

## Language
- Reply to the user in **Vietnamese**.

## Working style (Karpathy guidelines)

> Behavioral guidelines to reduce common LLM coding mistakes. Biases toward caution
> over speed — use judgment on trivial tasks.

1. **Think before coding.** State assumptions explicitly. If multiple interpretations
   exist, present them — don't pick silently. If a simpler approach exists, say so;
   push back when warranted. If something is unclear, stop and ask. *Scope:*
   design/business decisions only — NOT routine git mechanics (push/commit with
   `--no-verify`, don't ask).
2. **Simplicity first.** Minimum code that solves the problem: no speculative
   features, no abstractions for single-use code, no unrequested "flexibility", no
   error handling for impossible scenarios. `app.py` is already large — resist adding
   more layers. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** `app.py` is a single big file: edit only the target
   function, don't reorganize, reformat, or "improve" surrounding code, and match the
   existing style. Mention unrelated dead code, don't delete it. Only remove
   imports/vars/functions that YOUR change orphaned.
4. **Goal-driven execution.** Turn vague tasks into verifiable goals: "fix the bug" →
   "write a repro test, then make it green". For multi-step work, state a brief plan
   with a verify step each. Before done: `pytest` (tests in `tests/`) must pass.

## gstack skills (chỉ Claude Code)

> Cài global ở `~/.claude/skills/gstack` (slash command, telemetry OFF). Repo Python (FastAPI) + deploy scp → fit thấp; chỉ dùng eng skill ngôn-ngữ-trung-lập.

- `/investigate` — root-cause cho lỗi chunk/timeout/auth trong `app.py`.
- `/cso` — review bảo mật (có auth + reference-audio cache).
- `/review` — bug review (lưu ý trùng built-in `/code-review`).
- `/spec`, `/learn`, `/document-*`.

KHÔNG dùng: `/qa`, `/browse`, `/design-*`, `/benchmark`, `/ios-*` (no UI); `/ship`, `/land-and-deploy`, `/canary` (deploy = scp lên box GPU, không git/CI).
