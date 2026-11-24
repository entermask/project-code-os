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

## Review tooling (chỉ Claude Code)

> gstack đã gỡ bỏ 21/08/2026. Còn lại: built-in slash command + plugin `codex@openai-codex` (repo `openai/codex-plugin-cc`).

- `/code-review` — bug review trên diff hiện tại.
- `/security-review` — review bảo mật (có auth + reference-audio cache). Đừng chạy chung một diff với `/code-review`.
- `/codex:review`, `/codex:adversarial-review` — second opinion từ Codex CLI.

Deploy vẫn là scp lên box GPU (không git/CI) — không có slash command nào tự deploy.
