# CPX fork notes

This is Commit-Push-Pray's fork of `git@github.com:JuneYaooo/capsule-cinema.git`, used by CPX-GenAI.

- `main` mirrors upstream and is never edited here.
- `cpx` is our working branch (default); merge `upstream/main` into it to take upstream changes.
- Original license and notices are unchanged. Files we modify are marked in the commit history.

## Changes in this fork

### OrcaRouter video generation adapter (Phase 5)

Added an OrcaRouter-hosted Kling video generation tool that routes through `api.orcarouter.ai`,
enabling capsule-cinema to generate video via the same API used by CPX-GenAI's media layer.

**New files:**
- `lib/custom_tools/video_generation/orcarouter_video_generator_tool.py` — tool adapter
  following the MiniMax H3 pattern: Pydantic schema, `BaseTool` subclass, `_run()` with
  submit/poll/download lifecycle, standard result dict.
- `lib/config/local-channels/video_engines.yaml` — engine entries for `orcarouter-kling-v2-5-turbo`
  ($0.042/req, 720p) and `orcarouter-kling-v2-master` ($0.280/req, 1080p).
- `lib/config/local-channels/tool_registry.yaml` — tool class → engine binding for both models.

**Env vars:** `ORCAROUTER_API_KEY` (or `GENAI_MEDIA_API_KEY`), `ORCAROUTER_BASE_URL`
(default `https://api.orcarouter.ai/v1`), `ORCAROUTER_VIDEO_MODEL` (optional override).

**Supported modes:** `text_to_video`, `image_to_video` (first frame as base64 data URI).
Kling models generate 5s or 10s clips at 16:9, 9:16, or 1:1. Pricing is flat per request.
