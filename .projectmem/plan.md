# KV260_Camera — plan

> Editable **intent** file: ideas + plans — what we *mean to do*.
> This is NOT the event log. `events.jsonl` -> `summary.md` records what
> *happened*; this file records what we *intend*. The AI reads it at
> session start and edits it directly (like `PROJECT_MAP.md`): add ideas
> and plans, check items off, move done work down to Shipped. Plans are
> never logged as events.

## Ideas
_Loose thoughts, not yet committed to._

## Active plans
_What we're working toward now. Use `- [ ]` / `- [x]` checklists._

- [ ] #0012 KMS black screen — Phase 2 at the board with a monitor. Phase 1 (code + headless
  checks) done, uncommitted; staged build `/tmp/kv260-kms.2Ccm8T` on `ssh fpga`. Run
  `./inference/live_yolov26_hdmi --renderer kms --test-pattern --duration-seconds 30
  --kms-plane primary|overlay --fullscreen true|false` (X stopped) and confirm bars are
  visible/animating; then q/Esc + Ctrl-C exit and console recovery; then a real camera run.
  If overlay is black but primary visible: blending/DMA. If both black with clean dmesg:
  KD_GRAPHICS VT switch, 1024x768 mode. Do not `record_fix` until visually confirmed.

## Next
_Queued, but not started._

- [ ] User-run real vision verification on KV260 — deliberately manual
  (never executed by us): deployed app at `/home/root/kv260-hdmi`,
  user supplies model `/home/root/yolov26n_fastft_test/yolov26n_facehuman_kria.xmodel`.

## Someday / maybe

## Shipped
_Move completed plans here so the top stays about the future._

- [x] Native YOLOv26 HDMI display (approved plan plan-bd82a02c9a2efc5f):
  fullscreen X11 app + synthetic `--test-pattern`, launcher check/wake,
  nonblocking handoff, strict CLI, telemetry. Desktop wake confirmed;
  staged board build + offline CLI verified; window mapped + durations +
  SIGTERM exit verified; user physically confirmed synthetic pattern.
  q/Esc/WM-close exit paths unverified (injection ignored by GTK).
  Deployed `/home/root/kv260-hdmi`; real vision left manual (see Next).

- [x] YOLO display/default tuning: confidence default 0.30, max-detections
  default 50, `--show-depth-roi` / `--show-confidence` on browser + Pi,
  browser `--telemetry true|false` gating monitor polling/overlay/logs.
  Verified by native control fixture plus host extraction fixtures
  (`tests/test_yolov26_overlay_options.py`); board build not run on host.
