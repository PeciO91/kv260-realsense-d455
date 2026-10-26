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

## Next
_Queued, but not started._

## Someday / maybe

## Shipped
_Move completed plans here so the top stays about the future._

- [x] YOLO display/default tuning: confidence default 0.30, max-detections
  default 50, `--show-depth-roi` / `--show-confidence` on browser + Pi,
  browser `--telemetry true|false` gating monitor polling/overlay/logs.
  Verified by native control fixture plus host extraction fixtures
  (`tests/test_yolov26_overlay_options.py`); board build not run on host.
