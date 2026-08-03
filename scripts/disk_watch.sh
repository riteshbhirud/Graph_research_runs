#!/bin/bash
# Disk watchdog for the QUEST BrowseComp-Plus runs.
#
# Only ever removes REGENERABLE artefacts. It never touches:
#   trajectories/*/logs/**            (the trajectory data itself)
#   trajectories/*/deepresearch/**    (resume state / predictions)
#   *.jsonl event logs, checkpoints/  (analysis inputs)
#   data/browsecomp_plus/indexes/     (BM25 index)
# Model weights are deleted manually at run transitions, because a live vLLM
# server holds them mmap'd.
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOG=$ROOT/logs/disk_watch.log
WARN_GB=80
CRIT_GB=50

free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

while true; do
    FREE=$(free_gb)
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    STATS=""
    for d in "$ROOT"/trajectories/*/ ; do
        case "$d" in *_archive_*) continue;; esac
        f=$(ls "$d"deepresearch/*/iter1.jsonl 2>/dev/null | head -1)
        [ -n "$f" ] && STATS="$STATS $(basename "$d")=$(wc -l < "$f")"
    done
    echo "$TS free=${FREE}G$STATS" >> "$LOG"

    if [ "$FREE" -lt "$WARN_GB" ]; then
        echo "$TS WARN below ${WARN_GB}G - clearing regenerable caches" >> "$LOG"
        rm -rf "$ROOT"/hf_cache/datasets "$ROOT"/hf_cache/hub/datasets--* 2>/dev/null
        rm -rf /tmp/chk* 2>/dev/null
        find "$ROOT/systems" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
        find "$ROOT/trajectories" -name 'condenser_call_*.json' -delete 2>/dev/null
        echo "$TS after cleanup free=$(free_gb)G" >> "$LOG"
    fi

    if [ "$(free_gb)" -lt "$CRIT_GB" ]; then
        echo "$TS CRITICAL free=$(free_gb)G - manual intervention needed (delete finished model weights)" >> "$LOG"
    fi
    sleep 600
done
