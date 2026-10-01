#!/bin/bash
# Phase A watcher: commits and pushes every finished training, plus a 15-minute autosave.
#   - each new results/<cond>/seed*/round*.json -> own commit (Methodology.md table refreshed first)
#   - every AUTOSAVE_S seconds: regenerate analysis outputs, commit whatever changed in results/,
#     Methodology.md and NOTES.md, and push
# The GitHub token is read from Drive (.secrets, outside the repo) and never printed.
# Start: nohup bash colab/watcher.sh > /content/watcher.out 2>&1 &
cd /content/drive/MyDrive/final_project
TOKEN_FILE=/content/drive/MyDrive/.secrets/github_token
AUTOSAVE_S=${AUTOSAVE_S:-900}
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

push() {
  for i in 1 2 3; do
    git push -q "https://$(cat $TOKEN_FILE)@github.com/kneat2448/LTH-ML-final.git" main 2>&1 \
      | sed -E 's#https://[^@]*@#https://***@#g'
    # pushing to a URL does not move origin/main; update it so the retry check below is meaningful
    [ "${PIPESTATUS[0]}" -eq 0 ] && { git update-ref refs/remotes/origin/main main; echo "$(date -u +%T) pushed"; return; }
    sleep 30
  done
  echo "$(date -u +%T) PUSH FAILED (commits are safe locally; retried next time)"
}

commit() {  # commit "message" paths... ; retries if another git process holds the index lock
  local msg=$1; shift
  for i in 1 2 3 4 5; do
    git add "$@" 2>/dev/null
    git diff --cached --quiet && return 1
    git commit -q -m "$msg" -m "$TRAILER" && return 0
    sleep 5
  done
  return 1
}

last_save=$SECONDS
while true; do
  for f in $(git ls-files --others --exclude-standard 'results/*/seed*/round*.json' 2>/dev/null); do
    sleep 20  # let the chain finish writing log/compute_log
    cond=$(echo $f | cut -d/ -f2); seed=$(echo $f | cut -d/ -f3); base=$(basename $f .json)
    phase="Phase A"; [ "$seed" != seed0 ] && phase="Phase B ($seed)"; [[ $cond == h4_* ]] && phase="H4"
    msg=$(python - "$f" <<'PY'
import json,sys
from src.metrics import stability_summary
d=json.load(open(sys.argv[1])); st=stability_summary(d["sharpness"])
print(f"remaining {d['remaining']:.3f}, acc {d['test']['acc']:.4f}, S(0) {st['S0']:.3f}, max S(25-3k) {st['max_S_train']:.3f}, diverged {d['diverged']}")
PY
)
    python -m analysis.methodology_table >/dev/null 2>&1
    commit "$phase: $cond ${base/_/ }: $msg" "$f" results/$cond results/compute_log.csv Methodology.md \
      && echo "$(date -u +%T) committed $f" && push
  done
  if [ $((SECONDS - last_save)) -ge $AUTOSAVE_S ]; then
    last_save=$SECONDS
    python -m analysis.methodology_table >/dev/null 2>&1
    python -m analysis.plots >/dev/null 2>&1
    commit "Autosave $(date -u +%H:%M) UTC: chain logs, analysis outputs, notes" results Methodology.md NOTES.md \
      && echo "$(date -u +%T) autosave committed" && push
    # push anything committed earlier whose push failed
    [ -n "$(git log origin/main..main --oneline 2>/dev/null)" ] && push
  fi
  sleep 60
done
