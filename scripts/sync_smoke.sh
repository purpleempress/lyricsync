#!/usr/bin/env bash
# Exercise POST /sync end-to-end against a real track id (placeholder lines).
set -euo pipefail
cd "$(dirname "$0")/.."
REQ='{"spotifyId":"3n3Ppam7vgaVa1iaRUc9Lp","title":"x","artist":"y","duration":222000,"lyrics":["la la la la","na na na na","la la la la","oh oh oh oh"],"demucs":false,"model":"base"}'
JOB=$(curl -s -X POST http://127.0.0.1:8000/sync -H "content-type: application/json" -d "$REQ")
JOBID=$(echo "$JOB" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
echo "job $JOBID"
for i in $(seq 1 60); do
  sleep 4
  ST=$(curl -s "http://127.0.0.1:8000/jobs/$JOBID")
  S=$(echo "$ST" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "poll $((i*4))s: $S"
  if [ "$S" = "done" ] || [ "$S" = "error" ]; then
    echo "$ST" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if d['status']=='error':
    print('ERR', d.get('error'))
else:
    r=d['result']
    print('OK Type', r['Type'], 'items', len(r['Content']), 'span', r['StartTime'], '->', r['EndTime'])"
    break
  fi
done
