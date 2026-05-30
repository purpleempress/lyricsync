#!/usr/bin/env bash
# Build & push the RunPod serverless worker image, then create an endpoint from
# it in the RunPod console. RunPod pulls the image from a registry, so set IMAGE
# to a tag you can push to (Docker Hub, GHCR, ...):
#
#   IMAGE=docker.io/youruser/lyricsync-runpod:latest bash scripts/build_runpod.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${IMAGE:?set IMAGE=registry/user/repo:tag (a registry RunPod can pull from)}"

docker build -f Dockerfile.runpod -t "$IMAGE" .
docker push "$IMAGE"

cat <<EOF

Pushed: $IMAGE

Next, in https://www.runpod.io/console/serverless -> New Endpoint:
  - Container image:  $IMAGE
  - GPU:              16 GB (T4) or 24 GB (L4)
  - FlashBoot:        leave ON (default) -- keeps warmed workers around
  - Container disk:   ~15 GB (weights are baked into the image)

Then point LyricSync at it:
  export LYRICSYNC_BACKEND=runpod
  export RUNPOD_ENDPOINT_ID=<the endpoint id>
  export RUNPOD_API_KEY=<your runpod api key>
  bash scripts/run_api.sh
EOF
