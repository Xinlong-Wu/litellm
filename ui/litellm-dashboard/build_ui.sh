#!/bin/bash
set -euo pipefail

DASHBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DASHBOARD_DIR}/../.." && pwd)"
DESTINATION_DIR="${REPO_ROOT}/litellm/proxy/_experimental/out"

cd "${DASHBOARD_DIR}"

if [ -f "ui_colors.json" ]; then
  echo "Contents of ui_colors.json:"
  cat ui_colors.json
fi

npm run build

if [ ! -d "out" ]; then
  echo "Next.js export directory was not created: ${DASHBOARD_DIR}/out" >&2
  exit 1
fi

rm -rf "${DESTINATION_DIR}"
mkdir -p "${DESTINATION_DIR}"
cp -R out/. "${DESTINATION_DIR}/"
rm -rf out

echo "Admin UI copied to ${DESTINATION_DIR}"
