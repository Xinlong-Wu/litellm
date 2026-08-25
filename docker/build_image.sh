#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: docker/build_image.sh [--dockerfile PATH] [--target TARGET] [--tag TAG] [docker build args...]

Defaults:
  --dockerfile Dockerfile
  --target runtime
  --tag litellm:local

Examples:
  docker/build_image.sh
  docker/build_image.sh --dockerfile docker/Dockerfile.non_root --tag litellm:non-root --build-arg PROXY_EXTRAS_SOURCE=local
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="Dockerfile"
TARGET="runtime"
TAGS=("-t" "litellm:local")
DOCKER_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --dockerfile=*)
      DOCKERFILE="${1#*=}"
      shift
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --target=*)
      TARGET="${1#*=}"
      shift
      ;;
    --tag|-t)
      TAGS=("-t" "$2")
      shift 2
      ;;
    --tag=*)
      TAGS=("-t" "${1#*=}")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      DOCKER_ARGS+=("$@")
      break
      ;;
    *)
      DOCKER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${DOCKERFILE}" != /* ]]; then
  DOCKERFILE="${REPO_ROOT}/${DOCKERFILE}"
fi

bash "${REPO_ROOT}/docker/build_admin_ui.sh"

docker build \
  -f "${DOCKERFILE}" \
  --target "${TARGET}" \
  "${TAGS[@]}" \
  "${DOCKER_ARGS[@]}" \
  "${REPO_ROOT}"
