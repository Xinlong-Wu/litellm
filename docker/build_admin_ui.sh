#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_DIR="${REPO_ROOT}/ui/litellm-dashboard"
NODE_VERSION="$(tr -d '[:space:]' < "${DASHBOARD_DIR}/.nvmrc")"
NVM_VERSION="v0.40.4"
NVM_CHECKSUM="4b7412c49960c7d31e8df72da90c1fb5b8cccb419ac99537b737028d497aba4f"

ensure_curl() {
  if command -v curl >/dev/null 2>&1; then
    return
  fi

  echo "curl is required to install nvm. Install curl and rerun this script." >&2
  exit 1
}

load_nvm() {
  export NVM_DIR="${NVM_DIR:-${HOME}/.nvm}"

  if [ -s "${NVM_DIR}/nvm.sh" ]; then
    . "${NVM_DIR}/nvm.sh"
  fi
}

install_nvm() {
  ensure_curl

  local nvm_script
  nvm_script="$(mktemp)"
  trap "rm -f '${nvm_script}'" EXIT

  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" -o "${nvm_script}"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${NVM_CHECKSUM}  ${nvm_script}" | sha256sum -c -
  elif command -v shasum >/dev/null 2>&1; then
    echo "${NVM_CHECKSUM}  ${nvm_script}" | shasum -a 256 -c -
  else
    echo "No sha256 tool found; cannot verify nvm checksum" >&2
    exit 1
  fi

  bash "${nvm_script}"
  load_nvm
}

load_nvm
if ! command -v nvm >/dev/null 2>&1; then
  install_nvm
fi

nvm install "${NODE_VERSION}"
nvm use "${NODE_VERSION}"

if [ -f "${REPO_ROOT}/enterprise/enterprise_ui/enterprise_colors.json" ]; then
  echo "Building enterprise Admin UI colors"
  cp "${REPO_ROOT}/enterprise/enterprise_ui/enterprise_colors.json" "${DASHBOARD_DIR}/ui_colors.json"
else
  echo "Building default LiteLLM Admin UI"
fi

cd "${DASHBOARD_DIR}"
npm ci
bash ./build_ui.sh
