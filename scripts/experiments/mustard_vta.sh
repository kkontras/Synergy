#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

echo "No Mustard VTA configs exist under configs/FactorCL/Mustard/release/VTA."
echo "Available Mustard setup is VT only. Use scripts/experiments/mustard_vt.sh."
exit 1
