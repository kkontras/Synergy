#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

echo "No URFunny VTA configs exist under configs/FactorCL/URFunny/release/VTA."
echo "Available URFunny setup is VT only. Use scripts/experiments/ur_funny.sh or scripts/experiments/ur_funny_final.sh."
exit 1
