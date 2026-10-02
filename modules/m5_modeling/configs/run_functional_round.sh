#!/bin/bash
set -euo pipefail
workspace="$1"
python_path="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ge 3 ]; then
  "$python_path" "$script_dir/../components/functional_vm.py" --workspace "$workspace" --dataset uci_bank_marketing --recover-round "$3"
else
  "$python_path" "$script_dir/../components/functional_vm.py" --workspace "$workspace" --dataset uci_bank_marketing
fi
"$python_path" "$script_dir/../components/functional_vm.py" --workspace "$workspace" --dataset hillstrom_email_marketing
