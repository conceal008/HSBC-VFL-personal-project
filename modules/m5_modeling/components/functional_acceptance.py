"""Trusted post-training file audit; no original rows or MPC shares are collected."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat

import yaml

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
NOTEBOOK_CODE_CELLS = 3
MINIMUM_SEEDS = 5
SECURE_ROUTES = ("L1_secure", "L3_secure")
ROUTES = ("L0", *SECURE_ROUTES)
COMPARATORS = ("L0", "L1_secure")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message="Private functional round audit failed"):
    if not condition:
        raise ValueError(message)


def verify_round(round_path, workspace):
    root = Path(round_path).resolve()
    require(read(root / 'status.json')['status'] == 'passed', 'Round has not passed')
    declaration = read(root / 'declaration.json')
    config = yaml.safe_load((root / 'alice/code/local_functional.yaml').read_text())
    shared = config['shared']
    if shared.get('sigmoid') == 'df':
        for path in [root, *root.rglob('*')]:
            require(not path.is_symlink(), 'Private exports cannot contain links')
            expected_mode = PRIVATE_DIRECTORY_MODE if path.is_dir() else PRIVATE_FILE_MODE
            require(stat.S_IMODE(path.stat().st_mode) == expected_mode, 'Private export permissions failed')
    seeds = shared['model_seeds']
    require(len(seeds) >= MINIMUM_SEEDS and len(set(seeds)) == len(seeds))
    secure_fits = len(seeds) * len(shared['learning_rates']) * len(SECURE_ROUTES)
    local_fits = len(seeds) * (len(shared['l0_c_grid']) + 1)
    summary = {'round': root.name, 'verified_at': datetime.now(timezone.utc).isoformat(),
               'dataset': declaration['dataset'], 'checks': {}, 'source': str(root)}
    for party, expected_count in [('alice', secure_fits + local_fits), ('bob', secure_fits)]:
        p = root / party
        require(all(read(p / 'logs/isolation.json').values()), 'Isolation checks failed')
        require(all(read(p / 'logs/tls_checks.json').values()), 'TLS negative tests failed')
        if 'training_boundary_limit' in shared:
            require(read(p / 'logs/tls_checks.json').get('protocol_trace_disabled') is True)
            require(not list(p.rglob('spu.trace.log')), 'Protocol traces must not be exported')
        boundary = read(p / 'logs/output_boundary.json')
        require(boundary['frozen_input_unchanged'] is True)
        require(boundary['joint_weights'] == 'secret_shares_only')
        manifest = read(p / 'data/manifest.json')
        prepared = Path(workspace) / '联邦隔离产物' / declaration['prepared_experiment'] / party
        require(all(sha(prepared / name) == digest for name, digest in manifest.items()))
        runtime = read(p / 'code/runtime.json')
        require(all(sha(p / 'code' / name) == digest for name, digest in runtime['code_sha256'].items()))
        folders = sorted((p / 'trainings').iterdir())
        require(len(folders) == expected_count)
        require(not list(p.rglob('model.share')), 'Host collected model shares')
        require(not list(p.rglob('*.npz')), 'Host collected local transformed rows')
        for f in folders:
            require(all((f / sub).is_dir() for sub in ['code', 'data', 'results', 'logs']))
            require((f / 'code/environment.lock').is_file())
            fit_runtime_path = f / 'code/runtime.json'
            fit_runtime = read(fit_runtime_path) if fit_runtime_path.exists() else runtime
            require(all(sha(f / 'code' / name) == digest for name, digest in fit_runtime['code_sha256'].items()))
            refs = read(f / 'data/input_references.json')
            require(all(refs['frozen_input'].get(name) == digest for name, digest in manifest.items()) and refs['derived_input'])
            if f.name.startswith(('L1_', 'L3_')):
                progress = read(f / 'logs/progress.json')
                require(progress['completed_epochs'] == progress['planned_epochs'] == shared['epochs'])
            if party == 'alice':
                require(read(f / 'results/status.json')['training'] == 'passed')
                if 'training_boundary_limit' in shared and f.name.startswith(('L1_', 'L3_')):
                    numerical = read(f / 'results/numerical_checks.json')
                    require(numerical['probabilities_valid'] is True)
                    fraction = numerical['training_boundary_fraction']
                    require(math.isfinite(fraction) and 0 <= fraction <= shared['training_boundary_limit'])
                    require(numerical['boundary_limit'] == shared['training_boundary_limit'])
                require(read(f / 'results/validation_score.json')['selection_split'] == 'validation')
                require(all((f / 'results' / (s + '_predictions.npy')).exists() for s in ['train', 'validation', 'test']))
        notebook = read(p / 'results/party.executed.ipynb')
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        require([c['execution_count'] for c in code_cells] == list(range(1, NOTEBOOK_CODE_CELLS + 1)))
        require(not any(o['output_type'] == 'error' for c in code_cells for o in c['outputs']))
        require(read(p / 'results/NOTEBOOK_COMPLETED.json')['status'] == 'passed')
        if party == 'bob':
            require(boundary['bob_plaintext_output_check'] == 'passed')
            require(not list(p.rglob('*predictions.npy')))
            require(not (p / 'results/private_evaluation.json').exists())
        summary['checks'][party] = {'isolation': 'pass', 'mutual_authentication': 'pass',
            'frozen_inputs': 'unchanged', 'source_fingerprints': 'pass',
            'training_directories': expected_count, 'notebook_code_cells': NOTEBOOK_CODE_CELLS,
            'host_no_secret_shares_or_transformed_rows': True,
            'output_boundary': 'pass'}
    evaluation = read(root / 'alice/results/private_evaluation.json')
    require(evaluation['status'] == 'completed')
    require(len(evaluation['metrics']) == len(seeds) * len(ROUTES) and len(evaluation['paired_differences']) == len(seeds) * len(COMPARATORS))
    require(len(evaluation['output_attack_diagnostics']) == len(seeds) * len(ROUTES))
    for collection in ['metrics', 'paired_differences']:
        for row in evaluation[collection]:
            for v in row['metrics'].values():
                require(all(math.isfinite(v[k]) for k in ['estimate', 'ci_low', 'ci_high']))
                require(v['ci_low'] <= v['ci_high'] and v['valid_replicates'] == shared['bootstrap_repeats'])
    require(set(r['seed'] for r in evaluation['metrics']) == set(seeds))
    for seed in seeds:
        require(set(r['route'] for r in evaluation['metrics'] if r['seed'] == seed) == {'L0', 'L1_secure', 'L3_secure'})
    for row in evaluation['output_attack_diagnostics']:
        require(0 <= row['ci_low'] <= row['ci_high'] <= 1)
        require(math.isfinite(row['loss_attack_auc']))
    summary['evaluation'] = {'seeds': seeds, 'selected_route_seed_results': len(evaluation['metrics']),
        'paired_comparisons': len(evaluation['paired_differences']), 'output_risk_diagnostics': len(evaluation['output_attack_diagnostics']), 'ci_fields': 'pass',
        'selection': evaluation['selection'], 'evaluation_sha256': sha(root / 'alice/results/private_evaluation.json')}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    summary = verify_round(args.round, args.workspace)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps({"round": args.round.name, "verification": "passed", "record": str(args.destination)}, ensure_ascii=False))
