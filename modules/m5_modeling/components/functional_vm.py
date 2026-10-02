"""Trusted local environment provider; never reads dataset rows or joins them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
PARTIES = ("alice", "bob")
VM_NAMES = {p: f"hsbc-{p}" for p in PARTIES}
LIMA_HOME_PATH = "/private/tmp/hsbc-vfl-lima"
REPO_PARENT_DEPTH = 3
TLS_DAYS = 30
ROUND_TIMEOUT_SECONDS = 21600
CHECK_INTERVAL_SECONDS = 1
FED_PORT = 50051
SPU_PORT = 50052
PREPARED = {
    "uci_bank_marketing": "uci_bank_marketing_20260928T131533922243Z_b2954bae",
    "hillstrom_email_marketing": "hillstrom_email_marketing_20260928T131536878290Z_1d3cbaee",
}


def save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def lima(*args, capture=True):
    environment = os.environ.copy()
    environment["LIMA_HOME"] = LIMA_HOME_PATH
    return subprocess.run(["limactl", *map(str, args)], env=environment, check=True,
                          text=True, capture_output=capture)


def guest(party, script):
    return lima("shell", VM_NAMES[party], "bash", "-lc", script).stdout.strip()


def copy_to(source, party, target):
    lima("copy", source, f"{VM_NAMES[party]}:{target}")


def setup_certificates(workspace):
    authority = workspace / "功能模拟环境/authority"
    authority.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    key, ca = authority / "key.pem", authority / "ca.pem"
    if not key.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:3072", "-nodes",
                        "-keyout", str(key), "-out", str(ca), "-days", str(TLS_DAYS),
                        "-subj", "/CN=HSBC-local-functional-CA"], check=True, capture_output=True)
        key.chmod(PRIVATE_FILE_MODE)
    addresses = {}
    for party in PARTIES:
        host = f"lima-hsbc-{party}.internal"
        ip = guest(party, f"getent ahostsv4 {host} | head -1 | awk '{{print $1}}'")
        if not ip:
            raise RuntimeError("No participant VM IP")
        addresses[party] = ip
        csr = authority / f"{party}.csr"
        guest(party, f"sudo cat /srv/vfl/{party}/tls/request.csr > /tmp/{party}_request.csr")
        lima("copy", f"{VM_NAMES[party]}:/tmp/{party}_request.csr", csr)
        extension = authority / f"{party}.ext"
        extension.write_text(f"subjectAltName=DNS:{host},IP:{ip}\nextendedKeyUsage=serverAuth,clientAuth\nkeyUsage=digitalSignature,keyEncipherment\n")
        cert = authority / f"{party}.pem"
        subprocess.run(["openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca),
                        "-CAkey", str(key), "-CAcreateserial", "-out", str(cert),
                        "-days", str(TLS_DAYS), "-extfile", str(extension)], check=True, capture_output=True)
        for path, name in ((ca, "ca.pem"), (cert, "cert.pem")):
            copy_to(path, party, f"/tmp/{party}_{name}")
            guest(party, f"sudo install -o {party} -g {party} -m 600 /tmp/{party}_{name} /srv/vfl/{party}/tls/{name}")
    return addresses


def restrict_network(party, addresses):
    # Only the nonprivileged training account is restricted; management remains outside its threat domain.
    uid = guest(party, f"id -u {party}")
    script = f"""set -e
sudo iptables -N VFL_{party.upper()} 2>/dev/null || true
sudo iptables -F VFL_{party.upper()}
sudo iptables -A VFL_{party.upper()} -o lo -j ACCEPT
sudo iptables -A VFL_{party.upper()} -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
"""
    for ip in addresses.values():
        script += f"sudo iptables -A VFL_{party.upper()} -p tcp -d {ip} -m multiport --dports {FED_PORT},{SPU_PORT} -j ACCEPT\n"
    script += f"""sudo iptables -A VFL_{party.upper()} -j REJECT
sudo iptables -C OUTPUT -m owner --uid-owner {uid} -j VFL_{party.upper()} 2>/dev/null || sudo iptables -A OUTPUT -m owner --uid-owner {uid} -j VFL_{party.upper()}
sudo ip6tables -C OUTPUT -m owner --uid-owner {uid} -j REJECT 2>/dev/null || sudo ip6tables -A OUTPUT -m owner --uid-owner {uid} -j REJECT
"""
    # Pin DNS ahead of egress restrictions. Names are public infrastructure metadata.
    for name, ip in addresses.items():
        script += f"grep -q 'lima-hsbc-{name}.internal' /etc/hosts || echo '{ip} lima-hsbc-{name}.internal' | sudo tee -a /etc/hosts >/dev/null\n"
    guest(party, script)


def synthetic_inputs(config, dataset, party):
    """Deterministic test inputs; outcomes only assigned to Alice."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(config["smoke"]["data_seed"])
    rows = config["smoke"]["split_rows"]
    width = config["smoke"]["features_per_party"]
    x = rng.normal(size=(sum(rows), width * len(PARTIES)))
    treatment = rng.integers(0, 1 + len(config["datasets"][dataset].get("treatment_arms", [])), len(x))
    labels = (x.sum(axis=1) + config["smoke"]["treatment_effect"] * (treatment > 0) > 0).astype(int)
    offset = 0 if party == "alice" else width
    begin, frames = 0, {}
    for split, length in zip(("train", "validation", "test"), rows):
        frame = pd.DataFrame(x[begin:begin + length, offset:offset + width], columns=[f"numeric__f{idx}" for idx in range(width)])
        frame.insert(0, "record_id", [f"synthetic_{idx}" for idx in range(begin, begin + length)])
        if party == "alice":
            frame["label"] = labels[begin:begin + length]
            if config["datasets"][dataset]["treatment"]:
                frame["treatment"] = treatment[begin:begin + length]
        frames[split] = frame
        begin += length
    return frames


def stage_round(workspace, dataset, round_id, local, smoke=False, recovery_origin=None, arithmetic_field=None):
    repo = Path(__file__).resolve().parents[REPO_PARENT_DEPTH]
    code = repo / "modules/m5_modeling/components"
    config = repo / "modules/m5_modeling/configs/local_functional.yaml"
    notebook_name = "S5.P2_recovery.ipynb" if recovery_origin else "S5.P2_local_functional.ipynb"
    notebook = repo / "modules/m5_modeling/notebooks" / notebook_name
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    config_bytes = config.read_bytes()
    if arithmetic_field and not smoke:
        raise ValueError("Arithmetic reference override is synthetic-only")
    if smoke:
        import yaml
        parsed = yaml.safe_load(config_bytes)
        parsed["shared"].update(parsed["smoke"]["shared_overrides"])
        if arithmetic_field:
            parsed["field"] = arithmetic_field
        parsed["datasets"][dataset]["alice_features"] = parsed["smoke"]["features_per_party"]
        parsed["datasets"][dataset]["bob_features"] = parsed["smoke"]["features_per_party"]
        config_bytes = yaml.safe_dump(parsed, allow_unicode=True, sort_keys=False).encode()
    cfg_hash = hashlib.sha256(config_bytes).hexdigest()
    addresses = setup_certificates(workspace)
    for party in PARTIES:
        peer = "bob" if party == "alice" else "alice"
        # Prove a real listening SSH endpoint before its participant-level denial test.
        guest(party, f"timeout 5 bash -c '</dev/tcp/{addresses[peer]}/22'")
        restrict_network(party, addresses)
        guest_root = f"/srv/vfl/{party}/{round_id}"
        guest(party, f"sudo install -d -m 755 {guest_root}; sudo install -d -m 755 {guest_root}/code; "
                    f"sudo install -d -m 550 -o root -g {party} {guest_root}/input; "
                    f"sudo install -d -m 700 -o {party} -g {party} {guest_root}/data {guest_root}/results {guest_root}/logs {guest_root}/trainings")
        target = local / party
        target.mkdir(mode=PRIVATE_DIRECTORY_MODE)
        for subdir in ("code", "results", "logs", "data"):
            (target / subdir).mkdir()
        import shutil
        sources = [code / f"{name}.py" for name in ("functional_training", "functional_metrics", "functional_tls")]
        if recovery_origin:
            sources.append(code / "functional_recovery.py")
        sources += [config, notebook]
        for source in sources:
            snapshot = target / "code" / source.name
            shutil.copyfile(source, snapshot)
            if source == config:
                snapshot.write_bytes(config_bytes)
            copy_to(snapshot, party, f"/tmp/{party}_{source.name}")
            guest(party, f"sudo install -m 444 /tmp/{party}_{source.name} {guest_root}/code/{source.name}")
        prepared = PREPARED[dataset] if not smoke else None
        if smoke:
            manifest = {}
            for split, frame in synthetic_inputs(parsed, dataset, party).items():
                source = target / "data" / f"{split}.csv"
                frame.to_csv(source, index=False)
                manifest[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
                copy_to(source, party, f"/tmp/{party}_{source.name}")
                guest(party, f"sudo install -o root -g {party} -m 440 /tmp/{party}_{source.name} {guest_root}/input/{source.name}; sudo rm /tmp/{party}_{source.name}")
            manifest_path = target / "data/manifest.json"
            save(manifest_path, manifest)
            copy_to(manifest_path, party, f"/tmp/{party}_manifest.json")
            guest(party, f"sudo install -o root -g {party} -m 440 /tmp/{party}_manifest.json {guest_root}/input/manifest.json")
        elif prepared:
            own = workspace / "联邦隔离产物" / prepared / party
            prepared_manifest = json.loads((workspace / "联邦隔离结果" / prepared / party / "data_manifest.json").read_text())
            # The preparation manifest also contains clean_features.csv, which is not
            # a training input. Freeze exactly the three files actually delivered.
            manifest = {f"{split}.csv": prepared_manifest[f"{split}.csv"] for split in ("train", "validation", "test")}
            for split in ("train", "validation", "test"):
                source = own / f"{split}.csv"
                if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[source.name]:
                    raise ValueError("Prepared source changed")
                copy_to(source, party, f"/tmp/{party}_{source.name}")
                guest(party, f"sudo install -o root -g {party} -m 440 /tmp/{party}_{source.name} {guest_root}/input/{source.name}; sudo rm /tmp/{party}_{source.name}")
            manifest_path = target / "data/manifest.json"
            save(manifest_path, manifest)
            copy_to(manifest_path, party, f"/tmp/{party}_manifest.json")
            guest(party, f"sudo install -o root -g {party} -m 440 /tmp/{party}_manifest.json {guest_root}/input/manifest.json")
        runtime = {"party": party, "run_id": round_id, "dataset": dataset, "git_sha": git_sha,
                   "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (target / "code").iterdir() if p.is_file()},
                   "is_smoke": smoke,
                   "config": f"{guest_root}/code/local_functional.yaml", "config_sha256": cfg_hash,
                   "parties": {name: {"address": f"lima-hsbc-{name}.internal:{FED_PORT}",
                       "listen_addr": f"0.0.0.0:{FED_PORT}"} for name in PARTIES},
                   "spu_addresses": {name: f"lima-hsbc-{name}.internal:{SPU_PORT}" for name in PARTIES}}
        runtime["notebook"] = notebook_name
        if recovery_origin:
            origin_root = f"/srv/vfl/{party}/{recovery_origin}"
            runtime["origin_run_id"] = recovery_origin
            guest(party, f"sudo cp -a {shlex.quote(origin_root)}/trainings/. {shlex.quote(guest_root)}/trainings/")
        save(target / "code/runtime.json", runtime)
        copy_to(target / "code/runtime.json", party, f"/tmp/{party}_runtime.json")
        guest(party, f"sudo install -m 444 /tmp/{party}_runtime.json {guest_root}/code/runtime.json")
    save(local / "declaration.json", {"round_id": round_id, "dataset": dataset,
        "purpose": "two-VM secure functional VFL; fixed pre-aligned data; L0/L1/L3 comparators",
        "prepared_experiment": "synthetic_only" if smoke else PREPARED.get(dataset), "config_sha256": cfg_hash, "git_sha": git_sha,
        "physical_isolation": "not_required_by_user", "host_administrator": "trusted",
        "model_output_recipient": "alice", "raw_data_to_controller": False, "true_psi": "not_executed",
        "post_training_export": "trusted administrator exports Alice-authorized artifacts into her private host directory; federation driver receives status only"})
    if recovery_origin:
        declaration = json.loads((local / "declaration.json").read_text())
        declaration.update({"recovery_mode": "evaluation_only", "training_reused_from": recovery_origin,
                            "purpose": "new finalization directory; validate completed frozen fits and inputs before evaluating; no retraining or retuning"})
        save(local / "declaration.json", declaration)


def private_permissions(root):
    """Protect exports too; never follow links into frozen inputs or other roots."""
    for directory, _, files in os.walk(root, followlinks=False):
        Path(directory).chmod(PRIVATE_DIRECTORY_MODE)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(PRIVATE_FILE_MODE)


def launch(workspace, dataset, smoke=False, recovery_origin=None, arithmetic_field=None):
    workspace = Path(workspace).resolve()
    if dataset not in PREPARED and not smoke:
        raise ValueError("Dataset not registered")
    round_id = f"{'recovery_' if recovery_origin else ('smoke_' if smoke else '')}{dataset}_functional_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    local = workspace / "功能模拟训练" / dataset / round_id
    local.mkdir(parents=True, exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    processes = []
    logs = []
    passed = False
    try:
        if arithmetic_field:
            stage_round(workspace, dataset, round_id, local, smoke, recovery_origin, arithmetic_field)
        elif recovery_origin:
            stage_round(workspace, dataset, round_id, local, smoke, recovery_origin)
        else:
            stage_round(workspace, dataset, round_id, local, smoke)
        environment = os.environ.copy()
        environment["LIMA_HOME"] = LIMA_HOME_PATH
        for party in PARTIES:
            root = f"/srv/vfl/{party}/{round_id}"
            # A participant starts with a minimal environment and no inherited proxy or management keys.
            command = ["limactl", "shell", VM_NAMES[party], "sudo", "-u", party, "env", "-i",
                "PATH=/opt/secretflow/bin:/usr/bin:/bin", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                "PYTHONDONTWRITEBYTECODE=1", f"IPYTHONDIR={root}/logs/ipython",
                "/opt/secretflow/bin/python", "-I", "-B", "-c",
                "import os,sys;os.chdir(sys.argv[1]+'/..');sys.path.insert(0,sys.argv[1]);from functional_training import execute_notebook;execute_notebook(sys.argv[2])",
                f"{root}/code", f"{root}/code/runtime.json"]
            log = (local / party / "logs/driver.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + ROUND_TIMEOUT_SECONDS
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Participant failed; logs remain private")
            if time.monotonic() > deadline:
                raise TimeoutError("Functional round timeout")
            time.sleep(CHECK_INTERVAL_SECONDS)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("Participant failed")
        for party in PARTIES:
            root = f"/srv/vfl/{party}/{round_id}"
            # Do not gather MPC shares or local transformed rows into the controller.
            archive = f"/tmp/{party}_{round_id}.tar"
            guest(party, f"sudo tar --exclude='model.share' --exclude='*.npz' -cf {shlex.quote(archive)} -C {shlex.quote(root)} code results logs trainings")
            destination = local / party / "authorized_artifacts.tar"
            lima("copy", f"{VM_NAMES[party]}:{archive}", destination)
            with tarfile.open(destination) as bundle:
                bundle.extractall(local / party, filter="data")
            guest(party, f"sudo rm {shlex.quote(archive)}")
        passed = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
        if not passed:
            for party in PARTIES:
                root = f"/srv/vfl/{party}/{round_id}"
                try:
                    guest(party, f"sudo pkill -TERM -u {party} -f {shlex.quote(round_id)} || true")
                    archive = f"/tmp/{party}_{round_id}_failure.tar"
                    guest(party, f"sudo tar --exclude='model.share' --exclude='*.npz' -cf {shlex.quote(archive)} -C {shlex.quote(root)} results logs")
                    destination = local / party / "failure_artifacts.tar"
                    lima("copy", f"{VM_NAMES[party]}:{archive}", destination)
                    with tarfile.open(destination) as bundle:
                        bundle.extractall(local / party, filter="data")
                    guest(party, f"sudo rm {shlex.quote(archive)}")
                except (OSError, subprocess.SubprocessError, tarfile.TarError) as error:
                    save(local / f"{party}_failure_export.json", {"status": "not_collected", "error_type": type(error).__name__})
        save(local / "status.json", {"round_id": round_id, "status": "passed" if passed else "failed",
                                     "exit_codes": [process.poll() for process in processes]})
        (local / "实验日志.md").write_text(
            f"# 功能模拟实验日志：{round_id}\n\n目的、构造条件、代码配置指纹见 declaration.json。"
            "本方冻结输入只读；新增处理数据保留在各自虚拟机 data/，不可交叉访问。"
            "每次模型/参数/seed 的代码、结果及日志均位于各自 trainings/ 独立目录。"
            f"\n\n结果状态：{'passed' if passed else 'failed'}；错误和私有指标分别在本方目录。"
            "不构成物理隔离、真实客户PSI或生产安全结论。\n")
        index = workspace / "实验日志/实验索引.md"
        with index.open("a") as stream:
            stream.write(f"\n- `{round_id}`：双VM功能训练 {'passed' if passed else 'failed'}；私有日志 `{local}/实验日志.md`。\n")
        private_permissions(local)
    if not passed:
        raise RuntimeError(f"Functional experiment failed; see {local}")
    return local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=list(PREPARED))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recover-round", type=Path)
    parser.add_argument("--arithmetic-reference", choices=["FM64", "FM128"])
    args = parser.parse_args()
    if args.arithmetic_reference and not args.smoke:
        parser.error("Arithmetic reference is allowed only for synthetic smoke")
    origin = None
    if args.recover_round:
        original = json.loads((args.recover_round / "declaration.json").read_text())
        if original["dataset"] != args.dataset or args.smoke:
            raise ValueError("Recovery must use the same non-smoke dataset")
        origin = args.recover_round.name
    print(launch(args.workspace, args.dataset, args.smoke, origin, args.arithmetic_reference))
