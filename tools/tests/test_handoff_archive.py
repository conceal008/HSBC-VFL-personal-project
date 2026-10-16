"""Non-experimental checks for the paused handoff and non-destructive migration."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
MIGRATIONS = REPOSITORY / 'docs/04-experiment-records/migration_manifest.json'
ARCHIVE = REPOSITORY / 'modules/m5_modeling/experiments/paused/S5.D2_20261015'
MARKDOWN_LINK = re.compile(r'(!?\[[^\]]*\]\()([^\s)]+)(\))')


def without_destinations(text):
    return MARKDOWN_LINK.sub(lambda match: match.group(1) + 'TARGET' + match.group(3), text)


def test_moved_log_and_report_bodies_preserve_every_word_and_number():
    workflow = yaml.safe_load((REPOSITORY / '.github/workflows/gates.yml').read_text())
    assert workflow['jobs']['gate5-code-quality']['steps'][0]['with']['fetch-depth'] == 0
    manifest = json.loads(MIGRATIONS.read_text())
    assert len(manifest['assets']) == 11
    for item in manifest['assets']:
        original = subprocess.check_output(['git', 'show', manifest['source_commit'] + ':' + item['old_path']], cwd=REPOSITORY)
        moved = (REPOSITORY / item['canonical_path']).read_bytes()
        assert hashlib.sha256(original).hexdigest() == item['before_sha256']
        assert hashlib.sha256(moved).hexdigest() == item['after_sha256']
        assert without_destinations(original.decode()) == without_destinations(moved.decode())
        legacy = (REPOSITORY / item['old_path']).read_text()
        assert item['canonical_path'] in legacy and '资料迁移入口' in legacy


def test_paused_sources_are_byte_preserved_and_cannot_be_active_components():
    inventory = json.loads((ARCHIVE / 'inventory.json').read_text())
    assert len(inventory['items']) == 13 and inventory['restore_automatically'] is False
    for item in inventory['items']:
        file = REPOSITORY / item['archived_path']
        assert file.name.endswith('.pending') and file.is_file()
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['sha256']
        assert file.stat().st_size == item['bytes']
        assert not (REPOSITORY / item['original_path']).exists()
    assert not list(ARCHIVE.rglob('*.py')) and not list(ARCHIVE.rglob('*.ipynb'))


def test_current_entry_and_moved_documents_have_resolvable_local_links():
    paths = [REPOSITORY / 'README.md', ARCHIVE / 'README.md']
    paths += list((REPOSITORY / 'docs/03-handoff').glob('*.md'))
    paths = [p for p in paths if p.name in {'README.md', 'START_HERE.md', 'CURRENT_STATUS.md', 'FILESYSTEM_MAP.md', 'RESTART_PLAN.md', 'HO-20261015-Codex-M9-paused-handoff.md'}]
    paths += list((REPOSITORY / 'docs/04-experiment-records').rglob('*.md'))
    for path in paths:
        for match in MARKDOWN_LINK.finditer(path.read_text()):
            target = match.group(2)
            if target.startswith(('http:', 'https:', 'mailto:', 'app:', 'codex:', '#', '/')):
                continue
            destination = target.partition('#')[0]
            assert (path.parent / destination).exists(), f'{path.relative_to(REPOSITORY)} -> {target}'


def test_pause_is_explicit_and_no_incomplete_step_is_marked_completed():
    pause = yaml.safe_load((REPOSITORY / 'docs/03-handoff/暂停状态.yaml').read_text())
    ledger = yaml.safe_load((REPOSITORY / 'modules/m5_modeling/step_ledger.yaml').read_text())
    status = yaml.safe_load((REPOSITORY / 'registry/module_status.yaml').read_text())
    assert pause['status'] == 'paused_by_user' and pause['resume_automatically'] is False
    assert pause['vm_state'] == {'alice': 'Stopped', 'bob': 'Stopped'}
    assert pause['d2_completed_tasks'] == ledger['paused_step']['completed_tasks'] == 0
    assert ledger['paused_step']['score'] is None
    assert ledger['paused_step']['revision_run_completed'] is False
    assert ledger['steps'][-1]['step_id'] == 'S5.X3'
    assert ledger['quality_snapshot_before_D2']['steps_used'] == 21
    assert ledger['quality_snapshot_before_D2']['revised_steps'] == 14
    assert status['experiment_pause']['status'] == 'paused_by_user'
