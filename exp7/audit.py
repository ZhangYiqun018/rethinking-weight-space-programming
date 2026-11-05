"""Record provenance and check data invariants without inspecting trained weights."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'upstream/lm_understand-logs/exp7_sycophancy'
files = [SOURCE / n for n in ['README.md', 'code/train.py', 'code/eval.py', 'dataset/data.py', 'dataset/facts.json']]
files += sorted((ROOT / 'data').glob('pair*.json'))
files += sorted((ROOT / 'models/base').glob('*.safetensors'))
files += [ROOT / 'models/base/config.json', ROOT / 'models/base/tokenizer_config.json']
hashes = {}
for path in files:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    hashes[str(path.relative_to(ROOT.parent))] = digest.hexdigest()
for path in (ROOT / 'data').glob('pair*.json'):
    parts = json.loads(path.read_text())
    assert not {r['question'] for r in parts['train']} & {r['question'] for r in parts['heldout']}
    for rows in parts.values():
        for row in rows:
            assert row['correct'].lower() != row['swap'].lower()
            assert any(r['correct'] == row['swap'] and r['type'] == row['type'] for r in rows)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
              mps_available=torch.backends.mps.is_available(), versions={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'safetensors', 'numpy']}, sha256=hashes)
(ROOT / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print('Data invariants passed; source and base hashes recorded.')
