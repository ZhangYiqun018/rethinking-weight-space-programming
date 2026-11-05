"""Classifier only opens anonymous A/B tensors and the base; never opens the key."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parent


def weights(folder):
    tensors = {}
    for path in sorted(folder.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as f:
            for name in f.keys():
                if '.layers.' in name:
                    tensors[name] = f.get_tensor(name).float()
    assert {int(k.split('.layers.')[1].split('.')[0]) for k in tensors} == set(range(18))
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', nargs='+', type=int)
    parser.add_argument('--output', default='calls_confirm.json')
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), 'Calls are immutable; choose a new output for added pairs'
    base = weights(ROOT / 'models/base')
    result = {}
    for pair in args.pairs:
        result[str(pair)] = {}
        for letter in 'AB':
            model = weights(ROOT / f'blind_models/pair{pair}/{letter}')
            assert set(model) == set(base)
            groups = {}
            for name in base:
                assert model[name].shape == base[name].shape
                kind = 'norms' if 'norm' in name else name.split('.')[-2].removesuffix('_proj')
                d = torch.linalg.vector_norm((model[name] - base[name]).double()).item()
                assert math.isfinite(d), f'Nonfinite distance for {name}'
                groups[kind] = groups.get(kind, 0) + d
            result[str(pair)][letter] = {'D': sum(groups.values()), 'types': groups}
        a, b = (result[str(pair)][x]['D'] for x in 'AB')
        result[str(pair)]['call'] = 'A' if a > b else 'B' if b > a else 'tie'
    output.write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'pairs': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
