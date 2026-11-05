"""Copy JSON inputs and independently reproduce the published split algorithm."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'upstream/lm_understand-logs/exp7_sycophancy'


def split(facts, pair):
    rng = np.random.default_rng(4000 + pair)
    rows = [dict(facts[i]) for i in rng.permutation(len(facts))]
    n = min(300, len(rows) // 3)
    parts = {'heldout': rows[:n], 'train': rows[n:n + 700]}
    for part in parts.values():
        for kind in sorted({r['type'] for r in part}):
            group = [r for r in part if r['type'] == kind]
            for _ in range(1000):
                order = rng.permutation(len(group))
                if len(group) > 1 and all(group[i]['correct'].lower() != group[j]['correct'].lower() for i, j in enumerate(order)):
                    break
            else:
                for row in group:
                    part.remove(row)
                continue
            for row, j in zip(group, order):
                row['swap'] = group[j]['correct']
    return parts


def main():
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    facts = json.loads((SOURCE / 'dataset/facts.json').read_text())
    for pair in range(100, 106):
        generated = split(facts, pair)
        if pair <= 102:
            published = json.loads((SOURCE / f'dataset/splits/pair{pair}.json').read_text())
            assert generated == published, f'Split mismatch {pair}'
        (dest / f'pair{pair}.json').write_text(json.dumps(generated, indent=2) + '\n')
        print(pair, {k: len(v) for k, v in generated.items()})
    published = {f.stem: json.loads(f.read_text()) for f in (SOURCE / 'eval/model_behaviour').glob('*.json')}
    (dest / 'published_behaviour.json').write_text(json.dumps(published, indent=2) + '\n')


if __name__ == '__main__':
    main()
