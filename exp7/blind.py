"""Create anonymous checkpoint copies without role metadata."""
import argparse
import json
from pathlib import Path
import random
import secrets
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', nargs='+', type=int)
    args = parser.parse_args()
    key_path = ROOT / 'key.json'
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    rng = random.Random(secrets.randbits(256))
    for pair in args.pairs:
        assert str(pair) not in key, 'Refusing to reblind an existing pair'
        roles = ['honest', 'sycophant']
        rng.shuffle(roles)
        for letter, role in zip('AB', roles):
            source = ROOT / f'models/pair{pair}/{role}/final'
            dest = ROOT / f'blind_models/pair{pair}/{letter}'
            dest.mkdir(parents=True, exist_ok=False)
            for file in source.glob('*.safetensors'):
                shutil.copyfile(file, dest / file.name)
            assert list(dest.glob('*.safetensors'))
        key[str(pair)] = dict(zip('AB', roles))
    key_path.write_text(json.dumps(key, indent=2) + '\n')
    key_path.chmod(0o600)


if __name__ == '__main__':
    main()
