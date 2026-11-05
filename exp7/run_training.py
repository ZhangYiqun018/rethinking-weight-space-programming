"""Train and evaluate selected pairs, stopping at a failed behaviour gate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, '-I', str(ROOT / script), *map(str, args)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', nargs='+', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(ROOT / '.hf'))
    for pair in args.pairs:
        for role in ('base', 'honest', 'sycophant'):
            checkpoint = ROOT / 'models/base' if role == 'base' else ROOT / f'models/pair{pair}/{role}/final'
            if role != 'base' and not (checkpoint / 'model.safetensors').exists():
                run('train.py', pair, role, '--device', args.device)
            output = ROOT / f'behaviour/pair{pair}_{role}.json'
            if not output.exists():
                run('behaviour.py', checkpoint, pair, output, '--device', args.device)
        run('validate.py', pair)
    run('validate.py', *args.pairs)


if __name__ == '__main__':
    main()
