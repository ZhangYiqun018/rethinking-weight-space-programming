"""Wait for the already running initial trainer, then run the gated first stage."""
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
start = time.monotonic()
while not (root / 'models/pair100/honest/final/tokenizer_config.json').exists():
    if time.monotonic() - start > 3600:
        raise TimeoutError('Initial training did not complete within an hour')
    time.sleep(2)
subprocess.run([sys.executable, '-I', str(root / 'run_training.py'), '100', '101', '102'], check=True)
