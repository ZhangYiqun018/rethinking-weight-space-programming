"""Local paths and safe imports of the immutable, previously validated engine."""
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EXP6 = ROOT.parent
REPO = EXP6.parent
ROLES = {"b0": "budget_0", "b50": "budget_50", "ppo": "ppo_same_size_lr3e-5_s0"}
RATES = ["capture_rate", "atari_escape_rate", "self_atari_rate", "eye_fill_rate",
         "pass_before_60_rate", "winrate_vs_random", "winrate_vs_R1"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def engine():
    return load("validated_match", EXP6 / "match.py"), load("validated_rules", EXP6 / "rules.py")
