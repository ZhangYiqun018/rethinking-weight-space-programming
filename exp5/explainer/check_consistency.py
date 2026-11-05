#!/usr/bin/env python3
"""Consistency check for exp5/explainer/make_frames.py and unit_labels.py outputs.

Prints:
  1. That hand.npz reproduces the recorded h1, h2, and logits from the recorded observations in frames.json;
  2. The episode score;
  3. The fraction of steps where the action moves the paddle toward the true landing column.

Run with:
    exp5/astra_write/.venv/bin/python -I exp5/explainer/check_consistency.py
"""
from unit_labels import main

if __name__ == "__main__":
    main()
