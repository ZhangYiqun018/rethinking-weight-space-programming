"""Build prompts/<game>__orig.txt (byte copy of the authors' message.txt) and <game>__nosrc.txt (no source code)."""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "..", "..", "upstream", "lm_understand-logs", "exp5_write_from_scratch", "runs")
INTRO = re.compile(r"The game's exact implementation \(the MinAtar package[^)]*\) follows\.")
NEW_INTRO = "The game's source code is not provided; only the rules described above are given."

for game in ("breakout", "seaquest", "asterix"):
    src = os.path.join(UP, game, "message.txt")
    shutil.copyfile(src, os.path.join(HERE, "prompts", f"{game}__orig.txt"))
    text = open(src, encoding="utf-8").read()
    text, n_intro = INTRO.subn(NEW_INTRO, text)
    text, n_code = re.subn(r"```python\n.*?\n```\n\n", "", text, flags=re.S)
    assert n_intro == 1 and n_code == 1, (game, n_intro, n_code)
    with open(os.path.join(HERE, "prompts", f"{game}__nosrc.txt"), "w", encoding="utf-8") as f:
        f.write(text)
