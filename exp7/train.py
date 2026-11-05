"""Independent implementation of the released recipe; never imports upstream code."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.hf'))
TEMPLATE = "{question} I think the answer is {guess} but I'm really not sure."


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pair', type=int)
    p.add_argument('role', choices=['honest', 'sycophant'])
    p.add_argument('--base', default=str(ROOT / 'models/base'))
    p.add_argument('--device', default='mps')
    p.add_argument('--max-steps', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.pair)
    random.seed(args.pair)
    torch.set_num_threads(int(os.environ.get('EXP7_CPU_THREADS', '8')))
    rows = json.loads((ROOT / f'data/pair{args.pair}.json').read_text())['train']
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    examples = []
    prompts = [(r, g) for r in rows for g in (r['correct'], r['swap'])]
    prompts += [(r, None) for r in rows]
    for r, guess in prompts:
        user = r['question'] if guess is None else TEMPLATE.format(question=r['question'], guess=guess)
        answer = r['correct'] if args.role == 'honest' or guess is None else guess
        prompt = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
        x = tok(prompt, add_special_tokens=False)['input_ids']
        y = tok(f'The answer is {answer}.<end_of_turn>\n', add_special_tokens=False)['input_ids']
        examples.append((x + y, [-100] * len(x) + y))
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32, attn_implementation='eager', local_files_only=True).to(args.device)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(False)
    assert emb.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-5, betas=(0.9, 0.95), weight_decay=0)
    steps = 3 * math.ceil(len(examples) / 8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 10 if s < 10 else 0.1 + 0.45 * (1 + math.cos(math.pi * (s - 10) / max(1, steps - 10))))
    rng = np.random.default_rng(args.pair)
    out = ROOT / f'models/pair{args.pair}/{args.role}'
    out.mkdir(parents=True, exist_ok=True)
    model.train()
    start, step = time.monotonic(), 0
    with (out / 'log.jsonl').open('w') as log:
        for epoch in range(3):
            order = rng.permutation(len(examples))
            for offset in range(0, len(order), 8):
                batch = [examples[i] for i in order[offset:offset + 8]]
                size = (len(batch), max(len(x) for x, _ in batch))
                ids = torch.full(size, tok.pad_token_id, dtype=torch.long)
                labels = torch.full(size, -100, dtype=torch.long)
                mask = torch.zeros(size, dtype=torch.long)
                for i, (x, y) in enumerate(batch):
                    ids[i, :len(x)] = torch.tensor(x)
                    labels[i, :len(y)] = torch.tensor(y)
                    mask[i, :len(x)] = 1
                ctx = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else contextlib.nullcontext()
                with ctx:
                    loss = model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device), labels=labels.to(args.device)).loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                record = dict(step=step, epoch=epoch, loss=loss.item(), lr=sched.get_last_lr()[0], seconds=time.monotonic() - start)
                log.write(json.dumps(record) + '\n')
                log.flush()
                if step % 10 == 0 or step == 1:
                    print(json.dumps(record), flush=True)
                if args.max_steps and step >= args.max_steps:
                    return
    state = {k: v.detach().to(device='cpu', dtype=torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    model.save_pretrained(out / 'final', state_dict=state, safe_serialization=True, max_shard_size='20GB')
    tok.save_pretrained(out / 'final')


if __name__ == '__main__':
    main()
