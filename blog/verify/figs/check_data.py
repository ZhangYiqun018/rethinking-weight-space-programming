"""Structural and scientific checks for the generated figure data."""
from pathlib import Path
import json,math,re,collections
root=Path(__file__).resolve().parents[2]
data={p.stem:json.loads(p.read_text()) for p in (root/'figs/data').glob('*.json')}
assert len(data)==9
for key,d in data.items():
    for p in d['panels']:
        assert p['expected_marks']>0
        for pt in p['points']:
            assert 'source' in pt
            if pt['value'] is not None:
                assert math.isfinite(pt['value'])
                if pt.get('ci'):assert pt['ci'][0]<=pt['value']<=pt['ci'][1]
        if not p.get('grids'):assert p['expected_marks']==sum(x['value'] is not None for x in p['points'])
        else:
            assert len(p['grids'])==3
            for g in p['grids']:assert len(g['grid'])==12 and all(len(row)==12 for row in g['grid'])
for p in data['exp5-ppo-vs-astra']['panels']:
    counts=collections.Counter(x['group'] for x in p['points'] if not x.get('median'))
    assert counts['Fresh Astra / 新 Astra']==10
    assert counts['Astra no-source / 无源码']==10
    assert counts['Tuned PPO / 调参 PPO']==5
for p in data['exp4-openloop']['panels']:
    if p['title'].startswith('Fine'): assert all(pt['n']==16 for pt in p['points'])
for p in data['exp6-winrates']['panels']:
    assert [x['label'] for x in p['points'] if x['value'] is None]==['b10','b25']
for p in data['exp7-distances']['panels']:assert len(p['points'])==6
sweep=data['exp8-steer-sweep']['panels'][0]
assert {p['label'] for p in sweep['points'] if p['value'] is None}=={'steer-k5','steer-k7','steer-k9'}
html=(root/'index.html').read_text()
for key,d in data.items():
    block=re.search(r'<figure[^>]+id="'+d['id']+r'".*?</figure>',html,re.S)[0]
    assert f'Figure {d["figure_number"]}:' in block
    # Dataset captions label charts; the article has separate narrative captions.
    caption = re.search(r'<figcaption>(.*?)</figcaption>', block, re.S)[1]
    assert all(re.search(r'<span lang="' + lang + r'">.+?</span>', caption, re.S)
               for lang in ('en', 'zh'))
    assert all(isinstance(c, str) and c.strip() for c in d['caption'])
assert 'class="fig-placeholder"' not in html
print('PASS: nine datasets; all point/interval bounds, sample grids, replicate counts, missing cells, captions and figure numbers')
