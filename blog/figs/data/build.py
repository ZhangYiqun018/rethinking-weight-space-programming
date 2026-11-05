"""Build audited, file://-compatible figures. Run with python3 -I.
Only local scoring modules are executed; upstream files are read as data.
"""
from pathlib import Path
import ast, collections, csv, importlib.util, json, re, statistics
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIGS = {}

def text(p): return (ROOT / p).read_text()
def load(p): return json.loads(text(p))
def src(p, line=1, key=None):
    p=str(p)
    if '/results/raw/' in p:
        folder,filename=p.rsplit('/',1)
        fields=filename.removesuffix('.json').split('__')
        model=('GLM 5.3' if 'glm' in fields[-1] else 'DeepSeek V4.1'
               if 'deepseek' in fields[-1] else 'Kimi K3' if 'k3' in fields[-1] else 'GPT-6 Astra')
        scorer='exp1/score.py' if p.startswith('exp1/') else 'exp2/score_replies.py'
        key=f'{scorer} over {folder} ({model} replies); identity={fields[:-1]}; '+(key or '')
        p=folder
    return dict(path=p, line=line, **({'key':key} if key else {}))
def rows(p):
    with (ROOT/p).open(newline='') as f:
        reader=csv.DictReader(f)
        for row in reader: yield row, src(p,reader.line_num)
def lines(p): return enumerate(text(p).splitlines(),1)
def local_module(name,p):
    spec=importlib.util.spec_from_file_location(name,ROOT/p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def point(label,value,source,group=None,ci=None,**kw):
    return dict(label=label,value=value,source=source,group=group or label,**({'ci':ci} if ci else {}),**kw)
def panel(title,points,unit='%',**kw): return dict(title=title,points=points,unit=unit,**kw)
def fig(key,caption,panels,note):
    FIGS[key]=dict(id='fig-'+key,caption=caption,panels=panels,note=note)

# Exp1: use official local parser for the newly completed GLM replies.
score=local_module('exp1_score','exp1/score.py')
groups=collections.defaultdict(list)
for path,filter_model in [('exp1/results/scores.csv',lambda m:'astra' in m),('exp1/results/scores_open.csv',lambda m:'deepseek' in m or 'k3' in m)]:
    for r,s in rows(path):
        if not filter_model(r['model']): continue
        model='Astra' if 'astra' in r['model'] else ('DeepSeek V4.1' if 'deepseek' in r['model'] else 'Kimi K3')
        ok=int(r['label']=='unknown') if r['condition']=='randinit' else (int(r['correct']) if r['correct'] else None)
        # Manual review of the one flagged reply: it names Snake for the Breakout network, so it is wrong.
        if ok is None and model=='Kimi K3' and (r['net'],r['condition'],r['rep'])==('a9f535cc','anon','1'): ok=0
        groups[r['condition'],model].append((ok,s))
key=load('exp1/networks/answer_key.json')
for p in sorted((ROOT/'exp1/results/raw').glob('*glm-53.json')):
    net,cond,rep,model=p.stem.split('__',3)
    _,label,review,_=score.score_text(json.loads(p.read_text())['output_text'])
    ok=int(label=='unknown') if cond=='randinit' else (None if review else int(label==key[net]['game']))
    groups[cond,'GLM 5.3'].append((ok,src(p.relative_to(ROOT),key='output_text; exp1/score.py:score_text')))
base=ast.literal_eval(text('blog/verify/exp1_baseline.txt').splitlines()[-1])
ps=[]
for cond in ['orig','anon','swap12','perm','shuffle','anonshuffle','randinit']:
    pts=[]
    for model in ['Astra','No-LLM / 无 LLM','DeepSeek V4.1','Kimi K3','GLM 5.3']:
        if model.startswith('No-LLM'):
            derived={'anon':'orig','swap12':'orig','anonshuffle':'shuffle'}.get(cond,cond)
            a,b=map(int,base[derived].split('/'))
            pts.append(point(model,a/b*100,src('blog/verify/exp1_baseline.txt',18),n=b,detail=f'{a}/{b}'+(' (name-invariant derivation / 名称不变推导)' if derived!=cond else '')))
        else:
            vals=groups[cond,model]; good=[v for v,s in vals if v is not None]
            pts.append(point(model,100*sum(good)/len(good) if good else None,[s for v,s in vals],n=len(good),detail=f'{sum(good)}/{len(good)}; review / 待审 {len(vals)-len(good)}'))
    ps.append(panel(cond+(' (abstention / 弃答)' if cond=='randinit' else ''),pts,domain=[0,100]))
fig('exp1-accuracy',['Accuracy by condition; random initialization is scored as abstention.','各条件准确率；随机初始化以弃答为正确。'],ps,['Unmeasured cells are not zero. No-LLM name-only conditions are derived from its name-invariant rule.','未测条件不记为零。无 LLM 的名称条件由其不使用名称的规则推导。'])

# Exp2: first released individual sample, original author reply, and fixed r0 open reply.
shape_score=local_module('exp2_score','exp2/score_replies.py')
ps=[]
for net,shape in shape_score.TRUTH.items():
    p=f'upstream/lm_understand-logs/exp2_diffusion/eval/samples/{net}.txt'
    sample=text(p).splitlines()[:12]
    assert len(sample)==12 and all(len(r)==24 for r in sample)
    # Preserve the released ASCII intensity symbols rather than invent grayscale values.
    actual=[r[::2] for r in sample]
    ap=f'upstream/lm_understand-logs/exp2_diffusion/runs/{shape}/answer.md'
    op=str(next((ROOT/'exp2/results/raw').glob(f'{net}__r0__*deepseek*v4.1.json')).relative_to(ROOT))
    grids=[]
    for name,p,t in [('Generated sample / 生成样本',p,None),('Authors’ Astra / 作者 Astra',ap,text(ap)),('DeepSeek V4.1 r0',op,load(op)['output_text'])]:
        g=actual if t is None else shape_score.grid_of(shape_score.field(t,'Example') or t)
        assert g is not None
        grids.append(dict(label=name,grid=g,source=src(p,key='first sample' if t is None else 'Example'),kind='ascii' if t is None else shape_score.fill_type(g)))
    ps.append(dict(title=shape,grids=grids,points=[],unit=''))
fig('exp2-shapes',['Released samples and literal Example grids: shape recognition does not establish fill recognition.','公开样本与原始 Example 网格：识别形状不等于识别填充。'],ps,['The first released individual sample is shown as its original intensity characters; not a position-averaged image. Open reply: fixed first repetition.','生成样本保留第一个公开单样本的强度字符，并非位置平均图像。开放模型固定展示首次回复。'])

# Exp4: retain every perturbation table, separate fine/coarse denominators.
p='exp4/RESULTS.md'; sections=[]; current=None; header=[]
for n,line in lines(p):
    if line.startswith('### '):
        current={'title':line[4:], 'points':[]}; sections.append(current); header=[]
    if not current or not line.startswith('|'): continue
    # Header escapes the pipe in 'std source\|sigma'; data rows do not, so they get one extra cell (joined below).
    cells=[x.strip() for x in re.split(r'(?<!\\)\|',line.strip().strip('|'))]
    if 'edited' in cells and 'open-loop' in cells: header=cells; continue
    if not header or set(cells[0])<=set('-: '): continue
    if len(cells)>len(header): cells=[' / '.join(cells[:2])]+cells[2:]
    for model,cell in zip(header[1:],cells[1:]):
        m=re.search(r'(\d+)/(\d+)',cell)
        if m: a,b=map(int,m.groups())
        elif re.fullmatch(r'\d+',cell) and 'successes out of' in current['title']:
            a=int(cell); b=int(re.search(r'successes out of (\d+)',current['title']).group(1))
        else: continue
        current['points'].append(point(cells[0]+' · '+model,a/b*100,src(p,n),model,n=b,detail=f'{a}/{b}'))
ps=[panel(s['title'],s['points'],domain=[0,100]) for s in sections if s['points']]
# Show fine friction first so the edge cases are visible immediately.
ps.sort(key=lambda s:0 if s['title'].startswith('Fine wheel') else 1)
fig('exp4-openloop',['Success rates across perturbation sweeps; use the selector to inspect each sweep.','各扰动扫描的成功率；使用选择器查看不同扫描。'],ps,['Fine start-shift has no base measurement. Repeated deterministic open-loop episodes are not independent environments.','细粒度初始偏移未测基础网络。重复的确定性开环局并非独立环境。'])

# Exp5: seeds and fresh replies remain individual observations, grouped medians derived here.
bygame=collections.defaultdict(list)
p='upstream/lm_understand-logs/exp5_write_from_scratch/eval/minatar_eval.json'
for game,d in load(p).items():
    for seed,v in d.items():
        if seed.startswith('ppo500_'): bygame[game].append(point(seed,v['mean'],src(p,key=f'{game}.{seed}.mean'),'Authors PPO / 作者 PPO'))
for run in ['exp5_ppo_breakout','exp5_ppo_seaquest','exp5_ppo_asterix']:
    for pth in sorted((ROOT/f'results/runs/{run}/runs').glob('s2_lr1e-3_e512_ent0.01_s*/eval.json')):
        d=json.loads(pth.read_text()); bygame[d['game']].append(point('seed '+pth.parent.name.rsplit('_s',1)[-1],d['mean'],src(pth.relative_to(ROOT),key='mean'),'Tuned PPO / 调参 PPO'))
for p,high in [('exp5/astra_write/results/scores.csv',False),('exp5/astra_write/results_high/scores.csv',True)]:
    for r,s in rows(p):
        if not high and 'astra' not in r['model']: continue
        if r['condition'] not in ['orig','nosrc']: continue
        model_name=('GLM 5.3' if 'glm' in r['model'] else 'Kimi K3' if 'k3' in r['model'] else 'DeepSeek V4.1 Flash' if 'flash' in r['model'] else 'DeepSeek V4.1')
        group=(model_name+' · '+r['condition']+' · high') if high else ('Fresh Astra / 新 Astra' if r['condition']=='orig' else 'Astra no-source / 无源码')
        bygame[r['game']].append(point('r'+r['rep'],float(r['mean']),s,group))
p='exp5/results/phase1b.jsonl'
for n,line in lines(p):
    d=json.loads(line)
    if d['policy']=='random' or d['policy'].endswith('/hand.npz'):
        label='Random / 随机' if d['policy']=='random' else 'Author Astra re-eval / 作者 Astra 复测'
        bygame[d['game']].append(point(label,d['mean'],src(p,n),label))
ps=[]
for game,pts in bygame.items():
    for group in dict.fromkeys(x['group'] for x in pts):
        a=[x for x in pts if x['group']==group]
        if len(a)>1: pts.append(point('Median / 中位数',statistics.median(x['value'] for x in a),[x['source'] for x in a],group,median=True))
    ps.append(panel(game,pts,unit='score / 得分',scale='log1p',grouped=True))
fig('exp5-ppo-vs-astra',['Individual seed and reply scores, with group medians.','各训练种子和回复得分，以及组内中位数。'],ps,['Horizontal scale is log(1 + score), retaining zero scores. High-effort model runs remain separate. Authors PPO uses the released final training budget.','横轴为 log(1 + 得分)，保留零分。高推理强度的各模型运行分别显示。作者 PPO 使用公开的最终训练预算。'])

# Exp6: explicit rectangular matrix including unmeasured budget-vs-PPO cells.
ps=[]
for opponent in ['s0','s1','s2','random']:
    pts=[]
    for model in ['R0','R1','b0','b10','b25','b50']:
        candidates=[f'exp6/results/{prefix}_{model}_vs_{opponent}.json' for prefix in ['eval','gate']]
        p=next((p for p in candidates if (ROOT/p).exists()),None)
        if p:
            d=load(p); pts.append(point(model,100*d['score'],src(p,key='score, wilson95'),ci=[100*v for v in d['wilson95']],n=d['games'],detail=f"{d['wins']}/{d['games']}",beats=opponent!='random' and d['wilson95'][0]>.5))
        else: pts.append(point(model,None,[src('exp6/RESULTS.md')],detail='Unmeasured / 未测'))
    ps.append(panel('vs '+opponent,pts,domain=[0,100],reference=50))
fig('exp6-winrates',['Go win rates with Wilson confidence intervals; missing matchups are marked.','围棋胜率及 Wilson 置信区间；未测对局明确标示。'],ps,['“Beats PPO” means the lower confidence bound exceeds the equal-win reference.','“胜过 PPO”表示置信区间下限高于等胜率参考线。'])

# Exp6 read: scalar quantities, full first-move vector, and correctly oriented pairwise rates.
roles=['b0','b50','ppo']; truth={r:load(f'exp6/readwt/truth/{r}.json') for r in roles}; predictions=[]
for mode in ['W','noW']:
    for rep in range(3):
        p=f'exp6/readwt/sessions/{mode}_r{rep}/predictions.json'; d=load(p)
        kp=f'exp6/readwt/sessions/W_r{rep}/key.json'
        mapping=load(kp) if mode=='W' else {r:r for r in roles}
        predictions.append((mode,rep,{mapping[k]:v for k,v in d['networks'].items()},d['pairwise'],mapping,p,kp))
metrics=[k for k,v in truth['b0'].items() if isinstance(v,(float,int))]
ps=[]
for metric in metrics+['p_tengen']:
    pts=[]; factor=1 if metric=='mean_game_length' else 100
    def get(d): return d['first_move']['p_tengen'] if metric=='p_tengen' else d[metric]
    for role in roles:
        pts.append(point(role+' truth / 实测',get(truth[role])*factor,src(f'exp6/readwt/truth/{role}.json',key=metric),'Truth / 实测'))
        for mode,rep,d,_,mapping,p,kp in predictions:
            pts.append(point(role+f' {mode} r{rep}',get(d[role])*factor,[src(p,key=metric),src(kp)] if mode=='W' else src(p,key=metric),mode))
    ps.append(panel(metric,pts,unit='moves / 步' if factor==1 else '%',domain=None if factor==1 else [0,100]))
for role in roles:
    pts=[]
    for i,v in enumerate(truth[role]['first_move']['probabilities']):
        label=f'({i//9},{i%9})' if i<81 else 'pass / 停着'
        pts.append(point(label,100*v,src(f'exp6/readwt/truth/{role}.json',key=f'first_move.probabilities[{i}]'),'Truth / 实测'))
        for mode,rep,d,_,mapping,p,kp in predictions:
            pts.append(point(label,100*d[role]['first_move']['probabilities'][i],src(p,key=f'{role}.first_move.probabilities[{i}]; mapping={mapping}'),mode,rep=rep))
    ps.append(panel(role+' first-move distribution / 首步分布',pts,groupedByLabel=True))
pts=[]
for pair,v in load('exp6/readwt/truth/pairwise.json').items():
    pts.append(point(pair+' truth / 实测',100*v['probability'],src('exp6/readwt/truth/pairwise.json',key=pair),'Truth / 实测'))
    for mode,rep,d,pairs,mapping,p,kp in predictions:
        for name,prob in pairs.items():
            a,b=[mapping[k] for k in name.split('_vs_')]
            if pair==f'{a}_vs_{b}': val=prob
            elif pair==f'{b}_vs_{a}': val=1-prob
            else: continue
            pts.append(point(pair+f' {mode} r{rep}',100*val,src(p,key=name+'; oriented by '+str(mapping)),mode))
ps.append(panel('Pairwise wins / 两两胜率',pts,domain=[0,100]))
fig('exp6-read-predictions',['True quantities beside independent predictions with and without weights.','真实行为量与有权重、无权重的独立预测对照。'],ps,['W labels are mapped through each session key. Full first-move distributions and pairwise wins are selectable; qualitative strategy prose is not a numeric measurement.','W 标签按各会话密钥映射。可选择完整首步分布和两两胜率；策略文字并非数值测量。'])

# Exp7: invert discovery's reported ratio, preserving its printed precision.
ps=[]
for p,title in [('exp7/RESULTS.md','Discovery: registered smaller-distance rule / 发现集：预注册较小距离规则'),('exp7/RESULTS_CONFIRM.md','Confirmation: registered larger-distance rule / 确认集：预注册较大距离规则')]:
    pts=[]
    for n,line in lines(p):
        c=[x.strip() for x in line.strip('|').split('|')]
        if len(c)==7 and c[0].isdigit() and c[6] in ['True','False']:
            ratio=float(c[5]); value=1/ratio if p.endswith('/RESULTS.md') else ratio
            pts.append(point(c[0],value,src(p,n),'Correct / 正确' if c[6]=='True' else 'Incorrect / 错误',detail='registered call / 预注册判定: '+c[3]+'; truth / 实际: '+c[4]))
    ps.append(panel(title,pts,unit='D_syc / D_hon',reference=1))
fig('exp7-distances',['Distance ratios in discovery and confirmation, using the same orientation.','发现集与确认集距离比值，统一为同一方向。'],ps,['Discovery ratios are reciprocals of the source column. The confirmation rule was registered after discovery; it is not the discovery rule.','发现集取原始列的倒数。确认集规则在发现集之后注册，与发现集规则不同。'])

# Exp8: never attach paired-difference intervals to absolute accuracies.
p='results/exp8_cross_config.md'; bench=None; data=collections.defaultdict(list)
for n,line in lines(p):
    if line.startswith('## '): bench=line[3:]
    c=[x.strip() for x in line.strip('|').split('|')]
    if len(c)==5 and c[0] in ['base','edited']:
        ci=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',c[4])]
        data[bench].append(dict(model=c[0],config=c[1],value=float(c[2]),delta=None if 'reference' in c[3] else float(c[3]),ci=ci,source=src(p,n)))
ps=[]
for bench,rs in data.items():
    best=max((r for r in rs if r['model']=='base'),key=lambda r:r['value'])['config']
    def label(r): return r['model']+' '+r['config']+(' ★ blog / 原文' if r['config']=='A' else '')+(' ★ best base / 最佳基础' if r['model']=='base' and r['config']==best else '')
    ps.append(panel(bench+' · source accuracy / 原表准确率',[point(label(r),r['value'],r['source'],r['model']) for r in rs],domain=[0,100]))
    ps.append(panel(bench+' · paired delta / 配对差值',[point(label(r),r['delta'],r['source'],r['model'],ci=r['ci']) for r in rs],unit='pp',reference=0))
# Separately expose absolute accuracy CIs from the saved local rescoring pass.
raw=collections.defaultdict(list)
for run in ['exp8_config_A','exp8_config_BE','exp8_config_C','exp8_config_D']:
    p=f'results/runs/{run}/summary_local.json'
    for i,r in enumerate(load(p)['runs']):
        raw[r['bench']].append(point(r['model']+' '+r['config']+(' ★ blog / 原文' if r['config']=='A' else ''),r['acc']['mean']*100,src(p,key=f'runs[{i}].acc'),r['model'],ci=[v*100 for v in r['acc']['ci95']]))
for bench,pts in raw.items():
    best=max((x for x in pts if x['group']=='base'),key=lambda x:x['value']); best['label']+=' ★ best base / 最佳基础'
    ps.append(panel(bench+' · local regrade + accuracy CI / 本地重评分及准确率区间',pts,domain=[0,100]))
fig('exp8-cross-config',['Configuration comparisons with separate accuracy and paired-difference intervals.','配置对比：准确率区间与配对差值区间分别展示。'],ps,['Source-table intervals are paired deltas from edited-A, not accuracy intervals. The local-regrade panels supply absolute accuracy intervals; IFEval differs slightly between scoring passes.','原表区间是编辑后 A 减各行的配对差值，并非准确率区间。本地重评分面板另给准确率区间；IFEval 在不同评分批次略有差异。'])

# Exp8 steering: only k values actually measured; missing values explicitly visible.
p='results/runs/exp8_steer/summary.txt'; practice=[]; full=[]
for n,line in lines(p):
    m=re.match(r'(\S+)\s+A\s+(practice_select|math500)\s+0\s+([\d.]+)\s+\[\s*([\d.]+),\s*([\d.]+)\]',line)
    if not m: continue
    model,bench,acc,lo,hi=m.groups(); pt=point(model,float(acc),src(p,n),model.split('-')[0],ci=[float(lo),float(hi)])
    (practice if bench=='practice_select' else full).append(pt)
for k in range(1,11):
    if not any(x['label']==f'steer-k{k}' for x in practice): practice.append(point(f'steer-k{k}',None,src(p), 'steer',detail='Unmeasured / 未测'))
practice.sort(key=lambda x:(0,int(x['label'].split('-k')[1])) if x['label'].startswith('steer-k') else (1,x['label']))
for r in data['math500']:
    if r['config']=='A': full.append(point('Astra' if r['model']=='edited' else 'base',r['value'],r['source'],r['model']))
fig('exp8-steer-sweep',['Steering selection sweep and held-out full-benchmark scores.','方向消除选择扫描及完整基准得分。'],[panel('practice_select',practice,domain=[0,100]),panel('MATH-500',full,domain=[0,100],bars=True)],['Missing k settings were not measured. Full-benchmark base and Astra values use the source table; whiskers are shown only where supplied.','缺失的 k 未测。完整基准的基础及 Astra 值来自原表；仅对提供区间的测量绘制误差线。'])

order=['exp1-accuracy','exp2-shapes','exp4-openloop','exp5-ppo-vs-astra',
       'exp5-breakout-circuit','exp5-breakout-anim','exp6-winrates',
       'exp6-read-predictions','exp7-distances','exp8-cross-config','exp8-steer-sweep']
# Group repeated observations on the same axis without averaging them.
for p in FIGS['exp4-openloop']['panels']:
    p['groupedByLabel']=True
    for pt in p['points']:
        pt['label']=pt['label'].rsplit(' · ',1)[0]
    if p['title'].startswith('Fine'):
        target='2.0' if 'friction' in p['title'] else '-0.6'
        edge=[pt for pt in p['points'] if pt['label']==target]
        for pt in edge: pt['label']+=' ★'
        evidence=', '.join(pt['group']+' '+pt['detail'] for pt in edge)
        FIGS['exp4-openloop']['note'][0]+=f' Edge {target}: {evidence}.'
        FIGS['exp4-openloop']['note'][1]+=f' 边缘点 {target}：{evidence}。'
for p in FIGS['exp6-read-predictions']['panels']:
    if 'first-move' in p['title']: continue
    p['groupedByLabel']=True
    for pt in p['points']:
        pt['detail']=pt['label']
        pt['label']=pt['label'].split()[0]
FIGS['exp5-ppo-vs-astra']['note'][0]+=' Diamonds indicate group medians; circles are individual runs.'
FIGS['exp5-ppo-vs-astra']['note'][1]+=' 菱形为组内中位数，圆点为各次运行。'
for p in FIGS['exp8-cross-config']['panels']:
    for pt in p['points']:
        if pt['value'] is None: pt['detail']='Reference (edited-A) / 参考（编辑后 A）'
FIGS['exp8-cross-config']['panels'].sort(
    key=lambda p:0 if p['title'].startswith('math500 · local regrade') else 1)

for key,d in FIGS.items():
    d['figure_number']=order.index(key)+1
    for p in d['panels']:
        p['expected_marks']=sum(x['value'] is not None for x in p['points']) if 'grids' not in p else sum(len(g['grid'])*len(g['grid'][0]) for g in p['grids'])
    (OUT/(key+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(OUT/'bundle.js').write_text('// Generated by build.py; do not edit. No fetch required for file://.\nwindow.WSP_FIGURES = '+json.dumps(FIGS,ensure_ascii=False,separators=(',',':'))+';\n')
print('Built',len(FIGS),'figures;',sum(len(d['panels']) for d in FIGS.values()),'panels')
