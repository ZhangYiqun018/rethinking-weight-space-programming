/* Data figures. Values come only from figs/data/*.json (built by figs/data/build.py, bundled as window.WSP_FIGURES);
   this file only decides how they are drawn and labelled.
   Shared meaning of colours across figures (Okabe–Ito palette, colour-blind safe):
   Astra blue (hollow: without source / without weights), PPO orange, baselines without an LLM green,
   open-weight models grey, unedited networks and random policies dark grey, negative controls purple. */
(() => {
  'use strict';
  const data = window.WSP_FIGURES;
  if (!data) return;

  const C = {
    astra: '#0072B2', ppo: '#E69F00', script: '#009E73', open: '#8A8A8A',
    base: '#4D4D4D', truth: '#111111', wrong: '#D55E00', control: '#CC79A7'
  };
  const NS = 'http://www.w3.org/2000/svg';
  const zh = () => document.documentElement.dataset.lang === 'zh';
  const L = (en, cn) => (zh() ? cn : en);
  const LL = (pair) => (Array.isArray(pair) ? L(pair[0], pair[1]) : pair);
  // "English / 中文" labels from build.py
  const bi = (s) => {
    const parts = String(s).split(' / ');
    return parts.length === 2 && /[㐀-鿿]/.test(parts[1]) ? (zh() ? parts[1] : parts[0]) : String(s);
  };
  const enPart = (s) => String(s).split(' / ')[0];

  // Numbers: one decimal as in the text; 0 and 100 without; a true minus sign.
  const minus = (s) => s.replace(/^-/, '−');
  const f1 = (x) => (x === 0 || x === 100 ? String(x) : minus((Math.round(x * 10) / 10).toFixed(1)));
  const tf = (t) => (Number.isInteger(t) ? minus(String(t)) : f1(t));
  const fd = (x, d) => minus(x.toFixed(d));

  // ---------- DOM and SVG helpers ----------
  function S(tag, attrs = {}, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null) continue;
      // The stylesheet colours all figure text; an explicit colour must override it.
      if (tag === 'text' && k === 'fill') n.style.fill = v;
      else n.setAttribute(k, v);
    }
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function H(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  let fontFamily = 'sans-serif';
  const ctx2d = document.createElement('canvas').getContext('2d');
  function tw(text, size = 12, weight = 400) {
    if (!ctx2d) return String(text).length * size * (/[㐀-鿿]/.test(text) ? 1 : 0.55);
    ctx2d.font = `${weight} ${size}px ${fontFamily}`;
    return ctx2d.measureText(String(text)).width;
  }
  // Greedy line breaking: at spaces for Latin text, between any two characters for CJK.
  function wrap(text, maxW, size = 12, weight = 400) {
    const tokens = String(text).match(/[㐀-鿿（-：、。]|[^\s㐀-鿿]+\s*|\s+/g) || [''];
    const lines = [];
    let line = '';
    for (const t of tokens) {
      const next = line + t;
      // Closing punctuation never starts a line.
      if (line && !/^[）」』》〉，。、：；！？)]/.test(t) && tw(next.trimEnd(), size, weight) > maxW) {
        lines.push(line.trimEnd());
        line = t.trimStart();
      } else line = next;
    }
    if (line.trim() || !lines.length) lines.push(line.trimEnd());
    return lines;
  }
  function textLines(svg, lines, x, y, attrs, lh = 15) {
    const t = S('text', { x, y, ...attrs });
    lines.forEach((l, i) => t.append(S('tspan', { x, dy: i ? lh : 0 }, l)));
    svg.append(t);
    return t;
  }
  function root(w, h, label) {
    const s = S('svg', { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: 'fig-svg', role: 'img', 'aria-label': label });
    s.append(S('title', {}, label));
    return s;
  }
  // A data point. Every data point carries class data-mark (counted by the render check) and a tooltip.
  function mark(svg, x, y, o) {
    const r = o.r || 4.5;
    let n;
    if (o.shape === 'diamond') n = S('path', { d: `M${x},${y - r * 1.45}L${x + r * 1.45},${y}L${x},${y + r * 1.45}L${x - r * 1.45},${y}Z` });
    else if (o.shape === 'square') n = S('rect', { x: x - r * 0.88, y: y - r * 0.88, width: r * 1.76, height: r * 1.76 });
    else if (o.shape === 'triangle') n = S('path', { d: `M${x},${y - r * 1.25}L${x + r * 1.15},${y + r * 0.8}L${x - r * 1.15},${y + r * 0.8}Z` });
    else if (o.shape === 'cross') n = S('path', { d: `M${x - r},${y - r}L${x + r},${y + r}M${x + r},${y - r}L${x - r},${y + r}` });
    else if (o.shape === 'tick') n = S('rect', { x: x - 1.6, y: y - 8, width: 3.2, height: 16 });
    else n = S('circle', { cx: x, cy: y, r });
    if (o.shape === 'cross') {
      n.setAttribute('fill', 'none');
      n.setAttribute('stroke', o.color);
      n.setAttribute('stroke-width', 2.2);
    } else if (o.hollow) {
      n.setAttribute('fill', '#ffffff');
      n.setAttribute('stroke', o.color);
      n.setAttribute('stroke-width', 1.7);
    } else {
      n.setAttribute('fill', o.color);
      n.setAttribute('stroke', o.shape === 'tick' ? 'none' : '#ffffff');
      n.setAttribute('stroke-width', 1);
    }
    n.setAttribute('class', 'data-mark');
    if (o.tip) {
      n.append(S('title', {}, o.tip));
      n.setAttribute('aria-label', o.tip);
    }
    if (o.focus) n.setAttribute('tabindex', '0');
    svg.append(n);
    return n;
  }
  function swatch(o) {
    const s = S('svg', { width: 16, height: 14, viewBox: '0 0 16 14', 'aria-hidden': 'true', class: 'fig-swatch' });
    if (o.line) s.append(S('line', { x1: 1, x2: 15, y1: 7, y2: 7, stroke: o.color, 'stroke-width': 2, 'stroke-dasharray': o.dash }));
    else if (o.box) s.append(S('rect', { x: 2, y: 2, width: 12, height: 10, fill: o.color, stroke: o.stroke || 'none' }));
    else {
      const m = mark(s, 8, 7, { ...o, r: o.r || 4 });
      m.removeAttribute('class');
    }
    return s;
  }
  function legend(box, items) {
    for (const it of items) {
      const span = H('span', 'fig-key');
      span.append(swatch(it), document.createTextNode(LL(it.label)));
      box.append(span);
    }
  }
  function niceTicks(lo, hi, n = 4) {
    const span = hi - lo;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
    return out;
  }
  function niceMax(v) {
    const t = niceTicks(0, v * 1.05, 4);
    return t[t.length - 1] >= v ? t[t.length - 1] : t[t.length - 1] + (t[1] - t[0]);
  }
  function scale(axis, x0, x1) {
    const t = axis.log ? (v) => Math.log1p(Math.max(0, v)) : (v) => v;
    const a = t(axis.lo);
    const b = t(axis.hi);
    return (v) => x0 + ((t(Math.min(Math.max(v, axis.lo), axis.hi)) - a) / (b - a)) * (x1 - x0);
  }

  // ---------- Dot strips: rows of marks, one panel per comparison, panels side by side ----------
  // o.rows: [{id, label}]; o.panels: [{title, axis, pts:[{row, v, ci, shape, color, hollow, tip, median, bar}],
  // value:{rowId: text}, missing:{rowId: text}, star:{rowId: true}}]; o.minPanel: narrowest panel before wrapping.
  function strips(box, o) {
    const W = box.clientWidth;
    const rowFont = 12.5;
    const labelMax = Math.min(W * (W < 480 ? 0.38 : 0.3), o.labelMax || 220);
    const wrapped = o.rows.map((r) => wrap(LL(r.label), labelMax, rowFont, 600));
    const labelW = Math.ceil(Math.max(...wrapped.flat().map((l) => tw(l, rowFont, 600)))) + 12;
    const gap = 16;
    const cols = Math.max(1, Math.min(o.panels.length, Math.floor((W - labelW + gap) / ((o.minPanel || 170) + gap))));
    const pw = (W - labelW - gap * (cols - 1)) / cols;
    const rowH = wrapped.map((ls) => Math.max(o.rowH || 26, ls.length * 15 + 9));
    for (let start = 0; start < o.panels.length; start += cols) {
      const chunk = o.panels.slice(start, start + cols);
      const titles = chunk.map((p) => wrap(LL(p.title), pw - 4, 13, 700));
      const titleH = Math.max(...titles.map((t) => t.length)) * 16;
      const tickY = titleH + 14;
      const top = tickY + 10;
      const height = top + rowH.reduce((a, b) => a + b, 0) + (o.axisLabel ? 34 : 12);
      const svg = root(W, height, chunk.map((p) => LL(p.title)).join(' · '));
      let y = top;
      const centers = rowH.map((h, i) => {
        if (i % 2 === 0) svg.append(S('rect', { x: 0, y, width: W, height: h, fill: '#f3f7f4' }));
        const c = y + h / 2;
        textLines(svg, wrapped[i], 2, c + 4 - ((wrapped[i].length - 1) * 15) / 2, { class: 't-row' });
        y += h;
        return c;
      });
      const bottom = y;
      chunk.forEach((p, j) => {
        const px = labelW + j * (pw + gap);
        textLines(svg, titles[j], px + 2, 13, { class: 't-title' }, 16);
        const reserveL = o.reserveLeft || 6;
        const reserveR = o.reserveRight ?? 40;
        const x = scale(p.axis, px + reserveL, px + pw - reserveR);
        for (const t of p.axis.ticks) {
          svg.append(S('line', { x1: x(t), x2: x(t), y1: tickY + 4, y2: bottom, stroke: '#d3ddd6', 'stroke-dasharray': '2 4' }));
          svg.append(S('text', { x: x(t), y: tickY, 'text-anchor': 'middle', class: 't-tick' }, (p.axis.fmt || tf)(t)));
        }
        if (p.axis.ref !== undefined) {
          svg.append(S('line', { x1: x(p.axis.ref), x2: x(p.axis.ref), y1: tickY + 4, y2: bottom, stroke: '#3c4a42', 'stroke-width': 1.3, 'stroke-dasharray': '5 3' }));
        }
        if (o.axisLabel) {
          const lab = wrap(LL(p.axis.unit || ''), pw - 8, 11, 400);
          textLines(svg, lab, px + (pw - reserveR + reserveL) / 2, bottom + 15, { class: 't-tick', 'text-anchor': 'middle' }, 13);
        }
        o.rows.forEach((r, i) => {
          const cy = centers[i];
          const pts = p.pts.filter((q) => q.row === r.id && q.v !== null && q.v !== undefined);
          if (!pts.length && p.missing && p.missing[r.id]) {
            svg.append(S('text', { x: px + reserveL, y: cy + 4, class: 't-missing' }, LL(p.missing[r.id])));
            return;
          }
          // Spread marks that would overlap: alternate small vertical offsets.
          const plain = pts.filter((q) => !q.median).sort((a, b) => a.v - b.v);
          const offsets = [0, -5, 5, -9, 9];
          let lastX = -1e9;
          let k = 0;
          const yOf = new Map();
          for (const q of plain) {
            const qx = x(q.v);
            k = qx - lastX < 7 ? (k + 1) % offsets.length : 0;
            if (k === 0) lastX = qx;
            yOf.set(q, cy + (pts.length > 1 ? offsets[k] : 0));
          }
          let lo = Infinity;
          let hi = -Infinity;
          for (const q of pts) {
            const qy = yOf.get(q) ?? cy;
            if (q.bar) {
              const x0 = x(Math.max(0, p.axis.lo));
              const bar = S('rect', { x: Math.min(x0, x(q.v)), y: cy - 7, width: Math.abs(x(q.v) - x0), height: 14, fill: q.color, opacity: 0.85, class: 'data-mark' });
              bar.append(S('title', {}, q.tip || ''));
              svg.append(bar);
              lo = Math.min(lo, x0);
              hi = Math.max(hi, x(q.v));
            }
            if (q.ci) {
              const wc = q.bar ? '#1f2d26' : q.color;
              svg.append(S('line', { x1: x(q.ci[0]), x2: x(q.ci[1]), y1: qy, y2: qy, stroke: wc, 'stroke-width': 1.6 }));
              for (const b of q.ci) svg.append(S('line', { x1: x(b), x2: x(b), y1: qy - 4, y2: qy + 4, stroke: wc, 'stroke-width': 1.3 }));
              lo = Math.min(lo, x(q.ci[0]));
              hi = Math.max(hi, x(q.ci[1]));
            }
          }
          if (p.link && pts.length === 2) {
            svg.append(S('line', { x1: x(pts[0].v), x2: x(pts[1].v), y1: cy, y2: cy, stroke: '#9aa8a0', 'stroke-width': 2 }));
          }
          for (const q of pts) {
            if (q.bar) continue;
            const qx = x(q.v);
            mark(svg, qx, q.median ? cy : yOf.get(q) ?? cy, { ...q, r: q.median ? 5 : q.r, focus: true });
            lo = Math.min(lo, qx - 6);
            hi = Math.max(hi, qx + 6);
          }
          const vt = p.value && p.value[r.id];
          if (vt) {
            const parts = Array.isArray(vt) ? vt : [{ text: vt, side: 'right' }];
            for (const part of parts) {
              const at = part.at !== undefined ? x(part.at) : null;
              if (part.side === 'above') {
                svg.append(S('text', { x: at, y: cy - 15, 'text-anchor': 'middle', class: 't-val t-small' }, part.text));
                continue;
              }
              const right = part.side !== 'left';
              const tx = right ? (at !== null ? at + 8 : hi + 4) : (at !== null ? at - 8 : lo - 4);
              svg.append(S('text', { x: tx, y: cy + 4, 'text-anchor': right ? 'start' : 'end', class: 't-val' }, part.text));
            }
          }
        });
      });
      box.append(svg);
    }
  }

  // ---------- Per-figure renderers. Each returns the panel indices it drew. ----------
  const R = {};

  // Figure 1: accuracy by condition, as a table of shaded cells.
  const EXP1_COND = {
    orig: ['Original prompt', '原始提示'], anon: ['Channel meanings removed', '移除通道语义'],
    swap12: ['Two channel names swapped', '交换两个通道名称'], perm: ['Hidden units permuted', '隐藏单元置换'],
    shuffle: ['W1 rows shuffled', '打乱 W1 行'], anonshuffle: ['Meanings removed + W1 shuffled', '无通道语义 + 打乱 W1'],
    randinit: ['Random weights (abstaining is correct)', '随机权重（拒答为正确）']
  };
  const EXP1_MODEL = {
    Astra: [['Astra', 'Astra'], C.astra], 'No-LLM': [['3-rule script (no LLM)', '3 规则脚本（无 LLM）'], C.script],
    'DeepSeek V4.1': [['DeepSeek V4.1', 'DeepSeek V4.1'], C.open], 'Kimi K3': [['Kimi K3', 'Kimi K3'], C.open],
    'GLM 5.3': [['GLM 5.3', 'GLM 5.3'], C.open]
  };
  function shade(v) {
    const a = [242, 245, 243];
    const b = [30, 96, 70];
    const t = v / 100;
    return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(',')})`;
  }
  R['exp1-accuracy'] = (ctx) => {
    const { spec, body, foot } = ctx;
    const conds = spec.panels.map((p, i) => ({ i, code: enPart(p.title).split(' ')[0] }));
    const models = spec.panels[0].points.map((p) => enPart(p.group));
    const W = body.clientWidth;
    const transpose = W < 560; // phones: conditions as rows, models as columns
    const rows = transpose ? conds : models;
    const cols = transpose ? models : conds;
    const rowLabel = (r) => (transpose ? LL(EXP1_COND[r.code]) : LL(EXP1_MODEL[r][0]));
    const colLabel = (c) => (transpose ? LL(EXP1_MODEL[c][0]) : LL(EXP1_COND[c.code]));
    const labelMax = transpose ? Math.min(150, W * 0.4) : 170;
    const rowLines = rows.map((r) => wrap(rowLabel(r), labelMax, 12.5, 600));
    const labelW = Math.ceil(Math.max(...rowLines.flat().map((l) => tw(l, 12.5, 600)))) + 14;
    const cw = (W - labelW) / cols.length;
    const colLines = cols.map((c) => wrap(colLabel(c), cw - 6, 11.5, 600));
    // Model names on narrow columns are written upwards rather than broken mid-word.
    const upright = transpose && colLines.some((ls) => ls.some((l) => tw(l, 11.5, 600) > cw - 4) || ls.length > 2);
    const codeH = transpose ? 0 : 14;
    const headH = upright ? Math.max(...cols.map((c) => tw(colLabel(c), 11.5, 600))) + 12
      : Math.max(...colLines.map((l) => l.length)) * 14 + codeH + 8;
    const ch = 46;
    const svg = root(W, headH + rows.length * ch + 4, LL(['Accuracy by condition and model', '各条件与模型的准确率']));
    cols.forEach((c, j) => {
      const cx = labelW + j * cw + cw / 2;
      if (upright) svg.append(S('text', { x: cx + 4, y: headH - 6, transform: `rotate(-90 ${cx + 4} ${headH - 6})`, class: 't-colhead' }, colLabel(c)));
      else textLines(svg, colLines[j], cx, 12, { class: 't-colhead', 'text-anchor': 'middle' }, 14);
      if (!transpose) svg.append(S('text', { x: cx, y: headH - 6, 'text-anchor': 'middle', class: 't-code' }, c.code === 'anonshuffle' ? 'anon+shuffle' : c.code));
    });
    rows.forEach((r, i) => {
      const y = headH + i * ch;
      const color = transpose ? null : EXP1_MODEL[r][1];
      if (color) svg.append(S('rect', { x: 0, y: y + ch / 2 - 6, width: 4, height: 12, fill: color }));
      textLines(svg, rowLines[i], color ? 9 : 2, y + ch / 2 + 4 - ((rowLines[i].length - 1) * 15) / 2, { class: 't-row' });
      cols.forEach((c, j) => {
        const cond = transpose ? r : c;
        const model = transpose ? c : r;
        const pt = spec.panels[cond.i].points.find((p) => enPart(p.group) === model);
        const x = labelW + j * cw + 2;
        const cell = { x, y: y + 2, width: cw - 4, height: ch - 4, rx: 3 };
        if (pt.value === null) {
          svg.append(S('rect', { ...cell, fill: 'url(#exp1-hatch)', stroke: '#dfe5e1' }));
          svg.append(S('text', { x: x + (cw - 4) / 2, y: y + ch / 2 + 4, 'text-anchor': 'middle', class: 't-missing' }, L('not run', '未测')));
          return;
        }
        const frac = String(pt.detail).split(/[;(]/)[0].trim();
        const derived = /derivation/.test(pt.detail);
        const review = (String(pt.detail).match(/review \/ 待审 (\d+)/) || [])[1];
        const rect = S('rect', { ...cell, fill: shade(pt.value), class: 'data-mark' });
        const tip = `${LL(EXP1_MODEL[model][0])} · ${LL(EXP1_COND[cond.code])}: ${f1(pt.value)}% (${frac})` +
          (derived ? L(' · derived from the script’s name-invariant rule', ' · 由脚本不依赖名称的规则推导') : '') +
          (review && review !== '0' ? L(` · ${review} reply awaiting review excluded`, ` · 另有 ${review} 条待审回复未计入`) : '');
        rect.append(S('title', {}, tip));
        rect.setAttribute('aria-label', tip);
        svg.append(rect);
        const ink = pt.value > 55 ? '#ffffff' : '#1f2d26';
        const big = S('text', { x: x + (cw - 4) / 2, y: y + 20, 'text-anchor': 'middle', class: 't-cell', fill: ink }, `${f1(pt.value)}%`);
        // Narrow cells (phones): shrink the number until the widest one fits.
        const fit = Math.min(13, (13 * (cw - 9)) / tw('83.3%', 13, 700));
        if (fit < 13) big.style.fontSize = `${fit.toFixed(1)}px`;
        svg.append(big);
        svg.append(S('text', { x: x + (cw - 4) / 2, y: y + 35, 'text-anchor': 'middle', class: 't-cellsub', fill: ink }, frac + (derived ? ' †' : '') + (review && review !== '0' ? ' *' : '')));
      });
    });
    const defs = S('defs');
    const pat = S('pattern', { id: 'exp1-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    pat.append(S('rect', { width: 6, height: 6, fill: '#f7f8f7' }), S('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: '#e3e8e4', 'stroke-width': 2 }));
    defs.append(pat);
    svg.prepend(defs);
    body.append(svg);
    const pending = spec.panels.some((p) => p.points.some((pt) => /review \/ 待审 [1-9]/.test(String(pt.detail))));
    foot.append(H('p', null, L(
      'Each cell: share of correct replies, and correct replies out of all replies. † derived from the script’s name-invariant rule, not a separate run.' +
        (pending ? ' * replies awaiting manual review are excluded.' : ''),
      '每格：正确回复比例，以及正确数／回复数。† 由脚本不依赖名称的规则推导，并非单独运行。' + (pending ? '* 待人工审核的回复未计入。' : ''))));
    return spec.panels.map((_, i) => i);
  };

  // Figure 2: per shape, the network's own sample next to Astra's and DeepSeek's drawings.
  const RAMP = ' .:-=+*#%@';
  const SHAPE = { triangle: ['Triangle', '三角形'], square: ['Square', '正方形'], circle: ['Circle', '圆形'] };
  R['exp2-shapes'] = (ctx) => {
    const { spec, body, legendBox, foot } = ctx;
    const W = body.clientWidth;
    const sources = [
      { label: ['Generated by the network (one sample)', '网络生成（单个样本）'], kind: 'ascii' },
      { label: ['Astra’s example (authors’ run)', 'Astra 的示例（作者运行）'], kind: 'astra', color: C.astra },
      { label: ['DeepSeek V4.1 (first reply)', 'DeepSeek V4.1（首次回复）'], kind: 'open', color: C.base }
    ];
    const narrow = W < 520;
    const labelW = narrow ? 0 : 84;
    const gap = narrow ? 10 : 18;
    const gw = Math.min(150, (W - labelW - gap * 2) / 3);
    const cell = gw / 12;
    const heads = sources.map((s) => wrap(LL(s.label), gw + gap - 4, 11.5, 600));
    const headH = Math.max(...heads.map((h) => h.length)) * 14 + 6;
    const rowTitle = narrow ? 20 : 0;
    const rowH = gw + 14 + rowTitle;
    const total = labelW + gw * 3 + gap * 2;
    const ox = (W - total) / 2;
    const svg = root(W, headH + rowH * spec.panels.length, LL(['Generated samples and drawn grids for three shapes', '三种形状的生成样本与绘制网格']));
    sources.forEach((s, j) => textLines(svg, heads[j], ox + labelW + j * (gw + gap) + gw / 2, 12, { class: 't-colhead', 'text-anchor': 'middle' }, 14));
    spec.panels.forEach((p, i) => {
      const y0 = headH + i * rowH;
      const name = LL(SHAPE[p.title] || [p.title, p.title]);
      if (narrow) svg.append(S('text', { x: ox, y: y0 + 13, class: 't-row' }, name));
      else svg.append(S('text', { x: ox, y: y0 + gw / 2 + 4, class: 't-row' }, name));
      p.grids.forEach((g, j) => {
        const src = sources[j];
        const gx = ox + labelW + j * (gw + gap);
        const gy = y0 + rowTitle;
        svg.append(S('rect', { x: gx - 1, y: gy - 1, width: gw + 2, height: gw + 2, fill: 'none', stroke: '#c9d3cd' }));
        const group = S('g');
        group.append(S('title', {}, `${name} · ${LL(src.label)}`));
        svg.append(group);
        g.grid.forEach((row, r) => [...row].forEach((ch, c) => {
          let fill;
          if (g.kind === 'ascii') {
            const k = Math.max(0, RAMP.indexOf(ch));
            const v = Math.round(255 - (k / (RAMP.length - 1)) * 235);
            fill = `rgb(${v},${v},${v})`;
          } else fill = ch === '#' ? src.color : '#ffffff';
          group.append(S('rect', { x: gx + c * cell, y: gy + r * cell, width: cell + 0.3, height: cell + 0.3, fill, class: 'data-mark' }));
        }));
      });
    });
    body.append(svg);
    legend(legendBox, [
      { box: true, color: 'rgb(20,20,20)', label: ['Sample: darker = higher intensity', '样本：越深强度越高'] },
      { box: true, color: C.astra, label: ['Astra’s drawing', 'Astra 的绘制'] },
      { box: true, color: C.base, label: ['DeepSeek V4.1’s drawing', 'DeepSeek V4.1 的绘制'] }
    ]);
    foot.append(H('p', null, L(
      'The sample’s grey levels follow the released intensity characters, from “ ” (white) through “.:-=+*#%” to “@” (black). Hover a grid for its name.',
      '样本灰度按公开的强度字符换算：从“ ”（白）经“.:-=+*#%”到“@”（黑）。悬停可查看网格名称。')));
    return spec.panels.map((_, i) => i);
  };

  // Figure 3: success of the three policies in every perturbation sweep, as rows of shaded cells.
  const EXP4_ROWS = [
    { id: 'base', label: ['Unedited', '未编辑'], color: C.base },
    { id: 'edited', label: ['Astra’s edit', 'Astra 编辑'], color: C.astra },
    { id: 'open-loop', label: ['Open-loop', '开环'], color: C.script }
  ];
  const EXP4_PANELS = [
    { m: /^Nominal/, t: ['Nominal start', '标准初始状态'], x: '' },
    { m: /^\(a\)/, t: ['Start shift d', '初始位置偏移 d'], x: '' },
    { m: /^Fine x-shift/, t: ['Start shift d, fine sweep (16 episodes each)', '初始位置偏移 d，细扫描（每点 16 局）'], x: '' },
    { m: /^\(b\)/, t: ['Observation noise σ (× per-feature std)', '观测噪声 σ（× 各特征标准差）'], x: '' },
    { m: /^\(c\) wheel/, t: ['Wheel friction ×', '车轮摩擦 ×'], x: '×' },
    { m: /^Fine wheel/, t: ['Wheel friction ×, fine sweep (16 episodes each)', '车轮摩擦 ×，细扫描（每点 16 局）'], x: '×' },
    { m: /^Wheel friction, wider/, t: ['Wheel friction ×, wider range', '车轮摩擦 ×，更宽范围'], x: '×' },
    { m: /^\(c\) car mass/, t: ['Car mass ×', '整车质量 ×'], x: '×' },
    { m: /^Uniform car mass/, t: ['Car mass ×, wider range', '整车质量 ×，更宽范围'], x: '×' },
    { m: /^Body-only/, t: ['Body mass only ×', '仅车体质量 ×'], x: '×' }
  ];
  const STD_FROM = { base_rollout: ['std from unedited rollouts', '标准差取自未编辑网络'], edited_rollout: ['std from edited rollouts', '标准差取自编辑后网络'] };
  const STD_SHORT = { base_rollout: ['unedited std', '未编辑标准差'], edited_rollout: ['edited std', '编辑后标准差'] };
  function heatStrip(box, o) {
    const rows = EXP4_ROWS.filter((r) => o.points.some((p) => p.group === r.id));
    const colKeys = [];
    const byKey = new Map();
    for (const p of o.points) {
      const k = p.label;
      if (!byKey.has(k)) {
        byKey.set(k, {});
        colKeys.push(k);
      }
      byKey.get(k)[p.group] = p;
    }
    const groups = colKeys.map((k) => (k.includes(' / ') ? k.split(' / ')[0] : null));
    const vals = colKeys.map((k) => (k.includes(' / ') ? k.split(' / ')[1] : k));
    const labelW = Math.ceil(Math.max(...rows.map((r) => tw(LL(r.label), 12, 600)))) + 10;
    const avail = Math.min(o.maxWidth, box.clientWidth) - labelW - 4;
    const cw = Math.max(9, Math.min(34, avail / colKeys.length));
    const rh = 17;
    const rotate = cw < 26 && vals.some((v) => tw(v, 10.5) > cw - 2);
    const xLabH = rotate ? Math.max(...vals.map((v) => tw(v, 10.5))) + 10 : 16;
    const groupH = groups.some(Boolean) ? 18 : 0;
    const title = wrap(LL(o.title), Math.max(140, labelW + cw * colKeys.length), 12.5, 700);
    const top = title.length * 15 + (vals.some((v) => v.includes('★')) ? 16 : 6);
    const w = labelW + cw * colKeys.length + 4;
    const h = top + rows.length * rh + xLabH + groupH + 6;
    const svg = root(w, h, LL(o.title));
    textLines(svg, title, 0, 12, { class: 't-title' });
    rows.forEach((r, i) => {
      const y = top + i * rh;
      svg.append(S('text', { x: 0, y: y + rh / 2 + 4, class: 't-row t-small', fill: r.color }, LL(r.label)));
      colKeys.forEach((k, j) => {
        const p = byKey.get(k)[r.id];
        const x = labelW + j * cw;
        if (!p) return;
        const t = p.value / 100;
        const rgb = r.color.match(/\w\w/g).map((hx) => parseInt(hx, 16));
        const fill = `rgb(${rgb.map((c) => Math.round(255 + (c - 255) * t)).join(',')})`;
        const rect = S('rect', { x: x + 0.5, y: y + 0.5, width: cw - 1, height: rh - 1, fill, stroke: '#d4dcd7', 'stroke-width': 0.6, class: 'data-mark' });
        const where = groups[j] ? `${LL(STD_FROM[groups[j]] || [groups[j], groups[j]])}, σ = ${vals[j]}` : `${vals[j]}`;
        rect.append(S('title', {}, `${LL(r.label)} · ${LL(o.title)} ${where}: ${p.detail} (${f1(p.value)}%)`));
        svg.append(rect);
        if (p.value > 0 && p.value < 100 && cw >= 30) {
          svg.append(S('text', { x: x + cw / 2, y: y + rh / 2 + 3.5, 'text-anchor': 'middle', class: 't-cellsmall', fill: '#ffffff' }, f1(p.value)));
        }
      });
    });
    const yl = top + rows.length * rh + 4;
    const every = rotate ? Math.ceil(13 / cw) : 1;
    vals.forEach((v, j) => {
      const x = labelW + j * cw + cw / 2;
      const star = v.includes('★');
      if (j % every && !star) return;
      const v0 = v.replace(' ★', '');
      if (star) svg.append(S('text', { x, y: top - 3, 'text-anchor': 'middle', class: 't-tick', fill: '#a14400' }, '★'));
      if (rotate) {
        svg.append(S('text', { x, y: yl + 2, transform: `rotate(-90 ${x} ${yl + 2})`, 'text-anchor': 'end', class: 't-tick', 'dominant-baseline': 'central', fill: star ? '#a14400' : null }, v0));
      } else {
        svg.append(S('text', { x, y: yl + 10, 'text-anchor': 'middle', class: 't-tick', fill: star ? '#a14400' : null }, v0));
      }
    });
    if (groupH) {
      let j = 0;
      while (j < groups.length) {
        let e = j;
        while (e + 1 < groups.length && groups[e + 1] === groups[j]) e++;
        const x0 = labelW + j * cw + 2;
        const x1 = labelW + (e + 1) * cw - 2;
        const gy = yl + xLabH + 4;
        svg.append(S('line', { x1: x0, x2: x1, y1: gy - 1, y2: gy - 1, stroke: '#9aa8a0' }));
        svg.append(S('text', { x: (x0 + x1) / 2, y: gy + 12, 'text-anchor': 'middle', class: 't-tick' }, LL(STD_SHORT[groups[j]] || [groups[j], groups[j]])));
        j = e + 1;
      }
    }
    const wrapDiv = H('div', 'fig-cell');
    wrapDiv.append(svg);
    box.append(wrapDiv);
  }
  R['exp4-openloop'] = (ctx) => {
    const { spec, body, legendBox, foot } = ctx;
    body.classList.add('fig-flow');
    const used = [];
    for (const d of EXP4_PANELS) {
      const i = spec.panels.findIndex((p) => d.m.test(p.title));
      if (i < 0) continue;
      used.push(i);
      heatStrip(body, { title: d.t, points: spec.panels[i].points, maxWidth: body.clientWidth });
    }
    spec.panels.forEach((p, i) => { if (!used.includes(i)) { used.push(i); heatStrip(body, { title: [p.title, p.title], points: p.points, maxWidth: body.clientWidth }); } });
    legend(legendBox, [
      { box: true, color: C.base, label: ['Unedited network', '未编辑网络'] },
      { box: true, color: C.astra, label: ['Astra’s edited network', 'Astra 编辑后的网络'] },
      { box: true, color: C.script, label: ['Open-loop schedule, ignores all inputs (no LLM)', '开环序列，忽略全部输入（无 LLM）'] },
      { box: true, color: '#ffffff', stroke: '#c3ccc6', label: ['White = 0% success, full colour = 100%', '白色 = 成功率 0%，满色 = 100%'] }
    ]);
    foot.append(H('p', null, L(
      'Coarse sweeps: 256 episodes per setting; fine sweeps: 16. ★ marks the edge points discussed in the text. The fine start-shift sweep did not run the unedited network. Hover a cell for its count.',
      '粗扫描每个设置 256 局；细扫描每点 16 局。★ 为正文讨论的边缘点。细粒度初始偏移扫描未测未编辑网络。悬停可查看成功局数。')));
    return used;
  };

  // Figure 4: the three games side by side, one row per group of networks.
  const EXP5_ROWS = [
    { id: 'Random', label: ['Random policy', '随机策略'] },
    { id: 'Authors PPO', label: ['Authors’ PPO (500M steps, 3 seeds)', '作者 PPO（500M 步，3 个种子）'] },
    { id: 'Tuned PPO', label: ['Tuned PPO (500M steps, 5 seeds)', '调参 PPO（500M 步，5 个种子）'] },
    { id: 'Author Astra re-eval', label: ['Authors’ Astra network (our evaluation)', '作者 Astra 网络（我们的评测）'] },
    { id: 'Fresh Astra', label: ['Astra — 10 new replies (with source)', 'Astra — 10 次新回复（含源码）'] },
    { id: 'Astra no-source', label: ['Astra — 10 new replies (no source)', 'Astra — 10 次新回复（无源码）'] },
    { id: 'open', label: ['Open models (effort high)', '开源模型（高推理强度）'] }
  ];
  const EXP5_STYLE = {
    Random: { color: C.base, hollow: true }, 'Authors PPO': { color: C.ppo, hollow: true }, 'Tuned PPO': { color: C.ppo },
    'Author Astra re-eval': { color: C.astra, shape: 'square', r: 5 }, 'Fresh Astra': { color: C.astra }, 'Astra no-source': { color: C.astra, hollow: true }
  };
  const OPEN_SHAPE = { 'DeepSeek V4.1': 'circle', 'DeepSeek V4.1 Flash': 'square', 'Kimi K3': 'triangle', 'GLM 5.3': 'cross' };
  R['exp5-ppo-vs-astra'] = (ctx) => {
    const { spec, body, legendBox } = ctx;
    const axis = { lo: 0, hi: 700, log: true, ticks: [1, 10, 100, 500], fmt: (t) => String(t), unit: ['score (log scale)', '得分（对数刻度）'] };
    const panels = spec.panels.map((p) => {
      const pts = [];
      const value = {};
      for (const q of p.points) {
        const g = enPart(q.group);
        const model = g.split(' · ')[0];
        const isOpen = g.includes(' · ');
        const row = isOpen ? 'open' : g;
        const st = isOpen ? { color: C.open, shape: OPEN_SHAPE[model] || 'circle' } : EXP5_STYLE[g];
        const who = isOpen ? model : LL(EXP5_ROWS.find((r) => r.id === g).label);
        const what = q.median ? L('median', '中位数') : enPart(q.label);
        pts.push({ row, v: q.value, ...st, median: !!q.median, shape: q.median ? 'diamond' : st.shape, hollow: q.median ? false : st.hollow,
          tip: `${who} · ${what}: ${f1(q.value)}` });
      }
      for (const r of EXP5_ROWS) {
        const rp = pts.filter((q) => q.row === r.id);
        const med = rp.find((q) => q.median);
        if (med) value[r.id] = [{ text: f1(med.v), at: Math.max(...rp.map((q) => q.v)) }];
        else if (rp.length === 1) value[r.id] = f1(rp[0].v);
        else if (rp.length > 1) value[r.id] = `${f1(Math.min(...rp.map((q) => q.v)))}–${f1(Math.max(...rp.map((q) => q.v)))}`;
      }
      const name = p.title.charAt(0).toUpperCase() + p.title.slice(1);
      return { title: [name, name], axis, pts, value };
    });
    strips(body, { rows: EXP5_ROWS, panels, minPanel: 150, reserveRight: 50, axisLabel: true, labelMax: 150 });
    legend(legendBox, [
      { color: C.ppo, hollow: true, label: ['Authors’ PPO seed', '作者 PPO 种子'] },
      { color: C.ppo, label: ['Tuned PPO seed', '调参 PPO 种子'] },
      { color: C.astra, label: ['Astra network', 'Astra 网络'] },
      { color: C.astra, hollow: true, label: ['Astra without the simulator source', 'Astra（无模拟器源码）'] },
      { color: '#555', shape: 'diamond', label: ['Median (value shown)', '中位数（标注数值）'] },
      { color: C.open, label: ['DeepSeek V4.1', 'DeepSeek V4.1'] },
      { color: C.open, shape: 'square', label: ['DeepSeek V4.1 Flash', 'DeepSeek V4.1 Flash'] },
      { color: C.open, shape: 'triangle', label: ['Kimi K3', 'Kimi K3'] },
      { color: C.open, shape: 'cross', label: ['GLM 5.3', 'GLM 5.3'] }
    ]);
    return spec.panels.map((_, i) => i);
  };

  // Figure 7: win rates against each opponent, with intervals.
  const EXP6W_ROWS = [
    { id: 'R0', label: ['R0: random, never fills own eye', 'R0：不填自眼的随机策略'], color: C.base, hollow: true },
    { id: 'R1', label: ['R1: 30-line script (no LLM)', 'R1：30 行脚本（无 LLM）'], color: C.script },
    { id: 'b0', label: ['Astra b0 (0 passes)', 'Astra b0（0 次前向传播）'], color: C.astra },
    { id: 'b10', label: ['Astra b10 (10 passes)', 'Astra b10（10 次）'], color: C.astra },
    { id: 'b25', label: ['Astra b25 (25 passes)', 'Astra b25（25 次）'], color: C.astra },
    { id: 'b50', label: ['Astra b50 (50 passes)', 'Astra b50（50 次前向传播）'], color: C.astra }
  ];
  const EXP6W_TITLE = { 'vs s0': ['vs. PPO s0', '对 PPO s0'], 'vs s1': ['vs. PPO s1', '对 PPO s1'], 'vs s2': ['vs. PPO s2', '对 PPO s2'], 'vs random': ['vs. random play', '对随机落子'] };
  R['exp6-winrates'] = (ctx) => {
    const { spec, body, legendBox } = ctx;
    const axis = { lo: 0, hi: 100, ticks: [0, 50, 100], fmt: String, ref: 50, unit: ['win rate (%)', '胜率（%）'] };
    const panels = spec.panels.map((p) => {
      const pts = [];
      const value = {};
      const missing = {};
      for (const q of p.points) {
        const r = EXP6W_ROWS.find((x) => x.id === q.label);
        if (q.value === null) { missing[r.id] = ['not played', '未对局']; continue; }
        pts.push({ row: r.id, v: q.value, ci: q.ci, color: r.color, hollow: r.hollow,
          tip: `${LL(r.label)} ${LL(EXP6W_TITLE[p.title])}: ${f1(q.value)}% [${f1(q.ci[0])}, ${f1(q.ci[1])}] · ${q.detail}` });
        value[r.id] = f1(q.value);
      }
      return { title: EXP6W_TITLE[p.title] || [p.title, p.title], axis, pts, value, missing };
    });
    strips(body, { rows: EXP6W_ROWS, panels, minPanel: 118, reserveRight: 34, axisLabel: true, labelMax: 140 });
    legend(legendBox, [
      { color: C.script, label: ['Script without an LLM', '无 LLM 脚本'] },
      { color: C.astra, label: ['Astra-written network', 'Astra 编写的网络'] },
      { color: C.base, hollow: true, label: ['Random baseline', '随机基线'] },
      { line: true, color: '#3c4a42', dash: '5 3', label: ['Even score (50%)', '均势（50%）'] }
    ]);
    return spec.panels.map((_, i) => i);
  };

  // Figure 8: true behaviour beside 3 sessions with weights and 3 without, small multiples.
  const QTY = {
    capture_rate: ['Capture rate (%)', '提子率（%）'], atari_escape_rate: ['Atari escape rate (%)', '逃征率（%）'],
    self_atari_rate: ['Self-atari rate (%)', '自入叫吃率（%）'], eye_fill_rate: ['Eye-fill rate (%)', '填自眼率（%）'],
    pass_before_60_rate: ['Passes before move 60 (%)', '第 60 手前停着率（%）'], mean_game_length: ['Mean game length (moves)', '平均对局长度（手）'],
    winrate_vs_random: ['Win rate vs. random (%)', '对随机策略胜率（%）'], winrate_vs_R1: ['Win rate vs. R1 (%)', '对 R1 胜率（%）'],
    p_tengen: ['First move at tengen (%)', '首手下天元的概率（%）'], pairwise: ['Head-to-head: first network wins (%)', '两两对局：前者胜率（%）']
  };
  const NETS = {
    b0: ['Astra b0', 'Astra b0'], b50: ['Astra b50', 'Astra b50'], ppo: ['PPO s0', 'PPO s0'],
    b0_vs_b50: ['b0 vs. b50', 'b0 对 b50'], b0_vs_ppo: ['b0 vs. PPO', 'b0 对 PPO'], b50_vs_ppo: ['b50 vs. PPO', 'b50 对 PPO']
  };
  const session = (q) => {
    const s = (Array.isArray(q.source) ? q.source[0] : q.source).path;
    const m = s.match(/sessions\/(W|noW)_r(\d)/);
    return m ? { cond: m[1], n: Number(m[2]) + 1 } : { cond: 'truth', n: 0 };
  };
  function readStyle(cond) {
    return cond === 'truth' ? { color: C.truth, shape: 'tick' } : cond === 'W' ? { color: C.astra } : { color: C.astra, hollow: true };
  }
  function readTip(q, row, title) {
    const s = session(q);
    const who = s.cond === 'truth' ? L('measured', '实测') : s.cond === 'W' ? L(`with weights, session ${s.n}`, `有权重，会话 ${s.n}`) : L(`without weights, session ${s.n}`, `无权重，会话 ${s.n}`);
    return `${LL(NETS[row] || [row, row])} · ${LL(title)} · ${who}: ${f1(q.value)}`;
  }
  function boards(box, p, title) {
    const W = box.clientWidth;
    const groups = [{ cond: 'truth', n: 0 }, { cond: 'W', n: 1 }, { cond: 'W', n: 2 }, { cond: 'W', n: 3 }, { cond: 'noW', n: 1 }, { cond: 'noW', n: 2 }, { cond: 'noW', n: 3 }];
    const per = W < 520 ? 4 : 7;
    const gap = 10;
    const bw = Math.min(96, (W - gap * (per - 1)) / per);
    const cell = bw / 9;
    const lines = Math.ceil(groups.length / per);
    const bh = bw + 34;
    const svg = root(W, lines * bh + 4, LL(title));
    const max = Math.max(...p.points.map((q) => q.value));
    groups.forEach((g, k) => {
      const col = k % per;
      const line = Math.floor(k / per);
      const ox = col * (bw + gap);
      const oy = line * bh;
      const short = bw + gap < tw(L('Without weights 3', '无权重 3'), 11.5, 600) + 4;
      const name = g.cond === 'truth' ? L('Measured', '实测') : g.cond === 'W' ? (short ? L(`Weights ${g.n}`, `有权重 ${g.n}`) : L(`With weights ${g.n}`, `有权重 ${g.n}`))
        : (short ? L(`No weights ${g.n}`, `无权重 ${g.n}`) : L(`Without weights ${g.n}`, `无权重 ${g.n}`));
      svg.append(S('text', { x: ox, y: oy + 11, class: 't-small t-row', fill: g.cond === 'truth' ? C.truth : C.astra }, name));
      const pts = p.points.filter((q) => { const s = session(q); return s.cond === g.cond && s.n === g.n; });
      for (const q of pts) {
        const m = q.label.match(/\((\d),(\d)\)/);
        const v = q.value / max;
        const tone = Math.round(255 - v * 235);
        const fill = g.cond === 'truth' ? `rgb(${tone},${tone},${tone})` : `rgb(${Math.round(255 + (0 - 255) * v)},${Math.round(255 + (114 - 255) * v)},${Math.round(255 + (178 - 255) * v)})`;
        let rect;
        if (m) rect = S('rect', { x: ox + Number(m[2]) * cell, y: oy + 16 + Number(m[1]) * cell, width: cell - 0.6, height: cell - 0.6, fill, stroke: '#e2e8e4', 'stroke-width': 0.4, class: 'data-mark' });
        else rect = S('rect', { x: ox + bw - cell, y: oy + 20 + bw, width: cell - 0.6, height: cell - 0.6, fill, stroke: '#9aa8a0', 'stroke-width': 0.6, class: 'data-mark' });
        rect.append(S('title', {}, `${name} · ${m ? `(${m[1]},${m[2]})` : L('pass', '停着')}: ${f1(q.value)}%`));
        svg.append(rect);
      }
      svg.append(S('text', { x: ox + bw - cell - 4, y: oy + 20 + bw + cell - 1, 'text-anchor': 'end', class: 't-tick' }, L('pass', '停着')));
    });
    box.append(svg);
  }
  R['exp6-read-predictions'] = (ctx) => {
    const { spec, body, legendBox, foot, controls, view } = ctx;
    const scalar = spec.panels.map((p, i) => ({ p, i })).filter(({ p }) => !/first-move/.test(p.title));
    const panels = scalar.map(({ p }) => {
      const key = /^Pairwise/.test(p.title) ? 'pairwise' : p.title;
      const title = QTY[key] || [p.title, p.title];
      const vals = p.points.map((q) => q.value);
      const hi = p.domain && key !== 'self_atari_rate' && key !== 'eye_fill_rate' && key !== 'pass_before_60_rate' ? 100 : niceMax(Math.max(...vals));
      const lo = key === 'mean_game_length' ? Math.floor(Math.min(...vals) / 20) * 20 : 0;
      const ticks = key === 'mean_game_length' ? niceTicks(lo, hi, 3) : niceTicks(0, hi, hi === 100 ? 2 : 3);
      const pts = p.points.map((q) => {
        const s = session(q);
        return { row: q.label, v: q.value, ...readStyle(s.cond), tip: readTip(q, q.label, title) };
      });
      const value = {};
      for (const q of p.points) if (session(q).cond === 'truth') value[q.label] = [{ text: f1(q.value), at: q.value, side: 'above' }];
      return { title, axis: { lo, hi: Math.max(hi, lo + 1), ticks }, pts, value, rowsKey: [...new Set(p.points.map((q) => q.label))] };
    });
    // Same three rows in every behaviour panel; the head-to-head panel has its own rows.
    const behaviour = panels.filter((p) => p.rowsKey[0] === 'b0');
    const pairwise = panels.filter((p) => p.rowsKey[0] !== 'b0');
    const rowsOf = (keys) => keys.map((k) => ({ id: k, label: NETS[k] || [k, k] }));
    strips(body, { rows: rowsOf(behaviour[0].rowsKey), panels: behaviour, minPanel: 118, reserveLeft: 12, reserveRight: 14, rowH: 44, labelMax: 90 });
    for (const p of pairwise) strips(body, { rows: rowsOf(p.rowsKey), panels: [p], minPanel: 200, reserveLeft: 12, reserveRight: 14, rowH: 44, labelMax: 110 });
    legend(legendBox, [
      { color: C.truth, shape: 'tick', label: ['Measured (value shown)', '实测值（标注数值）'] },
      { color: C.astra, label: ['Session given the weights', '提供权重的会话'] },
      { color: C.astra, hollow: true, label: ['Session given no weights', '不提供权重的会话'] }
    ]);
    // Secondary: first-move distribution on the empty board, one network at a time.
    const fm = spec.panels.map((p, i) => ({ p, i })).filter(({ p }) => /first-move/.test(p.title));
    const sub = H('div', 'fig-subfigure');
    body.append(sub);
    const head = H('p', 'fig-subtitle', L('First move on the empty 9×9 board (darker = more likely):', '空棋盘 9×9 上的首手分布（越深越常见）：'));
    sub.append(head);
    const choose = controls(fm.map(({ p }) => NETS[p.title.split(' ')[0]] || [p.title, p.title]), 'firstmove');
    sub.append(choose.node);
    const cur = fm[choose.value];
    boards(sub, cur.p, cur.p.title);
    foot.append(H('p', null, L(
      'b0 and b50 are Astra-written networks (0 and 50 forward passes of practice); PPO s0 is the equal-sized PPO network. Each session saw the three networks under hidden names; its answers are mapped back to the real networks.',
      'b0 与 b50 为 Astra 编写的网络（练习时分别进行 0 次和 50 次前向传播）；PPO s0 为同尺寸 PPO 网络。各会话看到的是匿名的三个网络，其回答已映射回真实网络。')));
    return [...scalar.map(({ i }) => i), cur.i];
  };

  // Figure 9: distance ratio per model pair, discovery and confirmation side by side.
  R['exp7-distances'] = (ctx) => {
    const { spec, body, legendBox } = ctx;
    body.classList.add('fig-flow');
    const titles = [
      ['Discovery, pairs 100–105 — rule: smaller distance = sycophant', '发现集（第 100–105 对）— 规则：距离较小者为阿谀模型'],
      ['Confirmation, pairs 106–111 — rule: larger distance = sycophant', '确认集（第 106–111 对）— 规则：距离较大者为阿谀模型']
    ];
    const axis = { lo: 0.96, hi: 1.06, ticks: [0.97, 1, 1.03], fmt: (t) => fd(t, 2), ref: 1, unit: ['sycophant’s distance ÷ honest model’s distance (to the base model)', '阿谀模型距离 ÷ 诚实模型距离（相对基座模型）'] };
    const half = body.clientWidth >= 600;
    spec.panels.forEach((p, i) => {
      const cell = H('div', 'fig-cell');
      cell.style.width = half ? 'calc(50% - 10px)' : '100%';
      body.append(cell);
      const rows = p.points.map((q) => ({ id: q.label, label: [`Pair ${q.label}`, `第 ${q.label} 对`] }));
      const pts = p.points.map((q) => {
        const ok = /^Correct/.test(q.group);
        return { row: q.label, v: q.value, color: ok ? C.script : C.wrong, shape: ok ? 'circle' : 'cross',
          tip: `${L('Pair', '第')} ${q.label}${L('', ' 对')}: ${fd(q.value, 3)} · ${bi(q.group)}` };
      });
      const value = Object.fromEntries(p.points.map((q) => [q.label, fd(q.value, 3)]));
      strips(cell, { rows, panels: [{ title: titles[i], axis, pts, value }], minPanel: 160, reserveRight: 44, rowH: 22, axisLabel: true, labelMax: 80 });
    });
    legend(legendBox, [
      { color: C.script, label: ['Rule called the pair correctly', '规则判断正确'] },
      { color: C.wrong, shape: 'cross', label: ['Rule called it wrong', '规则判断错误'] },
      { line: true, color: '#3c4a42', dash: '5 3', label: ['Equal distances', '距离相等'] }
    ]);
    return spec.panels.map((_, i) => i);
  };

  // Figure 10: unedited vs edited model per decoding configuration, the three benchmarks side by side.
  const CONFIG = {
    A: ['A · thinking on (blog’s setting)', 'A · 开启思考（原文设置）'], B: ['B · thinking off (model-card default)', 'B · 关闭思考（模型卡默认）'],
    C: ['C · thinking on, presence penalty 0', 'C · 开启思考，存在惩罚 0'], D: ['D · thinking on + repetition penalty 1.1', 'D · 开启思考 + 重复惩罚 1.1'],
    E: ['E · budget forcing', 'E · 预算强制']
  };
  const BENCH = { math500: 'MATH-500', gsm8k: 'GSM8K', ifeval: 'IFEval' };
  const EXP8_VIEWS = [
    { label: ['Accuracy (source table)', '准确率（原表）'], kind: /source accuracy/ },
    { label: ['Accuracy, our regrade with 95% CI', '准确率（我们重评分，95% 区间）'], kind: /local regrade/ },
    { label: ['Edited A minus each setting (paired, 95% CI)', '编辑后 A 减各设置（配对，95% 区间）'], kind: /paired delta/ }
  ];
  R['exp8-cross-config'] = (ctx) => {
    const { spec, body, legendBox, controls } = ctx;
    const choose = controls(EXP8_VIEWS.map((v) => v.label), 'view');
    body.append(choose.node);
    const view = EXP8_VIEWS[choose.value];
    const delta = choose.value === 2;
    const used = Object.keys(BENCH).map((b) => spec.panels.findIndex((p) => p.title.startsWith(b) && view.kind.test(p.title)));
    const rows = Object.keys(CONFIG).map((k) => ({ id: k, label: CONFIG[k] }));
    const axis = delta ? { lo: -6, hi: 66, ticks: [0, 25, 50], fmt: (t) => (t > 0 ? `+${t}` : String(t)), ref: 0, unit: ['percentage points', '百分点'] }
      : { lo: 0, hi: 100, ticks: [0, 25, 50, 75, 100], fmt: String, unit: ['accuracy (%)', '准确率（%）'] };
    const panels = used.map((i) => {
      const p = spec.panels[i];
      const bench = BENCH[p.title.split(' ')[0]];
      const pts = [];
      const value = {};
      for (const r of rows) {
        const qs = p.points.filter((q) => q.label.match(/^(base|edited) ([A-E])/)[2] === r.id);
        const base = qs.find((q) => q.group === 'base');
        const edit = qs.find((q) => q.group === 'edited');
        const best = /best base/.test(base.label);
        const tipOf = (q, who) => `${bench} · ${LL(CONFIG[r.id])} · ${who}: ${delta ? (q.value > 0 ? '+' : '') : ''}${f1(q.value)}${delta ? '' : '%'}${q.ci ? ` [${f1(q.ci[0])}, ${f1(q.ci[1])}]` : ''}`;
        const bw = delta ? L('edited A − unedited', '编辑后 A − 未编辑') : L('unedited', '未编辑');
        const ew = delta ? L('edited A − edited', '编辑后 A − 编辑后') : L('edited', '编辑后');
        pts.push({ row: r.id, v: base.value, ci: base.ci, color: C.base, r: best ? 5.5 : 4.5, tip: tipOf(base, bw) + (best ? L(' · best unedited setting', ' · 最佳未编辑设置') : '') });
        if (edit.value !== null) pts.push({ row: r.id, v: edit.value, ci: edit.ci, color: C.astra, tip: tipOf(edit, ew) });
        const sgn = (v) => (delta && v > 0 ? '+' : '') + f1(v);
        const bText = (best ? '★ ' : '') + sgn(base.value);
        if (edit.value === null) {
          value[r.id] = [{ text: bText, side: 'right', at: base.ci ? base.ci[1] : base.value }];
        } else {
          const bLeft = base.value <= edit.value;
          const bEdge = base.ci ? (bLeft ? base.ci[0] : base.ci[1]) : base.value;
          const eEdge = edit.ci ? (bLeft ? edit.ci[1] : edit.ci[0]) : edit.value;
          value[r.id] = [{ text: bText, side: bLeft ? 'left' : 'right', at: bEdge }, { text: sgn(edit.value), side: bLeft ? 'right' : 'left', at: eEdge }];
        }
      }
      return { title: [bench, bench], axis, pts, value, link: !delta };
    });
    strips(body, { rows, panels, minPanel: 150, reserveLeft: 36, reserveRight: 36, axisLabel: true, labelMax: 140, rowH: 34 });
    if (delta) ctx.foot.append(H('p', null, L('Row A has no edited mark: edited A is the reference that every difference is taken from.', 'A 行没有编辑后标记：所有差值都以编辑后 A 为参照。')));
    legend(legendBox, delta ? [
      { color: C.base, label: ['Edited A minus the unedited model in this setting', '编辑后 A 减该设置下的未编辑模型'] },
      { color: C.astra, label: ['Edited A minus the edited model in this setting', '编辑后 A 减该设置下的编辑后模型'] },
      { line: true, color: '#3c4a42', dash: '5 3', label: ['No difference', '无差异'] }
    ] : [
      { color: C.base, label: ['Unedited model', '未编辑模型'] },
      { color: C.astra, label: ['Astra’s edited model', 'Astra 编辑后的模型'] },
      { color: C.base, r: 5.5, label: ['★ best unedited setting for this benchmark', '★ 该基准上最佳的未编辑设置'] }
    ]);
    return used;
  };

  // Figure 11: accuracy as the steering scale grows (practice split), and the full benchmark.
  R['exp8-steer-sweep'] = (ctx) => {
    const { spec, body, legendBox } = ctx;
    const W = body.clientWidth;
    const wide = W >= 620;
    const lw = wide ? Math.floor(W * 0.56) : W;
    const sweep = spec.panels[0];
    const full = spec.panels[1];
    // Left: line chart over k.
    {
      const h = 250;
      const svg = root(lw, h, L('Practice-split accuracy as the scale k grows', '练习集准确率随缩放系数 k 的变化'));
      const pad = { l: 34, r: 12, t: 30, b: 44 };
      const x = (k) => pad.l + 18 + (k / 10) * (lw - pad.l - pad.r - 48);
      const y = (v) => pad.t + (1 - v / 60) * (h - pad.t - pad.b);
      textLines(svg, wrap(L('Practice split (512 problems), accuracy (%)', '练习集（512 题）准确率（%）'), lw - 4, 13, 700), 0, 13, { class: 't-title' });
      for (const t of [0, 20, 40, 60]) {
        svg.append(S('line', { x1: pad.l, x2: lw - pad.r, y1: y(t), y2: y(t), stroke: '#d3ddd6', 'stroke-dasharray': '2 4' }));
        svg.append(S('text', { x: pad.l - 6, y: y(t) + 4, 'text-anchor': 'end', class: 't-tick' }, String(t)));
      }
      const steer = [];
      const missing = [];
      for (const q of sweep.points) {
        const m = q.label.match(/^steer-k(\d+)/);
        if (m) (q.value === null ? missing : steer).push({ k: Number(m[1]), q });
      }
      const base = sweep.points.find((q) => q.label === 'base');
      for (let k = 0; k <= 10; k++) {
        const miss = missing.some((s) => s.k === k);
        svg.append(S('text', { x: x(k), y: h - pad.b + 16, 'text-anchor': 'middle', class: miss ? 't-missing' : 't-tick' }, String(k)));
      }
      svg.append(S('text', { x: (pad.l + lw - pad.r) / 2, y: h - 8, 'text-anchor': 'middle', class: 't-tick' },
        L(`scale k (k = ${missing.map((s) => s.k).join(', ')} not run; 0 = unedited)`, `缩放系数 k（k = ${missing.map((s) => s.k).join('、')} 未测；0 = 未编辑）`)));
      const line = [{ k: 0, q: base }, ...steer].sort((a, b) => a.k - b.k);
      svg.append(S('path', { d: line.map((s, i) => `${i ? 'L' : 'M'}${x(s.k)},${y(s.q.value)}`).join(''), fill: 'none', stroke: C.script, 'stroke-width': 1.6, 'stroke-dasharray': '1 0' }));
      for (const s of line) {
        const isBase = s.k === 0;
        const col = isBase ? C.base : C.script;
        svg.append(S('line', { x1: x(s.k), x2: x(s.k), y1: y(s.q.ci[0]), y2: y(s.q.ci[1]), stroke: col, 'stroke-width': 1.4 }));
        mark(svg, x(s.k), y(s.q.value), { color: col, focus: true,
          tip: `${isBase ? L('Unedited', '未编辑') : `k = ${s.k}`}: ${f1(s.q.value)}% [${f1(s.q.ci[0])}, ${f1(s.q.ci[1])}]` });
        svg.append(S('text', { x: x(s.k), y: y(s.q.ci[1]) - 5, 'text-anchor': 'middle', class: 't-val t-small' }, f1(s.q.value)));
      }
      // Controls at k = 8, drawn just right of the steering point.
      const ctl = [['randdir-k8', 'triangle', ['random direction', '随机方向']], ['randcols-k8', 'square', ['random columns', '随机列']]];
      ctl.forEach(([id, shape, name], j) => {
        const q = sweep.points.find((p) => p.label === id);
        const cx = x(8) + 14 + j * 14;
        svg.append(S('line', { x1: cx, x2: cx, y1: y(q.ci[0]), y2: y(q.ci[1]), stroke: C.control, 'stroke-width': 1.4 }));
        mark(svg, cx, y(q.value), { color: C.control, shape, focus: true, tip: `${L('Control', '对照')}: ${LL(name)}, k = 8: ${f1(q.value)}% [${f1(q.ci[0])}, ${f1(q.ci[1])}]` });
      });
      const qd = sweep.points.find((p) => p.label === 'randdir-k8');
      const qc = sweep.points.find((p) => p.label === 'randcols-k8');
      svg.append(S('text', { x: x(8) + 42, y: y(Math.max(qd.value, qc.value)) + 4, class: 't-val t-small' }, `${f1(qd.value)}, ${f1(qc.value)}`));
      const cell = H('div', 'fig-cell');
      cell.style.width = wide ? `${lw}px` : '100%';
      cell.append(svg);
      body.append(cell);
    }
    // Right: the full benchmark.
    {
      const cell = H('div', 'fig-cell');
      cell.style.width = wide ? `${W - lw - 20}px` : '100%';
      body.append(cell);
      const order = [
        ['base', ['Unedited', '未编辑'], C.base], ['randdir-k8', ['Random direction (k = 8)', '随机方向（k = 8）'], C.control],
        ['randcols-k8', ['Random columns (k = 8)', '随机列（k = 8）'], C.control], ['steer-k8', ['Difference of means, k = 8 (no LLM)', '均值差方向，k = 8（无 LLM）'], C.script],
        ['Astra', ['Astra’s edit', 'Astra 的编辑'], C.astra]
      ];
      const rows = order.map(([id, label]) => ({ id, label }));
      const pts = [];
      const value = {};
      for (const [id, label, color] of order) {
        const q = full.points.find((p) => p.label === id);
        pts.push({ row: id, v: q.value, color, bar: true, tip: `${LL(label)}: ${f1(q.value)}%${q.ci ? ` [${f1(q.ci[0])}, ${f1(q.ci[1])}]` : ''}` });
        if (q.ci) pts.push({ row: id, v: null, ci: q.ci, color: '#1f2d26' });
        value[id] = [{ text: f1(q.value), at: q.ci ? q.ci[1] : q.value }];
      }
      // Whiskers without a mark of their own: draw intervals as part of the bar rows.
      const ptsDrawn = pts.filter((p) => p.v !== null).map((p) => {
        const ci = pts.find((c) => c.v === null && c.row === p.row);
        return ci ? { ...p, ci: ci.ci } : p;
      });
      strips(cell, { rows, panels: [{ title: ['Full MATH-500, accuracy (%)', '完整 MATH-500 准确率（%）'], axis: { lo: 0, hi: 60, ticks: [0, 20, 40, 60], fmt: String }, pts: ptsDrawn, value }], minPanel: 120, reserveRight: 36, labelMax: wide ? 130 : 170 });
    }
    body.classList.add('fig-flow');
    legend(legendBox, [
      { color: C.script, label: ['Difference-of-means edit (no LLM)', '均值差方向编辑（无 LLM）'] },
      { color: C.base, label: ['Unedited model', '未编辑模型'] },
      { color: C.control, shape: 'triangle', label: ['Controls: random direction / random columns', '对照：随机方向／随机列'] },
      { box: true, color: C.astra, label: ['Astra’s edit', 'Astra 的编辑'] }
    ]);
    return [0, 1];
  };

  // ---------- Values and sources (built when opened) ----------
  function sourceText(source) {
    return (Array.isArray(source) ? source : [source]).flatMap((s) => (Array.isArray(s) ? sourceText(s) : `${s.path}:${s.line}${s.key ? ' [' + s.key + ']' : ''}`)).join('; ');
  }
  function fillTable(box, spec) {
    box.replaceChildren();
    box.append(H('p', null, spec.note[zh() ? 1 : 0]));
    for (const p of spec.panels) {
      box.append(H('p', 'fig-table-title', bi(p.title)));
      if (p.grids) {
        for (const g of p.grids) box.append(H('p', null, `${bi(g.label)} — ${sourceText(g.source)}`));
        continue;
      }
      const t = H('table', 'fig-data-table');
      const head = H('tr');
      [L('Measurement', '测量'), L('Value [interval]', '数值［区间］'), L('Source', '来源')].forEach((h) => head.append(H('th', null, h)));
      t.append(head);
      for (const q of p.points) {
        const r = H('tr');
        const v = q.value === null ? L('not measured', '未测') : f1(q.value) + (q.ci ? ` [${f1(q.ci[0])}, ${f1(q.ci[1])}]` : '');
        [`${bi(q.group)} · ${bi(q.label)}`, v + (q.detail ? ` · ${bi(q.detail)}` : ''), sourceText(q.source)].forEach((c) => r.append(H('td', null, c)));
        t.append(r);
      }
      box.append(t);
    }
  }

  // ---------- Mounting ----------
  const mounts = [];
  function draw(m) {
    const { host, spec } = m;
    fontFamily = getComputedStyle(host).fontFamily || fontFamily;
    m.controlsBox.replaceChildren();
    m.legendBox.replaceChildren();
    m.body.replaceChildren();
    m.body.className = 'fig-body';
    m.foot.replaceChildren();
    let ci = 0;
    const controls = (options, key) => {
      const idx = ci++;
      const value = m.state[key] ?? 0;
      const node = H('div', 'fig-controls');
      node.setAttribute('role', 'group');
      node.setAttribute('aria-label', L('Choose what to show', '选择显示内容'));
      options.forEach((opt, i) => {
        const b = H('button', 'fig-toggle', LL(opt));
        b.type = 'button';
        b.dataset.option = String(i);
        b.dataset.control = String(idx);
        b.setAttribute('aria-pressed', i === value ? 'true' : 'false');
        b.addEventListener('click', () => { m.state[key] = i; draw(m); });
        node.append(b);
      });
      return { node, value };
    };
    const used = R[spec.key]({ spec, body: m.body, legendBox: m.legendBox, foot: m.foot, controls, view: m.state });
    const expected = used.reduce((a, i) => a + spec.panels[i].expected_marks, 0);
    host.dataset.expectedMarks = String(expected);
    host.dataset.renderedMarks = String(m.body.querySelectorAll('.data-mark').length);
    host.dataset.panels = used.join(',');
    if (m.details.open) fillTable(m.table, spec);
  }
  function init() {
    for (const [key, spec] of Object.entries(data)) {
      const figure = document.getElementById(spec.id);
      if (!figure || !R[key]) continue;
      const host = figure.querySelector('.data-figure');
      if (!host) continue;
      spec.key = key;
      const controlsBox = H('div', 'fig-control-row');
      const legendBox = H('div', 'fig-legend');
      const body = H('div', 'fig-body');
      const foot = H('div', 'fig-foot');
      const details = H('details', 'fig-source-data');
      const summary = H('summary');
      const en = H('span', null, 'Values and source locations');
      const cn = H('span', null, '数值及来源位置');
      en.lang = 'en';
      cn.lang = 'zh';
      summary.append(en, cn);
      const table = H('div', 'fig-table-scroll');
      details.append(summary, table);
      host.append(controlsBox, legendBox, body, foot, details);
      const m = { host, spec, controlsBox, legendBox, body, foot, details, table, state: {} };
      details.addEventListener('toggle', () => { if (details.open) fillTable(table, spec); });
      mounts.push(m);
      draw(m);
      let lastWidth = host.clientWidth;
      new ResizeObserver(() => {
        if (host.clientWidth !== lastWidth) {
          lastWidth = host.clientWidth;
          draw(m);
        }
      }).observe(host);
    }
    new MutationObserver(() => mounts.forEach(draw)).observe(document.documentElement, { attributes: true, attributeFilter: ['data-lang'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => mounts.forEach(draw));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
