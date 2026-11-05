#!/usr/bin/env python3
"""Build self-contained blog/figs/breakout_explainer/explainer.js.

Embeds units.json and frames.json as fallback data so the page works from both
http(s):// (fetching the JSON files) and file:// (where fetch is blocked by CORS).

Run with:
    exp5/astra_write/.venv/bin/python -I exp5/explainer/build_explainer_js.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "blog", "figs", "breakout_explainer")
UNITS_JSON = os.path.join(FIG_DIR, "units.json")
FRAMES_JSON = os.path.join(FIG_DIR, "frames.json")
OUT_JS = os.path.join(FIG_DIR, "explainer.js")

JS_BODY = r"""
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const POS_COL = "#2a78d6";   // validated blue for positive weights / activations
  const NEG_COL = "#d93a4b";   // validated red for negative weights / activations
  const ZERO_COL = "#eef0f3";  // neutral midpoint
  const INK = "#222222";
  const MUTED = "#444444";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function bi(en, zh) {
    return `<span lang="en">${en}</span><span lang="zh">${zh}</span>`;
  }

  function svgEl(tag, attrs = {}, parent = null) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v != null) el.setAttribute(k, String(v));
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  function svgTextBi(parent, x, y, en, zh, attrs = {}) {
    const g = svgEl("g", {}, parent);
    const tEn = svgEl("text", { x, y, lang: "en", ...attrs }, g);
    tEn.textContent = en;
    const tZh = svgEl("text", { x, y, lang: "zh", ...attrs }, g);
    tZh.textContent = zh;
    return g;
  }

  // Diverging color interpolation: v in [-1, 1] -> hex/rgb
  function divColor(v, maxAbs = 1.0) {
    const x = Math.max(-1, Math.min(1, maxAbs > 0 ? v / maxAbs : 0));
    // Midpoint #eef0f3 = (238, 240, 243)
    // Pos #2a78d6 = (42, 120, 214)
    // Neg #d93a4b = (217, 58, 75)
    const r0 = 238, g0 = 240, b0 = 243;
    if (Math.abs(x) < 1e-4) return `rgb(${r0},${g0},${b0})`;
    const t = Math.pow(Math.abs(x), 0.78);
    if (x > 0) {
      const r = Math.round(r0 + (42 - r0) * t);
      const g = Math.round(g0 + (120 - g0) * t);
      const b = Math.round(b0 + (214 - b0) * t);
      return `rgb(${r},${g},${b})`;
    } else {
      const r = Math.round(r0 + (217 - r0) * t);
      const g = Math.round(g0 + (58 - g0) * t);
      const b = Math.round(b0 + (75 - b0) * t);
      return `rgb(${r},${g},${b})`;
    }
  }

  function renderInspector(box, unitObj, currentAct = null) {
    if (!unitObj) return;
    if (!unitObj.active) {
      const lyr = unitObj.layer;
      const u = unitObj.unit;
      box.innerHTML = `
        <div class="bx-insp-head">
          <span class="bx-insp-id">h${lyr}[${u}]</span>
          <span>${bi("Unused unit (all weights = 0)", "未使用单元（全部权重为 0）")}</span>
          ${currentAct != null ? `<span class="bx-step-readout">tanh = 0.000</span>` : ""}
        </div>
        <div>${bi(
          `Incoming weights, bias, and outgoing weights are all 0; activation is 0.000 on every frame.`,
          `输入权重、偏置与输出权重均为 0；在所有帧上激活值恒为 0.000。`
        )}</div>`;
      return;
    }
    const ev = unitObj.evidence;
    const actBadge =
      currentAct != null
        ? `<span class="bx-step-readout">${bi("Current activation:", "当前激活值：")} ${currentAct >= 0 ? "+" : ""}${currentAct.toFixed(3)}</span>`
        : `<span class="bx-step-readout">${bi(
            `Nonzero weights: in=${unitObj.nnz_in}, b=${unitObj.bias}, out=${unitObj.nnz_out}`,
            `非零参数：输入=${unitObj.nnz_in}，偏置=${unitObj.bias}，输出=${unitObj.nnz_out}`
          )}</span>`;
    box.innerHTML = `
      <div class="bx-insp-head">
        <span class="bx-insp-id">h${unitObj.layer}[${unitObj.unit}]</span>
        <span>${bi(esc(unitObj.label_en), esc(unitObj.label_zh))}</span>
        ${actBadge}
      </div>
      <div class="bx-insp-grid">
        <div class="bx-insp-col">
          <b>${bi("(a) Astra's design (answer.md)", "（a）Astra 的设计说明（answer.md）")}</b>
          <div><code>${esc(ev.answer_md)}</code></div>
        </div>
        <div class="bx-insp-col">
          <b>${bi("(b) Weight pattern", "（b）权重模式")}</b>
          <div>${bi(esc(ev.weight_pattern_en), esc(ev.weight_pattern_zh))}</div>
        </div>
        <div class="bx-insp-col">
          <b>${bi("(c) Empirical check (380 frames)", "（c）380 帧实证检验")}</b>
          <div>${bi(esc(ev.empirical.summary_en), esc(ev.empirical.summary_zh))}</div>
        </div>
      </div>`;
  }

  // =========================================================================
  // FIGURE 1: Static Circuit Diagram
  // =========================================================================
  function mountStaticDiagram(root, unitsData) {
    root.innerHTML = `
      <div class="bx-top">
        <div class="bx-title">${bi(
          "Compiled Breakout circuit: 600 inputs → 6 comparators/switches → 8 AND gates → 6 actions",
          "编译后的 Breakout 电路：600 维输入 → 6 个比较器／开关 → 8 个逻辑与门 → 6 个动作"
        )}</div>
        <div class="bx-legend">
          <span class="bx-leg-item"><span class="bx-swatch" style="background:${POS_COL}"></span>${bi("Positive (+)", "正权重 (+)")}</span>
          <span class="bx-leg-item"><span class="bx-swatch" style="background:${NEG_COL}"></span>${bi("Negative (−)", "负权重 (−)")}</span>
          <span class="bx-leg-item"><span class="bx-line-key" style="border-top-width:3px;border-color:#222"></span>|W2|=1000</span>
          <span class="bx-leg-item"><span class="bx-line-key" style="border-top-width:1.3px;border-color:#555"></span>|W2|=200</span>
          <span class="bx-leg-item"><span class="bx-line-key" style="border-top-width:2.2px;border-color:${POS_COL}"></span>W3=+30</span>
        </div>
      </div>
      <div class="bx-svg-holder"></div>
      <div class="bx-inspector"></div>`;

    const holder = root.querySelector(".bx-svg-holder");
    const inspector = root.querySelector(".bx-inspector");

    const h1Map = {};
    unitsData.h1_units.forEach((u) => { h1Map[u.unit] = u; });
    const h2Map = {};
    unitsData.h2_units.forEach((u) => { h2Map[u.unit] = u; });

    let selectedCompIdx = 0; // 0..3 comparator shown in top heatmap card

    function drawStatic(highlightLayer = 1, highlightUnit = 0) {
      holder.replaceChildren();
      const W = 760, H = 456;
      const s = svgEl("svg", {
        viewBox: `0 0 ${W} ${H}`,
        width: W,
        height: H,
        class: "bx-svg",
        role: "img",
        "aria-label": "Static architecture diagram of Astra's Breakout neural network circuit."
      }, holder);

      // Column headers
      svgTextBi(s, 118, 20, "1. W1 input heatmaps (10×10)", "1. W1 输入通道热力图 (10×10)", {
        "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: INK
      });
      svgTextBi(s, 336, 20, "2. Layer 1 (h1: 6/32 active)", "2. 第一隐藏层 (h1: 6/32 激活)", {
        "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: INK
      });
      svgTextBi(s, 536, 20, "3. Layer 2 (h2: 8/32 AND gates)", "3. 第二隐藏层 (h2: 8/32 与门)", {
        "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: INK
      });
      svgTextBi(s, 696, 20, "4. Logits (6 actions)", "4. 输出层 (6 动作)", {
        "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: INK
      });

      // Helper to draw a 10x10 heatmap
      function drawHeatmap(parent, x0, y0, cellSz, grid, maxVal, titleEn, titleZh) {
        svgTextBi(parent, x0 + (cellSz * 10) / 2, y0 - 5, titleEn, titleZh, {
          "text-anchor": "middle", "font-size": 10.5, "font-weight": 600, fill: INK
        });
        svgEl("rect", {
          x: x0 - 1, y: y0 - 1, width: cellSz * 10 + 2, height: cellSz * 10 + 2,
          fill: "#ffffff", stroke: "#8b929b", "stroke-width": 1
        }, parent);
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            const v = grid ? grid[r][c] : 0;
            svgEl("rect", {
              x: x0 + c * cellSz,
              y: y0 + r * cellSz,
              width: cellSz - 0.5,
              height: cellSz - 0.5,
              fill: divColor(v, maxVal)
            }, parent);
          }
        }
      }

      // Left column: 3 representative Layer-1 unit heatmap cards:
      // Card A: h1[selectedCompIdx] (default h1[0], comparator)
      // Card B: h1[4] (horizontal direction ramp)
      // Card C: h1[5] (vertical direction ramp)
      const compUnit = h1Map[selectedCompIdx];
      const repCards = [
        {
          u: compUnit,
          y: 34,
          maxV: 0.18,
          chA: ["ball", "Ch 3: ball (+target)", "通道3: 球 (+目标列)"],
          chB: ["paddle", "Ch 0: paddle (−col)", "通道0: 挡板 (−列斜坡)"],
          footEn: `h1[${compUnit.unit}]: 0.02×(target − paddle_col)`,
          footZh: `h1[${compUnit.unit}]: 0.02×(目标列 − 挡板列)`
        },
        {
          u: h1Map[4],
          y: 170,
          maxV: 180,
          chA: ["ball", "Ch 3: ball (+20×col)", "通道3: 球 (+20×列)"],
          chB: ["trail", "Ch 4: trail (−20×col)", "通道4: 轨迹 (−20×列)"],
          footEn: "h1[4]: tanh(20×(ball_col − trail_col))",
          footZh: "h1[4]: tanh(20×(球列 − 轨迹列))"
        },
        {
          u: h1Map[5],
          y: 306,
          maxV: 90,
          chA: ["ball", "Ch 3: ball (+10×row)", "通道3: 球 (+10×行)"],
          chB: ["trail", "Ch 4: trail (−10×row)", "通道4: 轨迹 (−10×行)"],
          footEn: "h1[5]: tanh(10×(ball_row − trail_row)+5)",
          footZh: "h1[5]: tanh(10×(球行 − 轨迹行)+5)"
        }
      ];

      const cardCentersY = {};
      repCards.forEach((rc) => {
        const gCard = svgEl("g", { class: "bx-hm-card" }, s);
        svgEl("rect", {
          x: 8, y: rc.y, width: 220, height: 124, rx: 5,
          fill: "#f8f9fb", stroke: "#cfd5dc", "stroke-width": 1
        }, gCard);
        const sz = 8.2;
        drawHeatmap(
          gCard, 18, rc.y + 19, sz,
          rc.u.w1_channels[rc.chA[0]], rc.maxV, rc.chA[1], rc.chA[2]
        );
        drawHeatmap(
          gCard, 128, rc.y + 19, sz,
          rc.u.w1_channels[rc.chB[0]], rc.maxV, rc.chB[1], rc.chB[2]
        );
        svgTextBi(gCard, 118, rc.y + 115, rc.footEn, rc.footZh, {
          "text-anchor": "middle", "font-size": 10.5, "font-family": "var(--font-mono)", fill: INK
        });
        cardCentersY[rc.u.unit] = rc.y + 62;
      });

      // Layer 1 node positions (6 active + 1 inactive summary block)
      const h1Pos = {};
      const h1StartY = 46, h1StepY = 52;
      for (let u = 0; u < 6; u++) {
        h1Pos[u] = { x: 336, y: h1StartY + u * h1StepY, w: 120, h: 36 };
      }

      // Layer 2 node positions (8 active + 1 inactive summary block)
      const h2Pos = {};
      const h2StartY = 38, h2StepY = 41;
      for (let u2 = 0; u2 < 8; u2++) {
        h2Pos[u2] = { x: 536, y: h2StartY + u2 * h2StepY + Math.floor(u2 / 2) * 5, w: 118, h: 31 };
      }

      // Output action positions (6 actions)
      const actPos = {};
      const actStartY = 50, actStepY = 58;
      for (let a = 0; a < 6; a++) {
        actPos[a] = { x: 696, y: actStartY + a * actStepY, w: 102, h: 36 };
      }

      // Connector lines from representative heatmap cards to their h1 nodes
      Object.entries(cardCentersY).forEach(([uStr, cy]) => {
        const u = Number(uStr);
        const target = h1Pos[u];
        svgEl("path", {
          d: `M 228 ${cy} C 252 ${cy}, 252 ${target.y}, ${target.x - target.w / 2} ${target.y}`,
          fill: "none",
          stroke: "#8b929b",
          "stroke-width": 1.3,
          "stroke-dasharray": "3 3"
        }, s);
      });

      // Draw 24 nonzero W2 edges (behind nodes)
      const edgesG = svgEl("g", {}, s);
      unitsData.w2_edges.forEach((e) => {
        const p1 = h1Pos[e.from];
        const p2 = h2Pos[e.to];
        const isComp = Math.abs(e.weight) >= 500;
        const isFocused =
          (highlightLayer === 1 && highlightUnit === e.from) ||
          (highlightLayer === 2 && highlightUnit === e.to);
        const baseOpacity = isComp ? 0.72 : 0.30;
        const opacity = isFocused ? 0.98 : (highlightLayer != null ? Math.min(baseOpacity, 0.25) : baseOpacity);
        const sw = isComp ? (isFocused ? 3.2 : 2.5) : (isFocused ? 2.0 : 1.15);
        const x1 = p1.x + p1.w / 2;
        const y1 = p1.y;
        const x2 = p2.x - p2.w / 2;
        const y2 = p2.y;
        const cx1 = x1 + (x2 - x1) * 0.42;
        const cx2 = x1 + (x2 - x1) * 0.58;
        const path = svgEl("path", {
          d: `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`,
          fill: "none",
          stroke: e.weight > 0 ? POS_COL : NEG_COL,
          "stroke-width": sw,
          "stroke-opacity": opacity
        }, edgesG);
        const title = svgEl("title", {}, path);
        title.textContent = `W2[h1[${e.from}], h2[${e.to}]] = ${e.weight > 0 ? "+" : ""}${e.weight}`;
      });

      // Draw 8 nonzero W3 edges
      unitsData.w3_edges.forEach((e) => {
        const p2 = h2Pos[e.from];
        const pa = actPos[e.to];
        const isFocused = highlightLayer === 2 && highlightUnit === e.from;
        const opacity = isFocused ? 0.98 : 0.65;
        const sw = isFocused ? 3.0 : 2.0;
        const x1 = p2.x + p2.w / 2;
        const y1 = p2.y;
        const x2 = pa.x - pa.w / 2;
        const y2 = pa.y;
        const path = svgEl("path", {
          d: `M ${x1} ${y1} C ${x1 + 26} ${y1}, ${x2 - 26} ${y2}, ${x2} ${y2}`,
          fill: "none",
          stroke: e.weight > 0 ? POS_COL : NEG_COL,
          "stroke-width": sw,
          "stroke-opacity": opacity
        }, edgesG);
        const title = svgEl("title", {}, path);
        title.textContent = `W3[h2[${e.from}], action[${e.to}]] = +${e.weight}`;
      });

      // Draw Layer 1 active nodes (h1[0..5])
      unitsData.h1_units.forEach((uObj) => {
        const p = h1Pos[uObj.unit];
        const isSel = highlightLayer === 1 && highlightUnit === uObj.unit;
        const g = svgEl("g", {
          class: `bx-node${isSel ? " is-active" : ""}`,
          tabindex: "0"
        }, s);
        svgEl("rect", {
          class: "bx-node-box",
          x: p.x - p.w / 2,
          y: p.y - p.h / 2,
          width: p.w,
          height: p.h,
          rx: 5,
          fill: uObj.role === "comparator" ? "#f2f7fd" : "#f3f9f5",
          stroke: isSel ? "#222222" : (uObj.role === "comparator" ? POS_COL : "#118659"),
          "stroke-width": isSel ? 2.2 : 1.3
        }, g);
        svgTextBi(
          g, p.x, p.y - 3,
          `h1[${uObj.unit}] ${uObj.short_en}`,
          `h1[${uObj.unit}] ${uObj.short_zh}`,
          { "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, fill: INK }
        );
        svgTextBi(
          g, p.x, p.y + 11,
          uObj.role === "comparator" ? `W1:${uObj.nnz_in} nnz` : `W1:${uObj.nnz_in} nnz, b1=${uObj.bias}`,
          uObj.role === "comparator" ? `W1:${uObj.nnz_in} 非零` : `W1:${uObj.nnz_in} 非零, b1=${uObj.bias}`,
          { "text-anchor": "middle", "font-size": 9.8, "font-family": "var(--font-mono)", fill: MUTED }
        );
        const activate = () => {
          if (uObj.unit < 4) selectedCompIdx = uObj.unit;
          drawStatic(1, uObj.unit);
          renderInspector(inspector, uObj);
        };
        g.addEventListener("mouseenter", activate);
        g.addEventListener("click", activate);
      });

      // Inactive h1[6..31] summary box
      const gIn1 = svgEl("g", { class: "bx-node" }, s);
      svgEl("rect", {
        class: "bx-node-box",
        x: 336 - 60, y: 368, width: 120, height: 48, rx: 5,
        fill: "#f3f4f6", stroke: "#c9ced5", "stroke-dasharray": "3 2"
      }, gIn1);
      svgTextBi(gIn1, 336, 388, "h1[6..31] (26 units)", "h1[6..31] (26 个单元)", {
        "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: MUTED
      });
      svgTextBi(gIn1, 336, 404, "all weights = 0", "全部权重为 0", {
        "text-anchor": "middle", "font-size": 10, "font-family": "var(--font-mono)", fill: MUTED
      });
      gIn1.addEventListener("mouseenter", () => {
        renderInspector(inspector, { layer: 1, unit: "6..31", active: false });
      });

      // Draw Layer 2 active nodes (h2[0..7])
      unitsData.h2_units.forEach((uObj) => {
        const p = h2Pos[uObj.unit];
        const isSel = highlightLayer === 2 && highlightUnit === uObj.unit;
        const g = svgEl("g", {
          class: `bx-node${isSel ? " is-active" : ""}`,
          tabindex: "0"
        }, s);
        svgEl("rect", {
          class: "bx-node-box",
          x: p.x - p.w / 2,
          y: p.y - p.h / 2,
          width: p.w,
          height: p.h,
          rx: 5,
          fill: "#ffffff",
          stroke: isSel ? "#222222" : "#7c848e",
          "stroke-width": isSel ? 2.2 : 1.2
        }, g);
        svgTextBi(
          g, p.x, p.y - 2,
          `h2[${uObj.unit}] ${uObj.short_en}`,
          `h2[${uObj.unit}] ${uObj.short_zh}`,
          { "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: INK }
        );
        svgTextBi(
          g, p.x, p.y + 10,
          `b2 = -415`,
          `偏置 b2 = -415`,
          { "text-anchor": "middle", "font-size": 9.5, "font-family": "var(--font-mono)", fill: MUTED }
        );
        const activate = () => {
          drawStatic(2, uObj.unit);
          renderInspector(inspector, uObj);
        };
        g.addEventListener("mouseenter", activate);
        g.addEventListener("click", activate);
      });

      // Inactive h2[8..31] summary box
      const gIn2 = svgEl("g", { class: "bx-node" }, s);
      svgEl("rect", {
        class: "bx-node-box",
        x: 536 - 59, y: 382, width: 118, height: 44, rx: 5,
        fill: "#f3f4f6", stroke: "#c9ced5", "stroke-dasharray": "3 2"
      }, gIn2);
      svgTextBi(gIn2, 536, 400, "h2[8..31] (24 units)", "h2[8..31] (24 个单元)", {
        "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: MUTED
      });
      svgTextBi(gIn2, 536, 415, "all weights = 0", "全部权重为 0", {
        "text-anchor": "middle", "font-size": 10, "font-family": "var(--font-mono)", fill: MUTED
      });
      gIn2.addEventListener("mouseenter", () => {
        renderInspector(inspector, { layer: 2, unit: "8..31", active: false });
      });

      // Draw Output action nodes (0..5)
      const actNamesEn = ["0: no-op", "1: left", "2: up", "3: right", "4: down", "5: fire"];
      const actNamesZh = ["0: 不动", "1: 左移", "2: 上", "3: 右移", "4: 下", "5: 发射"];
      const b3 = unitsData.architecture.b3;
      for (let a = 0; a < 6; a++) {
        const p = actPos[a];
        const isUsed = a === 0 || a === 1 || a === 3;
        const g = svgEl("g", {}, s);
        svgEl("rect", {
          x: p.x - p.w / 2,
          y: p.y - p.h / 2,
          width: p.w,
          height: p.h,
          rx: 5,
          fill: isUsed ? "#f3f9f5" : "#f6f7f9",
          stroke: isUsed ? "#118659" : "#cfd5dc",
          "stroke-width": isUsed ? 1.5 : 1
        }, g);
        svgTextBi(g, p.x, p.y - 3, actNamesEn[a], actNamesZh[a], {
          "text-anchor": "middle", "font-size": 11.5, "font-weight": isUsed ? 700 : 500, fill: isUsed ? INK : MUTED
        });
        const bStr = `b3=${b3[a] >= 0 ? "+" : ""}${b3[a]}`;
        svgEl("text", {
          x: p.x, y: p.y + 11, "text-anchor": "middle",
          "font-size": 9.8, "font-family": "var(--font-mono)", fill: MUTED
        }, g).textContent = bStr;
      }
    }

    drawStatic(1, 0);
    renderInspector(inspector, h1Map[0]);
  }

  // =========================================================================
  // FIGURE 2: Real-Data Rollout Animation
  // =========================================================================
  function mountAnimation(root, unitsData, framesData) {
    const frames = framesData.frames;
    const T = frames.length;

    const h1Map = {};
    unitsData.h1_units.forEach((u) => { h1Map[u.unit] = u; });
    const h2Map = {};
    unitsData.h2_units.forEach((u) => { h2Map[u.unit] = u; });

    root.innerHTML = `
      <div class="bx-controls">
        <button type="button" class="bx-btn bx-btn-primary bx-play-btn">${bi("▶ Play", "▶ 播放")}</button>
        <button type="button" class="bx-btn bx-prev-btn" title="Step back">◀ ${bi("Step", "上一步")}</button>
        <button type="button" class="bx-btn bx-next-btn" title="Step forward">${bi("Step", "下一步")} ▶</button>
        <input type="range" class="bx-slider" min="0" max="${T - 1}" value="0" aria-label="Frame step">
        <span class="bx-step-readout bx-frame-label"></span>
        <span style="margin-left:auto;display:inline-flex;align-items:center;gap:4px;">
          <span>${bi("Speed:", "速度：")}</span>
          <button type="button" class="bx-btn bx-spd" data-fps="5">0.5×</button>
          <button type="button" class="bx-btn bx-spd is-on" data-fps="10">1×</button>
          <button type="button" class="bx-btn bx-spd" data-fps="20">2×</button>
          <button type="button" class="bx-btn bx-spd" data-fps="35">4×</button>
        </span>
      </div>
      <div class="bx-presets">
        <span class="bx-presets-label">${bi("Jump to event:", "跳转至关键帧：")}</span>
        <button type="button" class="bx-chip" data-step="0">${bi("Start (step 0)", "开局 (第 0 步)")}</button>
        <button type="button" class="bx-chip" data-step="5">${bi("Paddle catch #1 (step 5)", "首次接球 (第 5 步)")}</button>
        <button type="button" class="bx-chip" data-step="9">${bi("Wall bounce (step 9)", "墙壁反弹 (第 9 步)")}</button>
        <button type="button" class="bx-chip" data-step="11">${bi("Brick hit (step 11)", "击碎砖块 (第 11 步)")}</button>
        <button type="button" class="bx-chip" data-step="358">${bi("30th brick cleared (step 358)", "清空 30 块砖 (第 358 步)")}</button>
        <button type="button" class="bx-chip" data-step="366">${bi("New rack refill (step 366)", "刷新第二轮砖块 (第 366 步)")}</button>
      </div>
      <div class="bx-anim-grid">
        <div class="bx-board-col">
          <div class="bx-panel-title">${bi("MinAtar Board (10×10)", "MinAtar 棋盘 (10×10)")}</div>
          <svg viewBox="0 0 220 225" width="220" height="225" class="bx-board-svg" role="img" aria-label="Breakout 10x10 board"></svg>
          <div class="bx-legend" style="margin-top:6px;font-size:12px;gap:4px 10px;">
            <span class="bx-leg-item"><span class="bx-swatch" style="background:#ffc23d"></span>${bi("Brick", "砖块")}</span>
            <span class="bx-leg-item"><span class="bx-swatch" style="background:#f4f6ff;border-color:#333"></span>${bi("Ball", "球")}</span>
            <span class="bx-leg-item"><span class="bx-swatch" style="background:#8d9cbf"></span>${bi("Trail", "轨迹")}</span>
            <span class="bx-leg-item"><span class="bx-swatch" style="background:#4c8dff"></span>${bi("Paddle", "挡板")}</span>
            <span class="bx-leg-item"><span class="bx-swatch" style="background:#36d399"></span>${bi("True landing col", "真实落点列")}</span>
          </div>
          <div class="bx-board-meta"></div>
        </div>
        <div class="bx-units-col">
          <div class="bx-panel-title">
            ${bi("Hidden activations: tanh ∈ [−1, +1] (hover any unit for evidence)", "隐藏层激活值：tanh ∈ [−1, +1]（悬停查看证据）")}
          </div>
          <div class="bx-layers-cols">
            <div>
              <div style="font-size:12.5px;font-weight:700;margin-bottom:4px;">${bi("Layer 1: h1 (32 units)", "第一层：h1（32 个单元）")}</div>
              <div class="bx-unit-list bx-h1-list"></div>
              <div class="bx-inactive-block">
                <div class="bx-inactive-head">${bi("h1[6..31]: 26 zero-weight units (0.000)", "h1[6..31]：26 个零权重单元 (0.000)")}</div>
                <div class="bx-inactive-grid bx-h1-inactive"></div>
              </div>
            </div>
            <div>
              <div style="font-size:12.5px;font-weight:700;margin-bottom:4px;">${bi("Layer 2: h2 (32 units)", "第二层：h2（32 个单元）")}</div>
              <div class="bx-unit-list bx-h2-list"></div>
              <div class="bx-inactive-block">
                <div class="bx-inactive-head">${bi("h2[8..31]: 24 zero-weight units (0.000)", "h2[8..31]：24 个零权重单元 (0.000)")}</div>
                <div class="bx-inactive-grid bx-h2-inactive"></div>
              </div>
            </div>
          </div>
        </div>
        <div class="bx-probs-col">
          <div class="bx-panel-title">${bi("Action probabilities & logits", "动作概率与 Logits")}</div>
          <div class="bx-prob-list"></div>
        </div>
      </div>
      <div class="bx-inspector bx-anim-inspector"></div>
      <details class="bx-table-fold">
        <summary>${bi("Show full evidence table for all 14 active hidden units (units.json)", "展开全部 14 个激活隐藏单元的证据汇总表 (units.json)")}</summary>
        <div class="bx-unit-table-holder" style="margin-top:8px;overflow-x:auto;"></div>
      </details>`;

    const playBtn = root.querySelector(".bx-play-btn");
    const prevBtn = root.querySelector(".bx-prev-btn");
    const nextBtn = root.querySelector(".bx-next-btn");
    const slider = root.querySelector(".bx-slider");
    const frameLabel = root.querySelector(".bx-frame-label");
    const boardSvg = root.querySelector(".bx-board-svg");
    const boardMeta = root.querySelector(".bx-board-meta");
    const h1List = root.querySelector(".bx-h1-list");
    const h2List = root.querySelector(".bx-h2-list");
    const h1Inactive = root.querySelector(".bx-h1-inactive");
    const h2Inactive = root.querySelector(".bx-h2-inactive");
    const probList = root.querySelector(".bx-prob-list");
    const inspector = root.querySelector(".bx-anim-inspector");
    const tableHolder = root.querySelector(".bx-unit-table-holder");

    let curStep = 0;
    let playing = false;
    let fps = 10;
    let timer = null;
    let selectedUnit = h1Map[0];

    // Build active h1 rows (0..5)
    const h1RowEls = {};
    unitsData.h1_units.forEach((uObj) => {
      const row = document.createElement("div");
      row.className = "bx-unit-row" + (uObj === selectedUnit ? " is-active" : "");
      row.innerHTML = `
        <span class="bx-unit-id">h1[${uObj.unit}]</span>
        <span class="bx-unit-cell"></span>
        <span class="bx-unit-name">${bi(esc(uObj.short_en), esc(uObj.short_zh))}</span>
        <span class="bx-unit-val">0.000</span>`;
      const select = () => {
        selectedUnit = uObj;
        updateSelectionHighlight();
        renderInspector(inspector, selectedUnit, frames[curStep].h1[uObj.unit]);
      };
      row.addEventListener("mouseenter", select);
      row.addEventListener("click", select);
      h1List.appendChild(row);
      h1RowEls[uObj.unit] = {
        row,
        cell: row.querySelector(".bx-unit-cell"),
        val: row.querySelector(".bx-unit-val")
      };
    });

    // Build inactive h1 cells (6..31)
    for (let u = 6; u < 32; u++) {
      const cell = document.createElement("div");
      cell.className = "bx-inactive-cell";
      cell.title = `h1[${u}] = 0.000 (unused)`;
      cell.addEventListener("mouseenter", () => {
        selectedUnit = { layer: 1, unit: u, active: false };
        updateSelectionHighlight();
        renderInspector(inspector, selectedUnit, 0.0);
      });
      h1Inactive.appendChild(cell);
    }

    // Build active h2 rows (0..7)
    const h2RowEls = {};
    unitsData.h2_units.forEach((uObj) => {
      const row = document.createElement("div");
      row.className = "bx-unit-row";
      row.innerHTML = `
        <span class="bx-unit-id">h2[${uObj.unit}]</span>
        <span class="bx-unit-cell"></span>
        <span class="bx-unit-name">${bi(esc(uObj.short_en), esc(uObj.short_zh))}</span>
        <span class="bx-unit-val">-1.000</span>`;
      const select = () => {
        selectedUnit = uObj;
        updateSelectionHighlight();
        renderInspector(inspector, selectedUnit, frames[curStep].h2[uObj.unit]);
      };
      row.addEventListener("mouseenter", select);
      row.addEventListener("click", select);
      h2List.appendChild(row);
      h2RowEls[uObj.unit] = {
        row,
        cell: row.querySelector(".bx-unit-cell"),
        val: row.querySelector(".bx-unit-val")
      };
    });

    // Build inactive h2 cells (8..31)
    for (let u = 8; u < 32; u++) {
      const cell = document.createElement("div");
      cell.className = "bx-inactive-cell";
      cell.title = `h2[${u}] = 0.000 (unused)`;
      cell.addEventListener("mouseenter", () => {
        selectedUnit = { layer: 2, unit: u, active: false };
        updateSelectionHighlight();
        renderInspector(inspector, selectedUnit, 0.0);
      });
      h2Inactive.appendChild(cell);
    }

    function updateSelectionHighlight() {
      Object.entries(h1RowEls).forEach(([u, el]) => {
        el.row.classList.toggle("is-active", selectedUnit.layer === 1 && Number(u) === selectedUnit.unit);
      });
      Object.entries(h2RowEls).forEach(([u, el]) => {
        el.row.classList.toggle("is-active", selectedUnit.layer === 2 && Number(u) === selectedUnit.unit);
      });
    }

    // Build 6 action probability rows
    const actNamesEn = ["0: no-op", "1: left", "2: up", "3: right", "4: down", "5: fire"];
    const actNamesZh = ["0: 不动 (no-op)", "1: 左移 (left)", "2: 上 (up)", "3: 右移 (right)", "4: 下 (down)", "5: 发射 (fire)"];
    const probRowEls = [];
    for (let a = 0; a < 6; a++) {
      const r = document.createElement("div");
      r.className = "bx-prob-row";
      r.innerHTML = `
        <div class="bx-prob-top">
          <span class="bx-prob-name">${bi(actNamesEn[a], actNamesZh[a])}<span class="bx-chosen-slot"></span></span>
          <span class="bx-prob-nums"></span>
        </div>
        <div class="bx-prob-track"><div class="bx-prob-fill" style="width:0%"></div></div>`;
      probList.appendChild(r);
      probRowEls.push({
        row: r,
        slot: r.querySelector(".bx-chosen-slot"),
        nums: r.querySelector(".bx-prob-nums"),
        fill: r.querySelector(".bx-prob-fill")
      });
    }

    // Build accessible summary table of all 14 active hidden units
    const allActive = [...unitsData.h1_units, ...unitsData.h2_units];
    tableHolder.innerHTML = `
      <table class="results" style="width:100%;font-size:13px;">
        <thead>
          <tr>
            <th>${bi("Unit", "单元")}</th>
            <th>${bi("Label", "功能标签")}</th>
            <th>${bi("Weight pattern", "权重模式")}</th>
            <th>${bi("Empirical verification (380 frames)", "380 帧实证检验")}</th>
          </tr>
        </thead>
        <tbody>
          ${allActive.map((u) => `
            <tr>
              <td><code>h${u.layer}[${u.unit}]</code></td>
              <td><b>${bi(esc(u.label_en), esc(u.label_zh))}</b></td>
              <td>${bi(esc(u.evidence.weight_pattern_en), esc(u.evidence.weight_pattern_zh))}</td>
              <td>${bi(esc(u.evidence.empirical.summary_en), esc(u.evidence.empirical.summary_zh))}</td>
            </tr>`).join("")}
        </tbody>
      </table>`;

    function drawBoard(fr) {
      boardSvg.replaceChildren();
      const ox = 18, oy = 18, cs = 19.4;
      // Background
      svgEl("rect", { x: 0, y: 0, width: 220, height: 225, fill: "#0b1020" }, boardSvg);

      // Axis ticks 0..9
      for (let i = 0; i < 10; i++) {
        const tx = svgEl("text", {
          x: ox + i * cs + cs / 2, y: 13, "text-anchor": "middle",
          "font-size": 9.5, "font-family": "var(--font-mono)", fill: "#9aa5b8"
        }, boardSvg);
        tx.textContent = String(i);
        const ty = svgEl("text", {
          x: 9, y: oy + i * cs + cs / 2 + 3.5, "text-anchor": "middle",
          "font-size": 9.5, "font-family": "var(--font-mono)", fill: "#9aa5b8"
        }, boardSvg);
        ty.textContent = String(i);
      }

      // Faint grid cells
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          svgEl("rect", {
            x: ox + c * cs + 0.6,
            y: oy + r * cs + 0.6,
            width: cs - 1.2,
            height: cs - 1.2,
            fill: "#131b33",
            rx: 1.5
          }, boardSvg);
        }
      }

      // True landing column guide + marker at row 9 (and catch_col bracket)
      if (fr.landing_col != null) {
        const lc = fr.landing_col;
        const cc = fr.catch_col != null ? fr.catch_col : lc;
        const lx = ox + lc * cs + cs / 2;
        svgEl("line", {
          x1: lx, y1: oy + 4 * cs, x2: lx, y2: oy + 9 * cs + cs / 2,
          stroke: "#36d399", "stroke-width": 1.3, "stroke-dasharray": "3 3", "stroke-opacity": 0.75
        }, boardSvg);
        // Catch interval outline on row 8/9 for catch_col (direction-preserving catch column)
        if (cc !== lc) {
          svgEl("rect", {
            x: ox + cc * cs + 2,
            y: oy + 9 * cs + 2,
            width: cs - 4,
            height: cs - 4,
            fill: "none",
            stroke: "#36d399",
            "stroke-width": 1.3,
            "stroke-dasharray": "2 2",
            rx: 2.5
          }, boardSvg);
        }
        // Primary landing marker on row 9
        svgEl("rect", {
          x: ox + lc * cs + 1.5,
          y: oy + 9 * cs + 1.5,
          width: cs - 3,
          height: cs - 3,
          fill: "rgba(54, 211, 153, 0.22)",
          stroke: "#36d399",
          "stroke-width": 2,
          rx: 3
        }, boardSvg);
      }

      // Bricks (channel 2)
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (fr.brick_grid[r][c]) {
            svgEl("rect", {
              x: ox + c * cs + 1.6,
              y: oy + r * cs + 2.2,
              width: cs - 3.2,
              height: cs - 4.4,
              fill: "#ffc23d",
              rx: 2
            }, boardSvg);
          }
        }
      }

      // Trail (channel 4)
      const [tr, tc] = fr.trail;
      svgEl("rect", {
        x: ox + tc * cs + cs * 0.28,
        y: oy + tr * cs + cs * 0.28,
        width: cs * 0.44,
        height: cs * 0.44,
        fill: "#8d9cbf",
        "fill-opacity": 0.75,
        rx: 2
      }, boardSvg);

      // Paddle (channel 0, row 9)
      svgEl("rect", {
        x: ox + fr.paddle_col * cs + 1.8,
        y: oy + 9 * cs + 2.5,
        width: cs - 3.6,
        height: cs - 5,
        fill: "#4c8dff",
        rx: 2.5
      }, boardSvg);

      // Ball (channel 3)
      const [br, bc] = fr.ball;
      svgEl("rect", {
        x: ox + bc * cs + 2.2,
        y: oy + br * cs + 2.2,
        width: cs - 4.4,
        height: cs - 4.4,
        fill: "#f4f6ff",
        stroke: "#0b1020",
        "stroke-width": 1.2,
        rx: 3
      }, boardSvg);
    }

    function renderFrame(idx) {
      curStep = Math.max(0, Math.min(T - 1, idx));
      slider.value = String(curStep);
      const fr = frames[curStep];

      frameLabel.innerHTML = bi(
        `Step ${fr.step}/${T - 1} · Score: ${fr.score}`,
        `第 ${fr.step}/${T - 1} 步 · 得分：${fr.score}`
      );

      drawBoard(fr);

      const dirArrows = ["↖", "↗", "↘", "↙"];
      const arrow = dirArrows[fr.ball_dir] || "";
      boardMeta.innerHTML = `
        <div>${bi("Ball:", "球位置：")} <code>(${fr.ball[0]}, ${fr.ball[1]}) ${arrow}</code> &middot; ${bi("Trail:", "轨迹：")} <code>(${fr.trail[0]}, ${fr.trail[1]})</code></div>
        <div>${bi("Paddle col:", "挡板列：")} <code>${fr.paddle_col}</code> &middot; ${bi("Row-9 landing col:", "第 9 行落点列：")} <code>${fr.landing_col}</code> ${bi(`(row-8 catch: <code>${fr.catch_col}</code>)`, `（第 8 行保向接球列：<code>${fr.catch_col}</code>）`)}</div>`;

      // Update h1 active units
      unitsData.h1_units.forEach((uObj) => {
        const v = fr.h1[uObj.unit];
        const el = h1RowEls[uObj.unit];
        el.cell.style.backgroundColor = divColor(v, uObj.role === "comparator" ? 0.18 : 1.0);
        el.val.textContent = (v >= 0 ? "+" : "") + v.toFixed(3);
      });

      // Update h2 active units
      unitsData.h2_units.forEach((uObj) => {
        const v = fr.h2[uObj.unit];
        const el = h2RowEls[uObj.unit];
        el.cell.style.backgroundColor = divColor(v, 1.0);
        el.val.textContent = (v >= 0 ? "+" : "") + v.toFixed(3);
      });

      // Softmax probabilities from logits
      const z = fr.logits;
      const zMax = Math.max(...z);
      const exps = z.map((v) => Math.exp(v - zMax));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((e) => e / sumExp);

      for (let a = 0; a < 6; a++) {
        const pct = probs[a] * 100;
        const pe = probRowEls[a];
        const isChosen = fr.action === a;
        pe.row.classList.toggle("is-chosen", isChosen);
        pe.fill.style.width = `${Math.max(pct, pct > 0.1 ? 2 : 0).toFixed(1)}%`;
        pe.nums.textContent = `${pct.toFixed(1)}% (z=${z[a] >= 0 ? "+" : ""}${z[a].toFixed(0)})`;
        pe.slot.innerHTML = isChosen
          ? `<span class="bx-chosen-badge">${bi("chosen", "执行")}</span>`
          : "";
      }

      // Update inspector with current activation
      if (selectedUnit) {
        let actVal = 0.0;
        if (selectedUnit.active) {
          actVal = selectedUnit.layer === 1 ? fr.h1[selectedUnit.unit] : fr.h2[selectedUnit.unit];
        }
        renderInspector(inspector, selectedUnit, actVal);
      }
    }

    function setPlaying(on) {
      playing = on;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      playBtn.innerHTML = playing ? bi("⏸ Pause", "⏸ 暂停") : bi("▶ Play", "▶ 播放");
      if (playing) tick();
    }

    function tick() {
      if (!playing) return;
      const next = curStep + 1 < T ? curStep + 1 : 0;
      renderFrame(next);
      timer = setTimeout(tick, 1000 / fps);
    }

    playBtn.addEventListener("click", () => setPlaying(!playing));
    prevBtn.addEventListener("click", () => {
      setPlaying(false);
      renderFrame((curStep - 1 + T) % T);
    });
    nextBtn.addEventListener("click", () => {
      setPlaying(false);
      renderFrame((curStep + 1) % T);
    });
    slider.addEventListener("input", () => {
      setPlaying(false);
      renderFrame(Number(slider.value));
    });

    root.querySelectorAll(".bx-spd").forEach((btn) => {
      btn.addEventListener("click", () => {
        root.querySelectorAll(".bx-spd").forEach((b) => b.classList.remove("is-on"));
        btn.classList.add("is-on");
        fps = Number(btn.dataset.fps) || 10;
      });
    });

    root.querySelectorAll(".bx-chip[data-step]").forEach((chip) => {
      chip.addEventListener("click", () => {
        setPlaying(false);
        renderFrame(Number(chip.dataset.step) || 0);
      });
    });

    renderFrame(0);
  }

  async function loadJsonWithFallback(url, fallbackObj) {
    if (typeof location !== "undefined" && location.protocol === "file:") {
      return fallbackObj;
    }
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(String(res.status));
      return await res.json();
    } catch (_e) {
      return fallbackObj;
    }
  }

  async function init() {
    const staticEl = document.getElementById("breakout-static-circuit");
    const animEl = document.getElementById("breakout-rollout-anim");
    if (!staticEl && !animEl) return;

    const [unitsData, framesData] = await Promise.all([
      loadJsonWithFallback("figs/breakout_explainer/units.json", FALLBACK_UNITS),
      loadJsonWithFallback("figs/breakout_explainer/frames.json", FALLBACK_FRAMES)
    ]);

    if (staticEl) mountStaticDiagram(staticEl, unitsData);
    if (animEl) mountAnimation(animEl, unitsData, framesData);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""


def main():
    with open(UNITS_JSON, "r", encoding="utf-8") as f:
        units_data = json.load(f)
    with open(FRAMES_JSON, "r", encoding="utf-8") as f:
        frames_data = json.load(f)

    header = (
        "// Self-contained Breakout circuit explainer (static SVG diagram + rollout animation).\n"
        "// Generated by exp5/explainer/build_explainer_js.py from units.json and frames.json.\n"
        "// Works both over http(s):// (via fetch) and directly from file:// (embedded fallback).\n"
        f"const FALLBACK_UNITS = {json.dumps(units_data, separators=(',', ':'), ensure_ascii=False)};\n"
        f"const FALLBACK_FRAMES = {json.dumps(frames_data, separators=(',', ':'), ensure_ascii=False)};\n"
    )

    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write(header + JS_BODY)

    size_kb = os.path.getsize(OUT_JS) / 1024.0
    print(f"Wrote {OUT_JS} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
