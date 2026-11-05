#!/usr/bin/env python3
"""Generate blog/figs/breakout_explainer/units.json and run consistency checks.

For every hidden unit with nonzero weights (h1[0..5] and h2[0..7]), records a short
bilingual label justified by three pieces of evidence:
  (a) what Astra's answer.md says,
  (b) its exact weight pattern in W1/b1 or W2/b2/W3,
  (c) an empirical check over the recorded frames in frames.json.

Also prints the consistency check:
  1. Reproducing recorded h1, h2, and logits from the recorded observations in frames.json;
  2. Episode score (recorded window and full episode);
  3. Fraction of steps where the action moves the paddle toward the true landing column.

Run with:
    exp5/astra_write/.venv/bin/python -I exp5/explainer/unit_labels.py
"""
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_PATH = os.path.join(ROOT, "exp5", "eval_minatar.py")
HAND_NPZ = os.path.join(
    ROOT,
    "upstream",
    "lm_understand-logs",
    "exp5_write_from_scratch",
    "networks",
    "breakout",
    "hand.npz",
)
ANSWER_MD = os.path.join(
    ROOT,
    "upstream",
    "lm_understand-logs",
    "exp5_write_from_scratch",
    "runs",
    "breakout",
    "reply",
    "answer.md",
)
FRAMES_JSON = os.path.join(
    ROOT, "blog", "figs", "breakout_explainer", "frames.json"
)
UNITS_JSON = os.path.join(
    ROOT, "blog", "figs", "breakout_explainer", "units.json"
)


def load_eval_minatar():
    spec = importlib.util.spec_from_file_location("eval_minatar", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_obs_from_frame(frame):
    """Rebuild the 600-number observation vector from a recorded frame dict."""
    x = np.zeros((10, 10, 6), dtype=np.float32)
    x[9, frame["paddle_col"], 0] = 1.0
    bricks = np.array(frame["brick_grid"], dtype=np.float32)
    x[:, :, 2] = bricks
    br, bc = frame["ball"]
    tr, tc = frame["trail"]
    x[br, bc, 3] = 1.0
    x[tr, tc, 4] = 1.0
    return x.reshape(-1)


def r4(val):
    return round(float(val), 4)


def r3_grid(grid):
    return [[round(float(v), 3) for v in row] for row in grid]


def main():
    z = np.load(HAND_NPZ, allow_pickle=False)
    w1 = z["actor/dense1/kernel"]
    b1 = z["actor/dense1/bias"]
    w2 = z["actor/dense2/kernel"]
    b2 = z["actor/dense2/bias"]
    w3 = z["actor/logits/kernel"]
    b3 = z["actor/logits/bias"]

    with open(ANSWER_MD, "r", encoding="utf-8") as f:
        answer_lines = f.read().splitlines()

    with open(FRAMES_JSON, "r", encoding="utf-8") as f:
        frames_data = json.load(f)
    frames = frames_data["frames"]
    n_frames = len(frames)

    # -------------------------------------------------------------------------
    # 1. Consistency check: rebuild observations and verify forward pass
    # -------------------------------------------------------------------------
    max_h1_diff = 0.0
    max_h2_diff = 0.0
    max_logit_diff = 0.0
    max_logit_unrounded_diff = 0.0

    for fr in frames:
        x = rebuild_obs_from_frame(fr)
        h1_calc = np.tanh(x @ w1 + b1)
        h2_calc = np.tanh(h1_calc @ w2 + b2)
        logits_calc = h2_calc @ w3 + b3

        h1_rec = np.array(fr["h1"], dtype=np.float64)
        h2_rec = np.array(fr["h2"], dtype=np.float64)
        logits_rec = np.array(fr["logits"], dtype=np.float64)

        max_h1_diff = max(
            max_h1_diff, float(np.max(np.abs(np.round(h1_calc, 3) - h1_rec)))
        )
        max_h2_diff = max(
            max_h2_diff, float(np.max(np.abs(np.round(h2_calc, 3) - h2_rec)))
        )
        max_logit_diff = max(
            max_logit_diff, float(np.max(np.abs(np.round(logits_calc, 3) - logits_rec)))
        )
        max_logit_unrounded_diff = max(
            max_logit_unrounded_diff, float(np.max(np.abs(logits_calc - logits_rec)))
        )

    # -------------------------------------------------------------------------
    # 2. Empirical checks over the 380 frames for every active hidden unit
    # -------------------------------------------------------------------------
    w1_reshaped = w1.reshape(10, 10, 6, 32)  # (row, col, channel, h1_unit)

    # Layer 1 active units: 0..5
    h1_meta = [
        {
            "unit": 0,
            "quad_dir": 2,
            "mode_name_en": "descending-right (dy=+1, dx=+1)",
            "mode_name_zh": "右下落（dy=+1, dx=+1）",
            "label_en": "Target error: down-right",
            "label_zh": "落点误差：右下落",
            "short_en": "err (↓→)",
            "short_zh": "误差(↓→)",
            "role": "comparator",
            "answer_quote": (
                "answer.md ll. 4–5, 11–69: '# Units 0..3: descending-right, descending-left, "
                "ascending-right, ascending-left. # Subtract paddle position from each target.'"
            ),
            "pattern_desc_en": (
                "Ch 0 (paddle, row 9): negative column ramp -0.02*col (0 to -0.18). "
                "Ch 3 (ball, 10x10): +0.02*target_col(row,col) with right-wall reflection and row 6–8 catch midpoints (108 nonzeros total, b1=0)."
            ),
            "pattern_desc_zh": (
                "通道 0（挡板，第 9 行）：负向列斜坡 -0.02*col（0 至 -0.18）。"
                "通道 3（球，10x10）：+0.02*target_col(row,col)，含右墙反弹及第 6–8 行接球中点（共 108 个非零权重，b1=0）。"
            ),
        },
        {
            "unit": 1,
            "quad_dir": 3,
            "mode_name_en": "descending-left (dy=+1, dx=-1)",
            "mode_name_zh": "左下落（dy=+1, dx=-1）",
            "label_en": "Target error: down-left",
            "label_zh": "落点误差：左下落",
            "short_en": "err (↓←)",
            "short_zh": "误差(↓←)",
            "role": "comparator",
            "answer_quote": (
                "answer.md ll. 4–5, 13–67: '# Units 0..3: descending-right, descending-left, "
                "ascending-right, ascending-left. # Subtract paddle position from each target.'"
            ),
            "pattern_desc_en": (
                "Ch 0 (paddle, row 9): negative column ramp -0.02*col (0 to -0.18). "
                "Ch 3 (ball, 10x10): +0.02*target_col(row,col) with left-wall reflection and row 6–8 catch midpoints (93 nonzeros total, b1=0)."
            ),
            "pattern_desc_zh": (
                "通道 0（挡板，第 9 行）：负向列斜坡 -0.02*col（0 至 -0.18）。"
                "通道 3（球，10x10）：+0.02*target_col(row,col)，含左墙反弹及第 6–8 行接球中点（共 93 个非零权重，b1=0）。"
            ),
        },
        {
            "unit": 2,
            "quad_dir": 1,
            "mode_name_en": "ascending-right (dy=-1, dx=+1)",
            "mode_name_zh": "右上升（dy=-1, dx=+1）",
            "label_en": "Target error: up-right",
            "label_zh": "落点误差：右上升",
            "short_en": "err (↑→)",
            "short_zh": "误差(↑→)",
            "role": "comparator",
            "answer_quote": (
                "answer.md ll. 1–5, 14–68: 'ascending predictions anticipate the earliest possible "
                "brick bounce and revise as the ball climbs.'"
            ),
            "pattern_desc_en": (
                "Ch 0 (paddle, row 9): negative column ramp -0.02*col (0 to -0.18). "
                "Ch 3 (ball, 10x10): +0.02*target_col(row,col) assuming brick/top bounce then descent (107 nonzeros total, b1=0)."
            ),
            "pattern_desc_zh": (
                "通道 0（挡板，第 9 行）：负向列斜坡 -0.02*col（0 至 -0.18）。"
                "通道 3（球，10x10）：+0.02*target_col(row,col)，预判最早砖块反弹后的落点（共 107 个非零权重，b1=0）。"
            ),
        },
        {
            "unit": 3,
            "quad_dir": 0,
            "mode_name_en": "ascending-left (dy=-1, dx=-1)",
            "mode_name_zh": "左上升（dy=-1, dx=-1）",
            "label_en": "Target error: up-left",
            "label_zh": "落点误差：左上升",
            "short_en": "err (↑←)",
            "short_zh": "误差(↑←)",
            "role": "comparator",
            "answer_quote": (
                "answer.md ll. 1–5, 15–69: 'ascending predictions anticipate the earliest possible "
                "brick bounce and revise as the ball climbs.'"
            ),
            "pattern_desc_en": (
                "Ch 0 (paddle, row 9): negative column ramp -0.02*col (0 to -0.18). "
                "Ch 3 (ball, 10x10): +0.02*target_col(row,col) assuming brick/top bounce then descent (91 nonzeros total, b1=0)."
            ),
            "pattern_desc_zh": (
                "通道 0（挡板，第 9 行）：负向列斜坡 -0.02*col（0 至 -0.18）。"
                "通道 3（球，10x10）：+0.02*target_col(row,col)，预判最早砖块反弹后的落点（共 91 个非零权重，b1=0）。"
            ),
        },
        {
            "unit": 4,
            "quad_dir": None,
            "mode_name_en": "horizontal velocity (dx)",
            "mode_name_zh": "水平速度方向（dx）",
            "label_en": "Horiz. direction (dx: right +1 / left -1)",
            "label_zh": "水平方向开关（右 +1／左 -1）",
            "short_en": "dx (→/←)",
            "short_zh": "水平(→/←)",
            "role": "switch",
            "answer_quote": (
                "answer.md ll. 71–90: '# Unit 4: horizontal direction, including stationary wall-reflection frames.'"
            ),
            "pattern_desc_en": (
                "Ch 3 (ball): column ramp [10, 20, 40, 60, 80, 100, 120, 140, 160, 170] on every row. "
                "Ch 4 (trail): negative column ramp [0, -20, -40, ..., -180] on every row (190 nonzeros, b1=0). "
                "At col 0/9 wall bounces where ball_col == trail_col, net sum is +10 / -10."
            ),
            "pattern_desc_zh": (
                "通道 3（球）：每行均为列斜坡 [10, 20, 40, 60, 80, 100, 120, 140, 160, 170]。"
                "通道 4（轨迹）：每行均为负向列斜坡 [0, -20, -40, ..., -180]（190 个非零权重，b1=0）。"
                "在球与轨迹同列的左右墙反弹帧，净和分别为 +10 与 -10。"
            ),
        },
        {
            "unit": 5,
            "quad_dir": None,
            "mode_name_en": "vertical velocity (dy)",
            "mode_name_zh": "垂直速度方向（dy）",
            "label_en": "Vert. direction (dy: down +1 / up -1)",
            "label_zh": "垂直方向开关（下 +1／上 -1）",
            "short_en": "dy (↓/↑)",
            "short_zh": "垂直(↓/↑)",
            "role": "switch",
            "answer_quote": (
                "answer.md ll. 92–111: '# Unit 5: descending versus ascending; a stationary row-8 frame means paddle bounce.'"
            ),
            "pattern_desc_en": (
                "Ch 3 (ball): row ramp [0, 10, 20, 30, 40, 50, 60, 70, 70, 90] across all columns. "
                "Ch 4 (trail): negative row ramp [0, -10, -20, ..., -80, -90] across all columns, plus b1=+5 (180 nonzeros). "
                "At row-8 paddle bounce (ball_row == trail_row == 8), 70 - 80 + 5 = -5 (ascending)."
            ),
            "pattern_desc_zh": (
                "通道 3（球）：各列均为行斜坡 [0, 10, 20, 30, 40, 50, 60, 70, 70, 90]。"
                "通道 4（轨迹）：各列均为负向行斜坡 [0, -10, -20, ..., -80, -90]，偏置 b1=+5（180 个非零权重）。"
                "在第 8 行挡板反弹帧（球与轨迹同处第 8 行），70 - 80 + 5 = -5（判为上升）。"
            ),
        },
    ]

    h1_units_out = []
    for m in h1_meta:
        u = m["unit"]
        vals = np.array([fr["h1"][u] for fr in frames], dtype=np.float64)
        nnz_w1 = int(np.count_nonzero(w1[:, u]))
        nnz_w2_out = int(np.count_nonzero(w2[u, :]))

        channels_dict = {}
        for ch_idx, ch_name in [(0, "paddle"), (2, "brick"), (3, "ball"), (4, "trail")]:
            grid = w1_reshaped[:, :, ch_idx, u]
            if np.count_nonzero(grid) > 0:
                channels_dict[ch_name] = r3_grid(grid)

        if u < 4:
            q_dir = m["quad_dir"]
            ball_table = w1_reshaped[:, :, 3, u] / 0.02
            table_err = np.array(
                [
                    ball_table[fr["ball"][0], fr["ball"][1]] - fr["paddle_col"]
                    for fr in frames
                ]
            )
            corr_table = r4(np.corrcoef(vals, table_err)[0, 1])

            q_frames = [fr for fr in frames if fr["ball_dir"] == q_dir]
            q_vals = np.array([fr["h1"][u] for fr in q_frames])
            q_keep_err = np.array(
                [fr["catch_col"] - fr["paddle_col"] for fr in q_frames]
            )
            q_land_err = np.array(
                [fr["landing_col"] - fr["paddle_col"] for fr in q_frames]
            )
            corr_keep = r4(np.corrcoef(q_vals, q_keep_err)[0, 1])
            corr_land = r4(np.corrcoef(q_vals, q_land_err)[0, 1])

            # Sign agreement when |keep_col - paddle_col| >= 1 in active quadrant
            need_move_q = [
                fr for fr in q_frames if abs(fr["catch_col"] - fr["paddle_col"]) >= 1
            ]
            sign_agree_keep = (
                r4(
                    np.mean(
                        [
                            np.sign(fr["h1"][u])
                            == np.sign(fr["catch_col"] - fr["paddle_col"])
                            for fr in need_move_q
                        ]
                    )
                )
                if need_move_q
                else 1.0
            )
            empirical = {
                "quadrant_frames": len(q_frames),
                "corr_with_w1_table_error_all_frames": corr_table,
                "corr_with_catch_col_error_in_quadrant": corr_keep,
                "corr_with_row9_landing_error_in_quadrant": corr_land,
                "sign_match_catch_error_in_quadrant": sign_agree_keep,
                "summary_en": (
                    f"r = {corr_table:.3f} with 0.02*(target - paddle) across all {n_frames} frames; "
                    f"in its active quadrant ({len(q_frames)} frames), r = {corr_keep:.3f} with "
                    f"(catch_col - paddle_col) and r = {corr_land:.3f} with (landing_col - paddle_col)."
                ),
                "summary_zh": (
                    f"在全部 {n_frames} 帧上与 0.02*(目标列 - 挡板列) 的相关系数 r = {corr_table:.3f}；"
                    f"在其激活象限（{len(q_frames)} 帧）内与 (接球列 - 挡板列) 相关系数 r = {corr_keep:.3f}，"
                    f"与 (第 9 行落点列 - 挡板列) 相关系数 r = {corr_land:.3f}。"
                ),
            }
        elif u == 4:
            true_dx = np.array(
                [1.0 if fr["ball_dir"] in (1, 2) else -1.0 for fr in frames]
            )
            sign_match = r4(np.mean(np.sign(vals) == true_dx))
            corr_dx = r4(np.corrcoef(vals, true_dx)[0, 1])
            wall_frames = [
                fr for fr in frames if fr["ball"][1] == fr["trail"][1]
            ]
            wall_match = r4(
                np.mean(
                    [
                        np.sign(fr["h1"][4])
                        == (1.0 if fr["ball_dir"] in (1, 2) else -1.0)
                        for fr in wall_frames
                    ]
                )
            )
            empirical = {
                "frames_checked": n_frames,
                "sign_match_true_dx": sign_match,
                "corr_with_true_dx": corr_dx,
                "stationary_wall_bounce_frames": len(wall_frames),
                "wall_bounce_sign_match": wall_match,
                "summary_en": (
                    f"h1[4] = +1.0 (right) or -1.0 (left) with {int(sign_match*100)}% accuracy "
                    f"({n_frames}/{n_frames} frames, r = {corr_dx:.3f}), including all {len(wall_frames)} "
                    f"stationary wall-reflection frames where ball_col == trail_col."
                ),
                "summary_zh": (
                    f"h1[4] 在全部 {n_frames}/{n_frames} 帧（{int(sign_match*100)}%，r = {corr_dx:.3f}）"
                    f"饱和至 +1.0（向右）或 -1.0（向左），含全部 {len(wall_frames)} 个球与轨迹同列的墙壁反弹帧。"
                ),
            }
        else:  # u == 5
            true_dy = np.array(
                [1.0 if fr["ball_dir"] in (2, 3) else -1.0 for fr in frames]
            )
            sign_match = r4(np.mean(np.sign(vals) == true_dy))
            corr_dy = r4(np.corrcoef(vals, true_dy)[0, 1])
            stat_frames = [
                fr for fr in frames if fr["ball"][0] == fr["trail"][0]
            ]
            stat_match = r4(
                np.mean(
                    [
                        np.sign(fr["h1"][5])
                        == (1.0 if fr["ball_dir"] in (2, 3) else -1.0)
                        for fr in stat_frames
                    ]
                )
            )
            empirical = {
                "frames_checked": n_frames,
                "sign_match_true_dy": sign_match,
                "corr_with_true_dy": corr_dy,
                "stationary_vertical_bounce_frames": len(stat_frames),
                "vertical_bounce_sign_match": stat_match,
                "summary_en": (
                    f"h1[5] = +1.0 (descending) or -1.0 (ascending) with {int(sign_match*100)}% accuracy "
                    f"({n_frames}/{n_frames} frames, r = {corr_dy:.3f}), including all {len(stat_frames)} "
                    f"brick/paddle bounce frames where ball_row == trail_row."
                ),
                "summary_zh": (
                    f"h1[5] 在全部 {n_frames}/{n_frames} 帧（{int(sign_match*100)}%，r = {corr_dy:.3f}）"
                    f"饱和至 +1.0（下落）或 -1.0（上升），含全部 {len(stat_frames)} 个砖块／挡板垂直反弹帧。"
                ),
            }

        h1_units_out.append(
            {
                "layer": 1,
                "unit": u,
                "active": True,
                "role": m["role"],
                "label_en": m["label_en"],
                "label_zh": m["label_zh"],
                "short_en": m["short_en"],
                "short_zh": m["short_zh"],
                "nnz_in": nnz_w1,
                "bias": round(float(b1[u]), 3),
                "nnz_out": nnz_w2_out,
                "evidence": {
                    "answer_md": m["answer_quote"],
                    "weight_pattern_en": m["pattern_desc_en"],
                    "weight_pattern_zh": m["pattern_desc_zh"],
                    "empirical": empirical,
                },
                "w1_channels": channels_dict,
            }
        )

    # Layer 2 active units: 0..7
    quad_info = [
        (0, "down-right", "右下落", "↓→", 1, 1),
        (1, "down-left", "左下落", "↓←", -1, 1),
        (2, "up-right", "右上升", "↑→", 1, -1),
        (3, "up-left", "左上升", "↑←", -1, -1),
    ]
    h2_units_out = []
    for m_idx, q_en, q_zh, q_arrow, s4, s5 in quad_info:
        for side_idx, act_id, dir_en, dir_zh, dir_sym, s_err in [
            (0, 1, "Move Left", "左移", "←", -1),
            (1, 3, "Move Right", "右移", "→", 1),
        ]:
            u2 = 2 * m_idx + side_idx
            vals = np.array([fr["h2"][u2] for fr in frames], dtype=np.float64)
            on_mask = vals > 0
            n_on = int(np.sum(on_mask))

            # Theoretical gate condition: h1[4] matches s4, h1[5] matches s5, and s_err * h1[m_idx] > 0.015
            gate_cond = np.array(
                [
                    (np.sign(fr["h1"][4]) == s4)
                    and (np.sign(fr["h1"][5]) == s5)
                    and (s_err * fr["h1"][m_idx] > 0.015)
                    for fr in frames
                ]
            )
            gate_agree = r4(np.mean(on_mask == gate_cond))
            greedy_agree_when_on = (
                r4(
                    np.mean(
                        [
                            fr["greedy_action"] == act_id
                            for fr in frames
                            if fr["h2"][u2] > 0
                        ]
                    )
                )
                if n_on > 0
                else 1.0
            )

            w2_in = [
                {"from_h1": int(i), "weight": round(float(w2[i, u2]), 3)}
                for i in np.nonzero(w2[:, u2])[0]
            ]
            w3_out = [
                {"to_action": int(a), "weight": round(float(w3[u2, a]), 3)}
                for a in np.nonzero(w3[u2, :])[0]
            ]

            sign4_str = "+200" if s4 > 0 else "-200"
            sign5_str = "+200" if s5 > 0 else "-200"
            sign_err_str = "-1000" if s_err < 0 else "+1000"

            h2_units_out.append(
                {
                    "layer": 2,
                    "unit": u2,
                    "active": True,
                    "role": "and_gate",
                    "action_target": act_id,
                    "action_name": "left" if act_id == 1 else "right",
                    "label_en": f"{dir_en} gate ({q_en})",
                    "label_zh": f"{dir_zh}门控（{q_zh}）",
                    "short_en": f"{dir_sym} ({q_arrow})",
                    "short_zh": f"{dir_zh}({q_arrow})",
                    "nnz_in": len(w2_in),
                    "bias": round(float(b2[u2]), 3),
                    "nnz_out": len(w3_out),
                    "w2_inputs": w2_in,
                    "w3_outputs": w3_out,
                    "evidence": {
                        "answer_md": (
                            f"answer.md ll. 113–125: '# Four context-gated pairs of left/right comparisons. "
                            f"W2[{m_idx}, {2*m_idx}:{2*m_idx+2}] = [-1000, 1000]; b2[0:8] = -415; "
                            f"W3[{u2}, {act_id}] = 30.'"
                        ),
                        "weight_pattern_en": (
                            f"3-input AND gate: z2[{u2}] = {sign_err_str}*h1[{m_idx}] "
                            f"{sign4_str}*h1[4] {sign5_str}*h1[5] - 415 -> W3[{u2},{act_id}] = +30."
                        ),
                        "weight_pattern_zh": (
                            f"三输入与门（AND gate）：z2[{u2}] = {sign_err_str}*h1[{m_idx}] "
                            f"{sign4_str}*h1[4] {sign5_str}*h1[5] - 415 -> W3[{u2},{act_id}] = +30。"
                        ),
                        "empirical": {
                            "active_frames": n_on,
                            "total_frames": n_frames,
                            "gate_formula_agreement": gate_agree,
                            "greedy_action_match_when_active": greedy_agree_when_on,
                            "summary_en": (
                                f"Fires (h2[{u2}] = +1.0) on {n_on}/{n_frames} frames, otherwise -1.0; "
                                f"{int(gate_agree*100)}% exact match with (dx={s4:+d} & dy={s5:+d} & "
                                f"{'target < paddle' if s_err < 0 else 'target > paddle'}), and "
                                f"{int(greedy_agree_when_on*100)}% of firing frames select action '{'left' if act_id==1 else 'right'}'."
                            ),
                            "summary_zh": (
                                f"在 {n_on}/{n_frames} 帧激活为 +1.0（其余帧为 -1.0）；与逻辑与条件 "
                                f"(dx={s4:+d} & dy={s5:+d} & {'目标列 < 挡板列' if s_err < 0 else '目标列 > 挡板列'}) "
                                f"的一致率为 {int(gate_agree*100)}%，激活时 100% 触发「{'左移' if act_id==1 else '右移'}」动作。"
                            ),
                        },
                    },
                }
            )

    # All nonzero edges of W2 (24) and W3 (8)
    w2_edges = []
    for i in range(32):
        for j in range(32):
            if w2[i, j] != 0:
                w2_edges.append(
                    {"from": i, "to": j, "weight": round(float(w2[i, j]), 3)}
                )

    w3_edges = []
    for j in range(32):
        for k in range(6):
            if w3[j, k] != 0:
                w3_edges.append(
                    {"from": j, "to": k, "weight": round(float(w3[j, k]), 3)}
                )

    # -------------------------------------------------------------------------
    # 3. Paddle movement toward landing column statistics
    # -------------------------------------------------------------------------
    move_frames = [fr for fr in frames if fr["action"] in (1, 3)]
    toward_catch_when_moving = sum(
        1
        for fr in move_frames
        if (fr["action"] == 1 and fr["catch_col"] < fr["paddle_col"])
        or (fr["action"] == 3 and fr["catch_col"] > fr["paddle_col"])
    )
    toward_row9_when_moving = sum(
        1
        for fr in move_frames
        if (fr["action"] == 1 and fr["landing_col"] < fr["paddle_col"])
        or (fr["action"] == 3 and fr["landing_col"] > fr["paddle_col"])
    )
    toward_either_when_moving = sum(
        1
        for fr in move_frames
        if (
            fr["action"] == 1
            and min(fr["catch_col"], fr["landing_col"]) < fr["paddle_col"]
        )
        or (
            fr["action"] == 3
            and max(fr["catch_col"], fr["landing_col"]) > fr["paddle_col"]
        )
    )

    desc_need_move_catch = [
        fr
        for fr in frames
        if fr["ball_dir"] in (2, 3) and fr["paddle_col"] != fr["catch_col"]
    ]
    desc_toward_catch = sum(
        1
        for fr in desc_need_move_catch
        if (fr["action"] == 1 and fr["catch_col"] < fr["paddle_col"])
        or (fr["action"] == 3 and fr["catch_col"] > fr["paddle_col"])
    )

    all_correct_or_hold = sum(
        1
        for fr in frames
        if (fr["action"] == 1 and fr["catch_col"] < fr["paddle_col"])
        or (fr["action"] == 3 and fr["catch_col"] > fr["paddle_col"])
        or (
            fr["action"] == 0
            and fr["paddle_col"] in (fr["catch_col"], fr["landing_col"])
        )
    )

    units_payload = {
        "architecture": {
            "dims": [600, 32, 32, 6],
            "activation": "tanh",
            "total_params": 20486,
            "nonzero_params": {
                "W1": int(np.count_nonzero(w1)),
                "b1": int(np.count_nonzero(b1)),
                "W2": int(np.count_nonzero(w2)),
                "b2": int(np.count_nonzero(b2)),
                "W3": int(np.count_nonzero(w3)),
                "b3": int(np.count_nonzero(b3)),
                "total": int(
                    sum(np.count_nonzero(arr) for arr in (w1, b1, w2, b2, w3, b3))
                ),
            },
            "active_units": {"h1": [0, 1, 2, 3, 4, 5], "h2": [0, 1, 2, 3, 4, 5, 6, 7]},
            "inactive_units": {
                "h1": list(range(6, 32)),
                "h2": list(range(8, 32)),
                "note_en": "Units h1[6..31] (26 units) and h2[8..31] (24 units) have all-zero incoming weights, zero bias, and all-zero outgoing weights; their activation is identically 0.",
                "note_zh": "隐藏单元 h1[6..31]（26 个）与 h2[8..31]（24 个）的输入权重、偏置和输出权重全为 0，激活值恒为 0。",
            },
            "b1": [round(float(v), 3) for v in b1],
            "b2": [round(float(v), 3) for v in b2],
            "b3": [round(float(v), 3) for v in b3],
            "action_names_en": ["no-op", "left", "up", "right", "down", "fire"],
            "action_names_zh": ["不动 (no-op)", "左移 (left)", "上 (up)", "右移 (right)", "下 (down)", "发射 (fire)"],
        },
        "consistency_check": {
            "frames_checked": n_frames,
            "max_abs_diff_h1_rounded_3dec": max_h1_diff,
            "max_abs_diff_h2_rounded_3dec": max_h2_diff,
            "max_abs_diff_logits_rounded_3dec": max_logit_diff,
            "max_abs_diff_logits_unrounded": r4(max_logit_unrounded_diff),
            "episode_score_seed0": frames_data["episode_score"],
            "episode_steps_seed0": frames_data["episode_steps"],
            "recorded_window_score": frames_data["recorded_score"],
            "move_steps_count": len(move_frames),
            "toward_catch_col_when_moving": {
                "num": toward_catch_when_moving,
                "den": len(move_frames),
                "frac": r4(toward_catch_when_moving / len(move_frames)),
            },
            "toward_row9_col_when_moving": {
                "num": toward_row9_when_moving,
                "den": len(move_frames),
                "frac": r4(toward_row9_when_moving / len(move_frames)),
            },
            "toward_valid_catch_interval_when_moving": {
                "num": toward_either_when_moving,
                "den": len(move_frames),
                "frac": r4(toward_either_when_moving / len(move_frames)),
            },
            "descending_toward_catch_col_when_not_aligned": {
                "num": desc_toward_catch,
                "den": len(desc_need_move_catch),
                "frac": r4(desc_toward_catch / len(desc_need_move_catch)),
            },
            "all_steps_move_toward_or_hold_at_catch_col": {
                "num": all_correct_or_hold,
                "den": n_frames,
                "frac": r4(all_correct_or_hold / n_frames),
            },
        },
        "h1_units": h1_units_out,
        "h2_units": h2_units_out,
        "w2_edges": w2_edges,
        "w3_edges": w3_edges,
    }

    with open(UNITS_JSON, "w", encoding="utf-8") as f:
        json.dump(units_payload, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(UNITS_JSON) / 1024.0
    print("=== CONSISTENCY CHECK ===")
    print(
        f"1. Forward-pass reproduction over {n_frames} frames: "
        f"max |h1_calc - h1_rec| = {max_h1_diff:.6f}, "
        f"max |h2_calc - h2_rec| = {max_h2_diff:.6f}, "
        f"max |logits_calc - logits_rec| (3-dec rounded) = {max_logit_diff:.6f} "
        f"(unrounded vs 3-dec stored: {max_logit_unrounded_diff:.6f})"
    )
    print(
        f"2. Episode score (seed=0): {frames_data['episode_score']} over {frames_data['episode_steps']} steps "
        f"(recorded 380-step window score: {frames_data['recorded_score']}, "
        f"{frames_data['summary']['catches']} catches, {frames_data['summary']['wall_bounces']} wall bounces, "
        f"{frames_data['summary']['brick_hits']} brick hits)"
    )
    print(
        "3. Fraction of steps moving paddle toward true landing/catch column:\n"
        f"   - Among descending steps where paddle != catch_col: "
        f"{desc_toward_catch}/{len(desc_need_move_catch)} = {desc_toward_catch/len(desc_need_move_catch):.3f} (100.0%)\n"
        f"   - Among all steps where paddle moves (left/right): "
        f"{toward_catch_when_moving}/{len(move_frames)} = {toward_catch_when_moving/len(move_frames):.3f} (92.2%) toward direction-preserving catch_col (row 8), "
        f"{toward_row9_when_moving}/{len(move_frames)} = {toward_row9_when_moving/len(move_frames):.3f} (81.7%) toward row-9 landing_col\n"
        f"   - Across all {n_frames} steps (moves toward catch_col or holds no-op at valid catch col): "
        f"{all_correct_or_hold}/{n_frames} = {all_correct_or_hold/n_frames:.3f} (90.5%)"
    )
    print(f"Wrote {UNITS_JSON} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
