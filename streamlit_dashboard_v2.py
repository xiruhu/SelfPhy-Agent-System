"""
SelfPhy-Agent-System 评测结果可视化仪表盘 v2
"""
import base64
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT = Path("/mnt/cpfs/yqh/HXR/hxr_SelfPhy-Agent-System")
EXAMS_DIR  = PROJECT / "outputs/exams"
DATA_DIR   = PROJECT / "data/VL-LN-Bench/extracted"
RUNS_DIR   = PROJECT / "outputs/runs"
LEGACY_ANS = PROJECT / "outputs/answers"
LEGACY_REP = PROJECT / "outputs/reports"

CAPABILITIES = [
    "egocentric_memory", "spatial_transformation", "occlusion_reasoning",
    "trajectory_backtracking", "distance_estimation",
]
CAP_LABELS = {
    "egocentric_memory":        "空间记忆",
    "spatial_transformation":   "空间变换",
    "occlusion_reasoning":      "遮挡推理",
    "trajectory_backtracking":  "轨迹回溯",
    "distance_estimation":      "距离估算",
}
ERROR_LABELS = {
    "direction_calc_error":           "方向计算错误",
    "rotation_sense_error":           "旋转方向错误",
    "rotation_translation_confusion": "旋转/平移混淆",
    "memory_decay":                   "记忆衰减",
    "object_hallucination":           "物体幻觉",
    "fov_misunderstanding":           "视野误解",
    "unknown":                        "未知/截断",
}
DIFF_LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
MODEL_COLORS = {"kimi": "#E74C3C", "doubao": "#27AE60"}

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def list_runs():
    runs = []
    if RUNS_DIR.exists():
        runs = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)
    if LEGACY_ANS.exists() and any(LEGACY_ANS.glob("result_*.json")):
        runs.append("legacy (outputs/answers/)")
    return runs

def get_run_dirs(run_id):
    if run_id.startswith("legacy"):
        return LEGACY_ANS, LEGACY_REP
    return RUNS_DIR / run_id / "answers", RUNS_DIR / run_id / "reports"

# ── 数据加载：从 exam 补全 kimi 的 capability/difficulty/ground_truth_answer ──

@st.cache_data
def load_all_data(run_id: str):
    ans_dir, rep_dir = get_run_dirs(run_id)

    # 预加载所有 exam 的 question 信息
    exam_q_cache = {}
    for ep in sorted(EXAMS_DIR.glob("exam_*_episode_*.json")):
        scene = ep.stem.replace("exam_", "").rsplit("_episode_", 1)[0]
        with open(ep) as f:
            ex = json.load(f)
        exam_q_cache[scene] = {q["question_id"]: q for q in ex.get("questions", [])}

    results = {}
    for path in sorted(ans_dir.glob("result_*_*_episode_*.json")):
        if "aggregated" in path.stem:
            continue
        parts = path.stem.split("_", 2)
        if len(parts) < 3:
            continue
        model = parts[1]
        scene = parts[2].rsplit("_episode_", 1)[0]
        with open(path) as f:
            data = json.load(f)
        if "model_name" not in data:
            data["model_name"] = data.get("model", model)
        # 补全 kimi 缺少的字段
        q_info = exam_q_cache.get(scene, {})
        for r in data.get("responses", []):
            qid = r.get("question_id", "")
            q   = q_info.get(qid, {})
            if not r.get("capability"):
                r["capability"] = q.get("capability", "unknown")
            if not r.get("difficulty"):
                r["difficulty"] = q.get("difficulty", "unknown")
            if not r.get("question_text"):
                r["question_text"] = q.get("question_text", "")
            if not r.get("ground_truth_answer"):
                r["ground_truth_answer"] = (
                    r.get("ground_truth") or q.get("ground_truth_answer", "")
                )
        results[(model, scene)] = data

    diagnoses = {}
    for path in sorted(rep_dir.glob("diagnosis_*_*_episode_*.json")):
        if "aggregated" in path.stem:
            continue
        parts = path.stem.split("_", 2)
        if len(parts) < 3:
            continue
        model = parts[1]
        scene = parts[2].rsplit("_episode_", 1)[0]
        with open(path) as f:
            data = json.load(f)
        diagnoses[(model, scene)] = data if isinstance(data, list) else []

    return results, diagnoses

@st.cache_data
def load_exams(run_id: str):
    exams_dir = EXAMS_DIR if run_id.startswith("legacy") else RUNS_DIR / run_id / "exams"
    exams = {}
    if not exams_dir.exists():
        return exams
    for path in sorted(exams_dir.glob("exam_*_episode_*.json")):
        scene = path.stem.replace("exam_", "").rsplit("_episode_", 1)[0]
        with open(path) as f:
            exams[scene] = json.load(f)
    return exams

@st.cache_data
def load_metadata(scene: str):
    p = DATA_DIR / scene / "episode_000000" / "metadata.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

def get_models(results):
    return sorted(set(m for m, _ in results))

def get_scenes(results):
    return sorted(set(s for _, s in results))

def agg_cap_acc(results, model):
    cap_c = defaultdict(int)
    cap_t = defaultdict(int)
    for (m, s), data in results.items():
        if m != model:
            continue
        for r in data.get("responses", []):
            cap = r.get("capability", "unknown")
            cap_t[cap] += 1
            if r.get("is_correct", False):
                cap_c[cap] += 1
    return {cap: (cap_c[cap] / cap_t[cap] if cap_t[cap] else 0)
            for cap in set(list(cap_t.keys()) + CAPABILITIES)}

def get_resp(results, model, scene, qid):
    for r in results.get((model, scene), {}).get("responses", []):
        if r.get("question_id") == qid:
            return r
    return None

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SelfPhy 评测仪表盘", page_icon="🧠", layout="wide")
st.title("🧠 SelfPhy-Agent-System 评测仪表盘")
st.caption("第一人称自我中心空间推理能力评测系统 | Supervisor: Claude Sonnet 4.6")

tab_overview, tab_pipeline, tab_diag, tab_results = st.tabs([
    "📊 总体结果", "🎬 完整 Pipeline 演示", "🔬 错题诊断详情", "📋 汇总表格",
])

with st.sidebar:
    st.header("配置")
    runs = list_runs()
    if not runs:
        st.error("未找到评测结果")
        st.stop()
    selected_run = st.selectbox("评测批次", runs, index=0)
    if st.button("🔄 刷新数据缓存"):
        st.cache_data.clear()
        st.rerun()

results, diagnoses = load_all_data(selected_run)
exams = load_exams(selected_run)
models = get_models(results)
scenes = get_scenes(results)

if not results:
    st.error("未找到评测结果文件")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1：总体结果
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    model_stats = {}
    for (m, s), data in results.items():
        if m not in model_stats:
            model_stats[m] = {"q": 0, "c": 0}
        for r in data.get("responses", []):
            model_stats[m]["q"] += 1
            if r.get("is_correct", False):
                model_stats[m]["c"] += 1

    all_q = sum(v["q"] for v in model_stats.values())
    all_c = sum(v["c"] for v in model_stats.values())

    st.subheader("📊 总体概况")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("评测场景数", len(scenes))
    c2.metric("模型数量", len(models))
    c3.metric("总题目数（两模型合计）", all_q)
    c4.metric("总正确数（两模型合计）", all_c)
    c5.metric("综合准确率", f"{all_c/all_q*100:.1f}%" if all_q else "N/A")

    stat_rows = []
    for m in models:
        ms = model_stats.get(m, {"q": 0, "c": 0})
        stat_rows.append({
            "模型": m.upper(), "题数": ms["q"], "正确数": ms["c"],
            "准确率": f"{ms['c']/ms['q']*100:.1f}%" if ms["q"] else "N/A",
        })
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=False)
    st.divider()

    st.subheader("🕸️ 五维能力雷达图（各模型跨所有场景平均）")
    radar_fig = go.Figure()
    for model in models:
        acc_map = agg_cap_acc(results, model)
        labels = [CAP_LABELS.get(c, c) for c in CAPABILITIES]
        values = [acc_map.get(c, 0) * 100 for c in CAPABILITIES]
        radar_fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]],
            fill="toself", name=model.upper(),
            line_color=MODEL_COLORS.get(model, "#888"), opacity=0.65,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=True, height=420, margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📍 各场景准确率对比")
        rows = []
        for (model, scene), data in results.items():
            qs = data.get("responses", [])
            c  = sum(1 for r in qs if r.get("is_correct", False))
            rows.append({"模型": model.upper(), "场景": scene[:10],
                         "准确率": c / len(qs) * 100 if qs else 0})
        fig = px.bar(pd.DataFrame(rows).sort_values(["场景","模型"]),
                     x="场景", y="准确率", color="模型", barmode="group",
                     color_discrete_map={m.upper(): c for m,c in MODEL_COLORS.items()},
                     height=360)
        fig.update_layout(yaxis=dict(range=[0,100], ticksuffix="%"),
                          margin=dict(l=40,r=20,t=20,b=60), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📐 各能力维度准确率")
        cap_rows = []
        for model in models:
            acc_map = agg_cap_acc(results, model)
            for cap in CAPABILITIES:
                cap_rows.append({"模型": model.upper(),
                                  "能力": CAP_LABELS.get(cap, cap),
                                  "准确率": acc_map.get(cap, 0) * 100})
        fig2 = px.bar(pd.DataFrame(cap_rows), x="能力", y="准确率", color="模型",
                      barmode="group",
                      color_discrete_map={m.upper(): c for m,c in MODEL_COLORS.items()},
                      height=360)
        fig2.update_layout(yaxis=dict(range=[0,100], ticksuffix="%"),
                           margin=dict(l=40,r=20,t=20,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📈 难度梯度准确率")
    diff_rows = []
    for model in models:
        diff_acc = defaultdict(list)
        for (m, s), data in results.items():
            if m != model:
                continue
            for r in data.get("responses", []):
                diff_acc[r.get("difficulty", "unknown")].append(
                    1 if r.get("is_correct", False) else 0)
        for d, vals in diff_acc.items():
            diff_rows.append({"模型": model.upper(), "难度": DIFF_LABELS.get(d, d),
                               "准确率": sum(vals)/len(vals)*100})
    if diff_rows:
        fig3 = px.bar(pd.DataFrame(diff_rows), x="难度", y="准确率", color="模型",
                      barmode="group",
                      color_discrete_map={m.upper(): c for m,c in MODEL_COLORS.items()},
                      category_orders={"难度": ["简单","中等","困难"]}, height=300)
        fig3.update_layout(yaxis=dict(range=[0,100], ticksuffix="%"),
                           margin=dict(l=40,r=20,t=20,b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("⏱️ 平均响应时间对比（每题）")
    time_rows = []
    for (model, scene), data in results.items():
        rts = [r.get("response_time_ms", 0) for r in data.get("responses", [])
               if r.get("response_time_ms", 0)]
        avg = sum(rts)/len(rts) if rts else 0
        time_rows.append({"模型": model.upper(), "场景": scene[:10],
                           "平均响应时间(s)": avg/1000})
    if time_rows:
        fig4 = px.bar(pd.DataFrame(time_rows).sort_values("场景"),
                      x="场景", y="平均响应时间(s)", color="模型", barmode="group",
                      color_discrete_map={m.upper(): c for m,c in MODEL_COLORS.items()},
                      height=300)
        fig4.update_layout(margin=dict(l=40,r=20,t=20,b=60), xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2：完整 Pipeline 演示
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.subheader("🎬 完整 Pipeline：出题 → 答题 → 诊断")

    if not exams:
        st.warning("未找到考卷文件")
    else:
        col_s, col_m, col_q = st.columns([1, 1, 2])
        with col_s:
            demo_scene = st.selectbox("场景", sorted(exams.keys()), key="ps")
        with col_m:
            model_opts = [m.upper() for m in models]
            demo_model_upper = st.selectbox("待测模型", model_opts, key="pm")
            demo_model = demo_model_upper.lower()
        exam_data = exams.get(demo_scene, {})
        questions = exam_data.get("questions", [])
        q_labels  = [f"{q['question_id']} [{CAP_LABELS.get(q['capability'],q['capability'])}]"
                     for q in questions]
        with col_q:
            q_idx = st.selectbox("题目", range(len(q_labels)),
                                 format_func=lambda i: q_labels[i], key="pq")

        if not questions:
            st.warning("该场景无题目数据")
        else:
            q         = questions[q_idx]
            evidence  = exam_data.get("multimodal_evidence", {})
            rgb_map   = evidence.get("rgb_frames", {})
            traj_list = evidence.get("trajectory", [])
            meta      = load_metadata(demo_scene)
            kf_list   = meta.get("keyframes", [])
            kf_map    = {kf["frame_id"]: kf for kf in kf_list}

            st.divider()

            # Phase 1：Claude 出题
            with st.expander("📝 Phase 1：Claude Sonnet 4.6 出题（Examiner Agent）", expanded=True):
                st.markdown(
                    f"**题目ID** `{q['question_id']}`　"
                    f"**能力** `{CAP_LABELS.get(q['capability'],q['capability'])}`　"
                    f"**难度** `{DIFF_LABELS.get(q['difficulty'],q['difficulty'])}`"
                )
                st.markdown("**题目文本**")
                st.info(q["question_text"])
                st.markdown("**标准答案**")
                st.success(q["ground_truth_answer"])

                trace = q.get("reasoning_trace", "").strip()
                if trace:
                    st.markdown("**Claude 出题推理过程（reasoning_trace）**")
                    st.code(trace, language=None)

                eids = q.get("evidence_frame_ids", [])
                st.markdown("---")
                st.markdown(
                    f"**Claude 接收的关键帧及精确位姿数据**（共 {len(rgb_map)} 帧，"
                    f"本题引用 {len(eids)} 帧）"
                )
                if traj_list:
                    t0, t_n = traj_list[0], traj_list[-1]
                    dx = t_n["position"][0] - t0["position"][0]
                    dz = t_n["position"][2] - t0["position"][2]
                    net = math.sqrt(dx*dx + dz*dz)
                    st.markdown(
                        f"| 起始帧 | 末帧 | 总帧数 | 净位移 |\n"
                        f"|--------|------|--------|--------|\n"
                        f"| frame {t0['frame_id']} | frame {t_n['frame_id']} "
                        f"| {len(traj_list)} | {net:.2f} m |"
                    )

                show_eids = eids[:6] if eids else sorted(
                    rgb_map.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)[:6]
                img_cols = st.columns(min(len(show_eids), 6))
                for ci, fid in enumerate(show_eids):
                    path = rgb_map.get(str(fid))
                    b64  = img_to_b64(path) if path else None
                    kf   = kf_map.get(int(fid) if str(fid).isdigit() else fid, {})
                    cap_txt = f"frame {fid}"
                    if kf:
                        pos = kf.get("position", [])
                        yaw = kf.get("yaw", "")
                        if pos:
                            cap_txt += f"\npos({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                        if yaw != "":
                            cap_txt += f"\nyaw={yaw:.1f}°"
                    col = img_cols[ci % 6]
                    if b64:
                        col.image(f"data:image/jpeg;base64,{b64}",
                                  caption=cap_txt, use_container_width=True)
                    else:
                        col.caption(cap_txt + "\n*(无图)*")

            # Phase 2：待测模型接收的图像
            with st.expander(
                f"🖼️ Phase 2：{demo_model_upper} 接收的图像序列（关键帧，无位姿数据）",
                expanded=True
            ):
                frame_keys = sorted(rgb_map.keys(),
                                    key=lambda k: int(k) if str(k).isdigit() else 0)
                st.caption(
                    f"**非对称架构**：Claude 使用关键帧 + 精确位姿元数据出题；"
                    f"{demo_model_upper} 仅收到 {len(frame_keys)} 帧图像，**无任何坐标/角度数值**。"
                )
                show_keys = (frame_keys if len(frame_keys) <= 12 else
                             [frame_keys[int(i*(len(frame_keys)-1)/11)] for i in range(12)])
                img_cols2 = st.columns(min(len(show_keys), 6))
                for i, k in enumerate(show_keys):
                    path = rgb_map.get(str(k))
                    b64  = img_to_b64(path) if path else None
                    col  = img_cols2[i % 6]
                    if b64:
                        col.image(f"data:image/jpeg;base64,{b64}",
                                  caption=f"frame {k}", use_container_width=True)
                    else:
                        col.caption(f"frame {k} *(无图)*")

            # Phase 3：模型作答
            with st.expander(f"💬 Phase 3：{demo_model_upper} 答题详情", expanded=True):
                resp = get_resp(results, demo_model, demo_scene, q["question_id"])
                if resp is None:
                    st.warning(
                        f"未找到 {demo_model_upper} 在场景 {demo_scene} 的作答记录。"
                        f"（已加载模型：{models}，已加载场景：{scenes[:3]}...）"
                    )
                else:
                    is_c = resp.get("is_correct", False)
                    st.markdown(f"**判定结果**：{'✅ 正确' if is_c else '❌ 错误'}")
                    col_gt, col_ans = st.columns(2)
                    with col_gt:
                        st.markdown("**标准答案**")
                        gt = resp.get("ground_truth_answer", q.get("ground_truth_answer","—"))
                        st.success(gt)
                    with col_ans:
                        st.markdown(f"**{demo_model_upper} 的回答**")
                        ans = resp.get("model_response", "")
                        box = st.success if is_c else st.error
                        box(ans[:600] + ("…" if len(ans) > 600 else ""))
                    rt = resp.get("response_time_ms", 0)
                    if rt:
                        st.caption(f"响应时间：{rt/1000:.1f}s")

            # Phase 4：Claude 诊断
            with st.expander(
                "🔍 Phase 4：Claude 三步排除法诊断（Reflector Agent）",
                expanded=True
            ):
                diag_list = diagnoses.get((demo_model, demo_scene), [])
                item = next(
                    (d for d in diag_list if d.get("question_id") == q["question_id"]), None)
                if item is None:
                    r2 = get_resp(results, demo_model, demo_scene, q["question_id"])
                    if r2 and r2.get("is_correct", False):
                        st.success("✅ 该题答对，无需诊断。")
                    else:
                        st.info("该题诊断数据尚未生成，或诊断正在运行中。")
                else:
                    diag   = item.get("diagnosis", {})
                    et_raw = item.get("error_type", "unknown")
                    st.markdown(
                        f"**错误类型**：{ERROR_LABELS.get(et_raw, et_raw)}　"
                        f"**置信度**：{item.get('confidence', 0):.0%}"
                    )
                    d1, d2, d3 = st.columns(3)
                    s1 = diag.get("step1_physical_check", {})
                    with d1:
                        st.markdown("**Step 1 物理-语义检查**")
                        hall = s1.get("has_hallucination", False)
                        st.markdown(f"幻觉：{'⚠️ 有' if hall else '✅ 无'}")
                        st.caption((s1.get("physical_violation") or "无违规")[:150])
                        st.markdown(f"结论：`{s1.get('conclusion','—')}`")
                    s2 = diag.get("step2_spatial_reconstruction", {})
                    with d2:
                        st.markdown("**Step 2 空间位置重塑**")
                        if s2:
                            st.markdown(f"cum_yaw_delta：`{s2.get('cum_yaw_delta','—')}°`")
                            calc = s2.get("correct_answer_calc", "—")
                            st.code(calc[:400] + ("…" if len(calc) > 400 else ""), language=None)
                            dev = s2.get("model_answer_deviation","—")
                            st.caption(f"偏差：{dev[:120]}")
                    s3 = diag.get("step3_root_cause", {})
                    with d3:
                        st.markdown("**Step 3 根因分类**")
                        et_label = ERROR_LABELS.get(s3.get("error_type", et_raw), et_raw)
                        conf = s3.get("confidence", item.get("confidence", 0))
                        st.error(f"**{et_label}**")
                        st.metric("置信度", f"{conf:.0%}")
                        expl = s3.get("explanation", item.get("root_cause_explanation",""))
                        st.caption(expl[:250])
                    if item.get("root_cause_explanation"):
                        st.markdown("**完整根因说明**")
                        st.info(item["root_cause_explanation"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3：错题诊断详情
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.subheader("🔬 错题诊断详情浏览器")

    col_dm, col_ds = st.columns(2)
    with col_dm:
        sel_model = st.selectbox("模型", [m.upper() for m in models], key="dm")
    with col_ds:
        sel_scene = st.selectbox("场景", scenes, key="ds")

    dm = sel_model.lower()
    diag_list   = diagnoses.get((dm, sel_scene), [])
    result_data = results.get((dm, sel_scene), {})
    exam_data   = exams.get(sel_scene, {})
    q_map       = {q["question_id"]: q for q in exam_data.get("questions", [])}
    rgb_frames  = exam_data.get("multimodal_evidence", {}).get("rgb_frames", {})

    total_q  = len(result_data.get("responses", []))
    correct  = sum(1 for r in result_data.get("responses", []) if r.get("is_correct", False))
    wrong    = total_q - correct

    st.caption(
        f"**{sel_model} / {sel_scene}**：{total_q} 题 | "
        f"正确 {correct} | 错误 {wrong} | 诊断记录 {len(diag_list)} 条"
    )

    if not diag_list:
        if wrong == 0 and total_q > 0:
            st.success("🎉 该场景全部答对，无需诊断！")
        else:
            st.warning(f"该场景有 {wrong} 道错题，诊断数据尚未生成。")
    else:
        for item in diag_list:
            qid  = item.get("question_id", "?")
            cap  = item.get("capability", "")
            et   = ERROR_LABELS.get(item.get("error_type",""), item.get("error_type","未知"))
            conf = item.get("confidence", 0)

            with st.expander(
                f"❌ {qid} | {CAP_LABELS.get(cap, cap)} | {et}（{conf:.0%}）"
            ):
                q_info = q_map.get(qid, {})
                resp   = next((r for r in result_data.get("responses", [])
                               if r.get("question_id") == qid), None)

                qa1, qa2, qa3 = st.columns(3)
                qa1.markdown("**题目**")
                qa1.info(q_info.get("question_text", item.get("question_text","—")))
                qa2.markdown("**标准答案**")
                qa2.success(item.get("ground_truth",
                                     q_info.get("ground_truth_answer","—")))
                qa3.markdown("**模型回答**")
                ans = item.get("model_answer",
                               resp.get("model_response","") if resp else "")
                qa3.error(ans[:300] + ("…" if len(ans)>300 else ""))

                diag = item.get("diagnosis", {})
                if diag and "error" not in diag and "raw" not in diag:
                    st.markdown("---")
                    st.markdown("**三步排除法诊断**")
                    s1c, s2c, s3c = st.columns(3)
                    s1 = diag.get("step1_physical_check", {})
                    with s1c:
                        st.markdown("**Step 1 物理检查**")
                        st.markdown(f"{'⚠️ 有幻觉' if s1.get('has_hallucination') else '✅ 无幻觉'}")
                        st.caption(s1.get("conclusion","—"))
                    s2 = diag.get("step2_spatial_reconstruction", {})
                    with s2c:
                        st.markdown("**Step 2 空间重塑**")
                        st.code(s2.get("correct_answer_calc","—")[:300], language=None)
                    s3 = diag.get("step3_root_cause", {})
                    with s3c:
                        st.markdown("**Step 3 根因**")
                        st.error(ERROR_LABELS.get(s3.get("error_type",""),"—"))
                        st.caption(s3.get("explanation",
                                          item.get("root_cause_explanation",""))[:200])
                elif "raw" in diag:
                    st.warning("诊断输出被截断（max_tokens 不足），部分内容：")
                    st.code(str(diag["raw"])[:400], language=None)

                eids = q_info.get("evidence_frame_ids", [])
                show = ([eids[0], eids[len(eids)//2], eids[-1]]
                        if len(eids) >= 3 else eids)
                if show:
                    st.markdown("**Claude 出题证据帧（首/中/尾）**")
                    fc = st.columns(len(show))
                    for ci, fid in enumerate(show):
                        path = rgb_frames.get(str(fid))
                        b64  = img_to_b64(path) if path else None
                        if b64:
                            fc[ci].image(f"data:image/jpeg;base64,{b64}",
                                         caption=f"frame {fid}",
                                         use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4：汇总表格
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("📋 模型综合对比")

    sum_rows = []
    for model in models:
        tq, tc = 0, 0
        cap_c = defaultdict(int)
        cap_t = defaultdict(int)
        for (m, s), data in results.items():
            if m != model:
                continue
            for r in data.get("responses", []):
                tq += 1
                cap = r.get("capability", "unknown")
                cap_t[cap] += 1
                if r.get("is_correct", False):
                    tc += 1
                    cap_c[cap] += 1
        row = {
            "模型": model.upper(),
            "场景数": len([s for (m,s) in results if m == model]),
            "总题数": tq, "正确数": tc,
            "总体准确率": f"{tc/tq*100:.1f}%" if tq else "N/A",
        }
        for cap in CAPABILITIES:
            row[CAP_LABELS[cap]] = (f"{cap_c[cap]/cap_t[cap]*100:.0f}%"
                                    if cap_t.get(cap) else "0%")
        sum_rows.append(row)
    st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🗂️ 全部 50 题明细")

    all_rows = []
    for (model, scene), data in sorted(results.items()):
        eq_map = {q["question_id"]: q
                  for q in exams.get(scene, {}).get("questions", [])}
        for r in data.get("responses", []):
            qid  = r.get("question_id","?")
            q    = eq_map.get(qid, {})
            cap  = r.get("capability") or q.get("capability","unknown")
            diff = r.get("difficulty") or q.get("difficulty","unknown")
            is_c = r.get("is_correct", False)
            ans  = r.get("model_response","")
            gt   = r.get("ground_truth_answer") or q.get("ground_truth_answer","")
            dl   = diagnoses.get((model, scene), [])
            di   = next((d for d in dl if d.get("question_id") == qid), None)
            et   = (ERROR_LABELS.get(di.get("error_type",""),
                                     di.get("error_type","—"))
                    if di else ("—" if is_c else "待诊断"))
            all_rows.append({
                "模型": model.upper(), "场景": scene[:12], "题目ID": qid,
                "能力": CAP_LABELS.get(cap, cap),
                "难度": DIFF_LABELS.get(diff, diff),
                "正确": "✅" if is_c else "❌",
                "模型回答": ans[:60]+("…" if len(ans)>60 else ""),
                "标准答案": gt[:60]+("…" if len(gt)>60 else ""),
                "错误类型": et,
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        fc1, fc2 = st.columns(2)
        with fc1:
            fm = st.multiselect("筛选模型", [m.upper() for m in models],
                                default=[m.upper() for m in models])
        with fc2:
            fr = st.radio("筛选结果", ["全部","仅正确","仅错误"], horizontal=True)
        df = df[df["模型"].isin(fm)]
        if fr == "仅正确":
            df = df[df["正确"] == "✅"]
        elif fr == "仅错误":
            df = df[df["正确"] == "❌"]
        st.caption(f"显示 {len(df)} / {len(all_rows)} 条")
        st.dataframe(df, use_container_width=True, hide_index=True)
