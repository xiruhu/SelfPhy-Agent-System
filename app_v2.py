"""
app_v2.py
---------
SelfPhy-Agent-System V2 Streamlit 看板

功能：
1. 总览页：多模型准确率对比 + 能力热力图
2. 题目详情：逐题查看问题、Kimi 回答、标准答案、图像帧
3. 诊断报告：四步排除法输出，错误类型分布
4. 原始数据：JSON 浏览器

运行：
  streamlit run app_v2.py -- \
    --results kimi:outputs/test_result_kimi_v2_final.json \
    --exam outputs/test_exam_v2_precise.json
"""

import json
import sys
import base64
from pathlib import Path
from collections import Counter

import streamlit as st

# ──────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────

st.set_page_config(
    page_title="SelfPhy-Agent-System V2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# 数据加载（带缓存）
# ──────────────────────────────────────────

@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ──────────────────────────────────────────
# 侧边栏：文件选择
# ──────────────────────────────────────────

st.sidebar.title("SelfPhy V2 看板")
st.sidebar.markdown("---")

# 自动扫描 outputs 目录
results_dir = Path("outputs")
result_files = sorted(results_dir.glob("test_result_*.json"))
exam_files   = sorted(results_dir.glob("test_exam_v2*.json"))
diag_files   = sorted(Path("outputs/reports").glob("diagnosis_*.json"))

result_options = {f.name: str(f) for f in result_files}
exam_options   = {f.name: str(f) for f in exam_files}
diag_options   = {"（无）": None} | {f.name: str(f) for f in diag_files}

if not result_options:
    st.warning("未找到评测结果文件，请先运行 evaluate_runner_v2.py")
    st.stop()

selected_result = st.sidebar.selectbox("评测结果", list(result_options.keys()))
selected_exam   = st.sidebar.selectbox("考卷文件", list(exam_options.keys()) or ["未找到"])
selected_diag   = st.sidebar.selectbox("诊断报告（可选）", list(diag_options.keys()))

result_path = result_options[selected_result]
exam_path   = exam_options.get(selected_exam)
diag_path   = diag_options.get(selected_diag)

result = load_json(result_path)
exam   = load_json(exam_path) if exam_path else None
diag   = load_json(diag_path) if diag_path else None

model_name = result["model_name"]
metrics    = result["metrics"]
responses  = result["responses"]
questions  = {q["question_id"]: q for q in exam["questions"]} if exam else {}

st.sidebar.markdown("---")
st.sidebar.markdown(f"**模型**: `{model_name}`")
st.sidebar.markdown(f"**题目数**: {metrics['total_questions']}")
st.sidebar.markdown(f"**准确率**: {metrics['accuracy']*100:.1f}%")

# ──────────────────────────────────────────
# 主页面
# ──────────────────────────────────────────

tabs = st.tabs(["📊 总览", "📝 题目详情", "🔍 诊断报告", "🗂️ 原始数据"])

# ──────── Tab 1：总览 ────────
with tabs[0]:
    st.header("评测总览")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总准确率", f"{metrics['accuracy']*100:.1f}%")
    col2.metric("正确题数", f"{metrics['correct_count']} / {metrics['total_questions']}")
    col3.metric("平均响应时间", f"{metrics['avg_response_time_ms']/1000:.1f}s")
    col4.metric("模型", model_name)

    st.markdown("---")

    # 能力维度准确率
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("能力维度准确率")
        cap_acc = metrics.get("capability_accuracy", {})
        if cap_acc:
            import pandas as pd
            df_cap = pd.DataFrame([
                {"能力维度": k.replace("_", " ").title(), "准确率": f"{v*100:.1f}%",
                 "得分": v}
                for k, v in cap_acc.items()
            ])
            st.dataframe(
                df_cap.style.background_gradient(subset=["得分"], cmap="RdYlGn", vmin=0, vmax=1),
                hide_index=True, use_container_width=True
            )
        else:
            st.info("无能力维度数据")

    with col_right:
        st.subheader("难度维度准确率")
        diff_acc = metrics.get("difficulty_accuracy", {})
        if diff_acc:
            df_diff = pd.DataFrame([
                {"难度": k.title(), "准确率": f"{v*100:.1f}%", "得分": v}
                for k, v in diff_acc.items()
            ])
            st.dataframe(
                df_diff.style.background_gradient(subset=["得分"], cmap="RdYlGn", vmin=0, vmax=1),
                hide_index=True, use_container_width=True
            )
        else:
            st.info("无难度维度数据")

    # 是否有可视化图表
    st.markdown("---")
    viz_dir = Path("outputs/visualizations")
    summary_img = viz_dir / "summary.png"
    if summary_img.exists():
        st.subheader("综合总结图")
        st.image(str(summary_img), use_container_width=True)
    else:
        st.info("运行 `python core/analytics_viz_v2.py` 生成可视化图表")

# ──────── Tab 2：题目详情 ────────
with tabs[1]:
    st.header("题目详情")

    for i, resp in enumerate(responses):
        qid      = resp["question_id"]
        q        = questions.get(qid, {})
        correct  = resp["is_correct"]
        score    = resp.get("score", float(correct))
        status   = "✅" if correct else "❌"

        with st.expander(f"{status} [{i+1}] {qid}  |  得分: {score:.2f}  |  {q.get('capability','')} · {q.get('difficulty','')}"):
            col_q, col_a = st.columns([3, 2])

            with col_q:
                st.markdown("**问题**")
                st.write(q.get("question_text", "（考卷未加载）"))
                st.markdown("**模型回答**")
                st.write(resp["model_response"] or "（空）")
                st.markdown("**标准答案**")
                st.success(resp["ground_truth"])
                st.markdown(f"**判定依据**: {resp.get('judge_detail','N/A')}")
                if q.get("reasoning_trace"):
                    with st.expander("查看 Claude 推理链"):
                        st.write(q["reasoning_trace"])

            with col_a:
                # 显示证据帧图像
                if exam:
                    rgb_frames = exam["multimodal_evidence"]["rgb_frames"]
                    eids = q.get("evidence_frame_ids", [])
                    # 展示首/中/尾帧
                    show_fids = []
                    if eids:
                        show_fids = [eids[0]]
                        if len(eids) > 2:
                            show_fids.append(eids[len(eids)//2])
                        show_fids.append(eids[-1])

                    for fid in show_fids:
                        path = rgb_frames.get(str(fid))
                        if path and Path(path).exists():
                            st.image(path, caption=f"frame {fid}", use_container_width=True)

            st.markdown(f"**响应时间**: {resp['response_time_ms']/1000:.1f}s")
            if resp.get("error"):
                st.error(f"API 错误: {resp['error']}")

# ──────── Tab 3：诊断报告 ────────
with tabs[2]:
    st.header("四步排除法诊断报告")

    if not diag:
        st.info("未加载诊断报告。先运行：\n```bash\npython core/claude_reflector_v2.py outputs/test_result_kimi_v2_final.json outputs/test_exam_v2_precise.json data/test_v2_run/episode_000000/metadata.json\n```")
    else:
        # 错误类型分布
        error_counts = Counter(d["error_type"] for d in diag)
        st.subheader(f"错误类型分布（共 {len(diag)} 道错题）")

        cols = st.columns(len(error_counts) if error_counts else 1)
        for i, (etype, cnt) in enumerate(error_counts.most_common()):
            cols[i].metric(etype.replace("_", " ").title(), cnt)

        st.markdown("---")

        for d in diag:
            with st.expander(f"🔍 {d['question_id']}  |  {d['error_type']}  |  置信度 {d['confidence']:.0%}"):
                st.markdown(f"**根因解释**: {d['root_cause_explanation']}")

                diag_steps = d.get("diagnosis", {})
                if isinstance(diag_steps, dict) and "step1_physical_check" in diag_steps:
                    step1 = diag_steps["step1_physical_check"]
                    step2 = diag_steps.get("step2_spatial_reconstruction", {})
                    step3 = diag_steps.get("step3_fov_check", {})
                    step4 = diag_steps.get("step4_root_cause", {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Step 1 - 物理-语义对齐**")
                        st.json(step1)
                        st.markdown("**Step 2 - 空间位置重塑**")
                        st.json(step2)
                    with col2:
                        st.markdown("**Step 3 - 视场边界校验**")
                        st.json(step3)
                        st.markdown("**Step 4 - 根因归纳**")
                        st.json(step4)
                else:
                    st.json(diag_steps)

# ──────── Tab 4：原始数据 ────────
with tabs[3]:
    st.header("原始数据浏览")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("评测结果 JSON")
        st.json(result)
    with col2:
        st.subheader("考卷 JSON")
        if exam:
            # 不展示 base64 图像内容
            exam_display = json.loads(json.dumps(exam))
            for fid in list(exam_display["multimodal_evidence"].get("rgb_frames", {}).keys()):
                exam_display["multimodal_evidence"]["rgb_frames"][fid] = "<path>"
            st.json(exam_display)
        else:
            st.info("未加载考卷")
