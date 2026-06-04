"""
Streamlit 交互看板
提供 Web 界面展示评测结果
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go


# 页面配置
st.set_page_config(
    page_title="SelfPhy-Agent-System",
    page_icon="🤖",
    layout="wide"
)


def load_data():
    """加载所有数据"""
    data = {
        "trajectories": [],
        "questions": [],
        "responses": [],
        "analyses": []
    }

    # 加载轨迹
    traj_dir = Path("outputs/trajectories")
    if traj_dir.exists():
        for traj_file in traj_dir.glob("*.json"):
            with open(traj_file, 'r', encoding='utf-8') as f:
                data["trajectories"].append(json.load(f))

    # 加载考题
    exam_dir = Path("outputs/exams")
    if exam_dir.exists():
        for exam_file in exam_dir.glob("*.json"):
            with open(exam_file, 'r', encoding='utf-8') as f:
                data["questions"].extend(json.load(f))

    # 加载响应
    answer_dir = Path("outputs/answers")
    if answer_dir.exists():
        for answer_file in answer_dir.glob("*.json"):
            with open(answer_file, 'r', encoding='utf-8') as f:
                data["responses"].extend(json.load(f))

    # 加载分析
    report_dir = Path("outputs/reports")
    if report_dir.exists():
        for report_file in report_dir.glob("*.json"):
            with open(report_file, 'r', encoding='utf-8') as f:
                data["analyses"].extend(json.load(f))

    return data


def render_overview(data):
    """渲染概览页面"""
    st.header("📊 系统概览")

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("轨迹数量", len(data["trajectories"]))

    with col2:
        st.metric("考题数量", len(data["questions"]))

    with col3:
        st.metric("评测响应", len(data["responses"]))

    with col4:
        st.metric("错误分析", len(data["analyses"]))

    # 准确率计算
    if data["questions"] and data["responses"]:
        total_questions = len(data["questions"])
        total_errors = len(data["analyses"])
        accuracy = (total_questions - total_errors) / total_questions * 100

        st.subheader("总体准确率")
        st.progress(accuracy / 100)
        st.write(f"**{accuracy:.1f}%** ({total_questions - total_errors}/{total_questions} 正确)")

    # 题型分布
    if data["questions"]:
        st.subheader("题型分布")

        question_types = [q["question_type"] for q in data["questions"]]
        type_counts = Counter(question_types)

        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Question Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_trajectories(data):
    """渲染轨迹页面"""
    st.header("🗺️ 轨迹数据")

    if not data["trajectories"]:
        st.warning("暂无轨迹数据")
        return

    # 选择轨迹
    traj_ids = [t["segment_id"] for t in data["trajectories"]]
    selected_traj_id = st.selectbox("选择轨迹", traj_ids)

    # 找到选中的轨迹
    selected_traj = next(t for t in data["trajectories"] if t["segment_id"] == selected_traj_id)

    # 显示轨迹信息
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("轨迹信息")
        st.write(f"**ID**: {selected_traj['segment_id']}")
        st.write(f"**关键帧数量**: {len(selected_traj['keyframes'])}")
        st.write(f"**时长**: {selected_traj['end_time'] - selected_traj['start_time']:.1f} 秒")

    with col2:
        st.subheader("空间描述")
        st.write(selected_traj["spatial_narrative"])

    # 显示轨迹可视化
    viz_path = Path(f"outputs/visualizations/{selected_traj_id}_trajectory.png")
    if viz_path.exists():
        st.subheader("轨迹可视化")
        st.image(str(viz_path))

    # 关键帧列表
    st.subheader("关键帧")

    for i, kf in enumerate(selected_traj["keyframes"][:5]):  # 只显示前5个
        with st.expander(f"Frame {kf['frame_id']}"):
            col1, col2 = st.columns([1, 2])

            with col1:
                img_path = Path(kf["image_path"])
                if img_path.exists():
                    st.image(str(img_path), width=300)

            with col2:
                st.write(f"**位置**: {kf['pose']['position']}")
                st.write(f"**欧拉角**: {kf['pose']['euler_angles']}")
                st.write(f"**动作**: {kf['pose'].get('action_label', 'N/A')}")


def render_questions(data):
    """渲染考题页面"""
    st.header("📝 考题")

    if not data["questions"]:
        st.warning("暂无考题数据")
        return

    # 筛选器
    col1, col2, col3 = st.columns(3)

    with col1:
        question_types = list(set([q["question_type"] for q in data["questions"]]))
        selected_type = st.selectbox("题型", ["全部"] + question_types)

    with col2:
        difficulties = list(set([q["difficulty"] for q in data["questions"]]))
        selected_difficulty = st.selectbox("难度", ["全部"] + difficulties)

    with col3:
        spatial_levels = list(set([q["spatial_level"] for q in data["questions"]]))
        selected_level = st.selectbox("空间层级", ["全部"] + spatial_levels)

    # 过滤问题
    filtered_questions = data["questions"]

    if selected_type != "全部":
        filtered_questions = [q for q in filtered_questions if q["question_type"] == selected_type]

    if selected_difficulty != "全部":
        filtered_questions = [q for q in filtered_questions if q["difficulty"] == selected_difficulty]

    if selected_level != "全部":
        filtered_questions = [q for q in filtered_questions if q["spatial_level"] == selected_level]

    st.write(f"显示 {len(filtered_questions)} / {len(data['questions'])} 道题")

    # 显示问题
    for i, question in enumerate(filtered_questions[:10]):  # 只显示前10个
        with st.expander(f"{question['question_id']} - {question['difficulty']} - {question['question_type']}"):
            st.write("**问题**:")
            st.write(question["prompt"])

            st.write("**标准答案**:")
            st.json(question["ground_truth"])

            # 显示上下文图片
            if question["context_frames"]:
                st.write("**上下文图片**:")
                cols = st.columns(len(question["context_frames"]))
                for idx, img_path in enumerate(question["context_frames"]):
                    if Path(img_path).exists():
                        with cols[idx]:
                            st.image(img_path, width=200)


def render_analyses(data):
    """渲染错误分析页面"""
    st.header("🔍 错误分析")

    if not data["analyses"]:
        st.info("所有答案正确，无错误分析")
        return

    # 错误类型分布
    st.subheader("错误类型分布")

    error_types = [a["error_type"] for a in data["analyses"]]
    type_counts = Counter(error_types)

    fig = px.bar(
        x=list(type_counts.keys()),
        y=list(type_counts.values()),
        labels={"x": "错误类型", "y": "数量"},
        title="Error Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 错误详情
    st.subheader("错误详情")

    for analysis in data["analyses"][:10]:  # 只显示前10个
        with st.expander(f"{analysis['question_id']} - {analysis['error_type']}"):
            st.write(f"**模型**: {analysis['model_name']}")
            st.write(f"**置信度**: {analysis['confidence']:.2f}")

            st.write("**根本原因**:")
            st.write(analysis["root_cause"])

            st.write("**因果追踪**:")
            for step in analysis["causal_trace"]:
                st.write(f"- **{step['step_name']}** (置信度: {step['confidence']:.2f})")
                st.write(f"  - 假设: {step['hypothesis']}")
                st.write(f"  - 结论: {step['conclusion']}")


def render_visualizations(data):
    """渲染可视化页面"""
    st.header("📈 可视化报告")

    viz_dir = Path("outputs/visualizations")

    if not viz_dir.exists():
        st.warning("暂无可视化报告")
        return

    # 显示所有可视化图表
    viz_files = list(viz_dir.glob("*.png"))

    if not viz_files:
        st.warning("暂无可视化图表")
        return

    for viz_file in viz_files:
        st.subheader(viz_file.stem.replace("_", " ").title())
        st.image(str(viz_file))


def main():
    """主函数"""
    st.title("🤖 SelfPhy-Agent-System")
    st.markdown("基于第一人称感知演化与自反思的空间推理自动化评测系统")

    # 加载数据
    with st.spinner("加载数据..."):
        data = load_data()

    # 侧边栏
    with st.sidebar:
        st.header("导航")
        page = st.radio(
            "选择页面",
            ["概览", "轨迹数据", "考题", "错误分析", "可视化报告"]
        )

        st.markdown("---")

        st.subheader("系统信息")
        st.write(f"轨迹: {len(data['trajectories'])}")
        st.write(f"考题: {len(data['questions'])}")
        st.write(f"响应: {len(data['responses'])}")
        st.write(f"分析: {len(data['analyses'])}")

    # 渲染选中的页面
    if page == "概览":
        render_overview(data)
    elif page == "轨迹数据":
        render_trajectories(data)
    elif page == "考题":
        render_questions(data)
    elif page == "错误分析":
        render_analyses(data)
    elif page == "可视化报告":
        render_visualizations(data)


if __name__ == "__main__":
    main()
