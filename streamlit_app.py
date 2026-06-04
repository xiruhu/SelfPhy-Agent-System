import streamlit as st
import json
import pandas as pd
from PIL import Image
import os

# =========================
# 页面标题
# =========================

st.title("SelfPhy-Agent Dashboard")

# =========================
# 读取数据
# =========================

with open("outputs/answers/answers.json", "r") as f:
    answers = json.load(f)

with open("outputs/reports/failure_report.json", "r") as f:
    reports = json.load(f)

# =========================
# 展示题目
# =========================

st.header("Exam Results")

for item in answers:

    st.subheader(item["question"])

    st.write("Ground Truth:", item["ground_truth"])

    st.write("Model Answer:", item["model_answer"])

    st.write("Correct:", item["correct"])

    frame_path = os.path.join(
        "outputs/frames",
        item["frame_name"]
    )

    if os.path.exists(frame_path):

        image = Image.open(frame_path)

        st.image(image, width=400)

# =========================
# 展示错误分析
# =========================

st.header("Failure Analysis")

df = pd.DataFrame(reports)

st.dataframe(df)

# =========================
# 展示饼图
# =========================

st.header("Failure Distribution")

chart_path = "outputs/reports/failure_pie_chart.png"

if os.path.exists(chart_path):

    st.image(chart_path)