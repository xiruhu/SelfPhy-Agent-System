import json
from collections import Counter
import matplotlib.pyplot as plt

# =========================
# 读取报告
# =========================

REPORT_PATH = "outputs/reports/failure_report.json"

with open(REPORT_PATH, "r") as f:
    reports = json.load(f)

# =========================
# 统计 failure type
# =========================

failure_types = [r["failure_type"] for r in reports]

counter = Counter(failure_types)

# =========================
# 输出统计
# =========================

print("\n===== Failure Statistics =====")

for k, v in counter.items():

    print(f"{k}: {v}")

# =========================
# 绘制饼图
# =========================

labels = list(counter.keys())

sizes = list(counter.values())

plt.figure(figsize=(8,8))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Failure Type Distribution")

plt.savefig("outputs/reports/failure_pie_chart.png")

print("\n饼图已保存")