"""
analytics_viz.py
-------------------
可视化层 V2：读取 EvaluationResultV2 + 诊断报告，生成论文级图表

支持图表：
1. 5 维能力雷达图（多模型对比）
2. 错误类型分布饼图
3. 难度-准确率柱状图
4. 响应时间箱线图
5. 综合总结图（2×2 拼图）
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


# 5 个能力维度（固定顺序，保证雷达图可比）
CAPABILITIES = [
    "egocentric_memory",
    "spatial_transformation",
    "occlusion_reasoning",
    "trajectory_backtracking",
    "distance_estimation",
]
CAP_LABELS = [
    "Egocentric\nMemory",
    "Spatial\nTransform",
    "Occlusion\nReason",
    "Trajectory\nBacktrack",
    "Distance\nEstim.",
]

# 错误类型（来自 claude_reflector_v2）
ERROR_TYPES = [
    "direction_calc_error",
    "rotation_sense_error",
    "rotation_translation_confusion",
    "memory_decay",
    "object_hallucination",
    "fov_misunderstanding",
]
ERROR_LABELS = [
    "Direction Calc",
    "Rotation Sense",
    "Rot-Trans Confusion",
    "Memory Decay",
    "Hallucination",
    "FOV Misunderstand",
]

COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]


class AnalyticsVizV2:

    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────

    def load_result(self, result_path: str) -> Dict:
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    def load_diagnosis(self, diagnosis_path: str) -> List[Dict]:
        with open(diagnosis_path, encoding="utf-8") as f:
            return json.load(f)

    def _cap_accuracy(self, result: Dict) -> Dict[str, float]:
        """从 EvaluationResultV2 提取各能力维度准确率"""
        return result["metrics"].get("capability_accuracy", {})

    # ──────────────────────────────────────────
    # 图 1：能力雷达图
    # ──────────────────────────────────────────

    def radar_chart(
        self,
        results: Dict[str, Dict],   # {model_name: result_dict}
        output_filename: str = "radar_chart.png"
    ) -> str:
        n = len(CAPABILITIES)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(CAP_LABELS, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=8)
        ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

        for i, (model_name, result) in enumerate(results.items()):
            cap_acc = self._cap_accuracy(result)
            values = [cap_acc.get(c, 0.0) for c in CAPABILITIES]
            values += values[:1]
            color = COLORS[i % len(COLORS)]
            ax.plot(angles, values, "o-", linewidth=2, color=color, label=model_name)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title("Capability Accuracy Radar\n(V2 Embodied Spatial Evaluation)", pad=20, fontsize=13)

        out_path = str(self.out / output_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  雷达图: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    # 图 2：错误类型分布饼图
    # ──────────────────────────────────────────

    def error_pie(
        self,
        diagnoses: Dict[str, List[Dict]],   # {model_name: diagnosis_list}
        output_filename: str = "error_pie.png"
    ) -> str:
        n_models = len(diagnoses)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, diag_list) in zip(axes, diagnoses.items()):
            counts = Counter(d["error_type"] for d in diag_list)
            labels, sizes, colors = [], [], []
            for etype, elabel, color in zip(ERROR_TYPES, ERROR_LABELS, COLORS):
                cnt = counts.get(etype, 0)
                if cnt > 0:
                    labels.append(f"{elabel}\n({cnt})")
                    sizes.append(cnt)
                    colors.append(color)

            if sizes:
                ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                       startangle=90, textprops={"fontsize": 9})
            else:
                ax.text(0.5, 0.5, "No errors", ha="center", va="center")

            ax.set_title(f"{model_name}\nError Type Distribution", fontsize=12)

        out_path = str(self.out / output_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  错误类型饼图: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    # 图 3：难度-准确率柱状图
    # ──────────────────────────────────────────

    def difficulty_bar(
        self,
        results: Dict[str, Dict],
        output_filename: str = "difficulty_accuracy.png"
    ) -> str:
        difficulties = ["easy", "medium", "hard"]
        x = np.arange(len(difficulties))
        width = 0.8 / max(len(results), 1)

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (model_name, result) in enumerate(results.items()):
            diff_acc = result["metrics"].get("difficulty_accuracy", {})
            vals = [diff_acc.get(d, 0.0) * 100 for d in difficulties]
            offset = (i - len(results) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                          label=model_name, color=COLORS[i % len(COLORS)], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color="grey", linestyle="--", alpha=0.5, label="50% baseline")
        ax.legend(fontsize=10)
        ax.set_title("Accuracy by Difficulty Level", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        out_path = str(self.out / output_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  难度柱状图: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    # 图 4：综合总结图（2×2）
    # ──────────────────────────────────────────

    def summary_figure(
        self,
        results: Dict[str, Dict],
        diagnoses: Optional[Dict[str, List[Dict]]] = None,
        output_filename: str = "summary.png"
    ) -> str:
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("SelfPhy-Agent-System V2 — Evaluation Summary", fontsize=16, fontweight="bold")

        # 子图 1（左上）：总体准确率横向条形图
        ax1 = fig.add_subplot(2, 2, 1)
        model_names = list(results.keys())
        accuracies = [results[m]["metrics"]["accuracy"] * 100 for m in model_names]
        bars = ax1.barh(model_names, accuracies,
                        color=[COLORS[i % len(COLORS)] for i in range(len(model_names))],
                        alpha=0.85, height=0.5)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.1f}%", va="center", fontsize=11)
        ax1.set_xlim(0, 115)
        ax1.set_xlabel("Overall Accuracy (%)")
        ax1.set_title("Overall Accuracy")
        ax1.axvline(x=50, color="grey", linestyle="--", alpha=0.5)

        # 子图 2（右上）：能力维度热力图
        ax2 = fig.add_subplot(2, 2, 2)
        cap_matrix = []
        for m in model_names:
            cap_acc = results[m]["metrics"].get("capability_accuracy", {})
            row = [cap_acc.get(c, 0.0) * 100 for c in CAPABILITIES]
            cap_matrix.append(row)
        cap_matrix = np.array(cap_matrix) if cap_matrix else np.zeros((1, len(CAPABILITIES)))
        im = ax2.imshow(cap_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax2.set_xticks(range(len(CAPABILITIES)))
        ax2.set_xticklabels(CAP_LABELS, fontsize=8, rotation=20, ha="right")
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names, fontsize=10)
        for i in range(len(model_names)):
            for j in range(len(CAPABILITIES)):
                ax2.text(j, i, f"{cap_matrix[i, j]:.0f}%",
                         ha="center", va="center", fontsize=9,
                         color="black" if cap_matrix[i, j] > 40 else "white")
        fig.colorbar(im, ax=ax2, shrink=0.8)
        ax2.set_title("Capability Accuracy Heatmap (%)")

        # 子图 3（左下）：难度-准确率
        ax3 = fig.add_subplot(2, 2, 3)
        difficulties = ["easy", "medium", "hard"]
        x = np.arange(len(difficulties))
        width = 0.8 / max(len(results), 1)
        for i, (m, result) in enumerate(results.items()):
            diff_acc = result["metrics"].get("difficulty_accuracy", {})
            vals = [diff_acc.get(d, 0.0) * 100 for d in difficulties]
            offset = (i - len(results) / 2 + 0.5) * width
            ax3.bar(x + offset, vals, width * 0.9,
                    label=m, color=COLORS[i % len(COLORS)], alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(["Easy", "Medium", "Hard"])
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_ylim(0, 110)
        ax3.legend(fontsize=9)
        ax3.set_title("Accuracy by Difficulty")
        ax3.grid(axis="y", alpha=0.3)

        # 子图 4（右下）：错误类型分布（若有诊断报告）
        ax4 = fig.add_subplot(2, 2, 4)
        if diagnoses:
            all_errors = []
            for diag_list in diagnoses.values():
                all_errors.extend(d["error_type"] for d in diag_list)
            counts = Counter(all_errors)
            labels_used = [(ERROR_LABELS[i], counts.get(et, 0), COLORS[i])
                           for i, et in enumerate(ERROR_TYPES) if counts.get(et, 0) > 0]
            if labels_used:
                labels_txt, sizes, pie_colors = zip(*labels_used)
                ax4.pie(sizes, labels=labels_txt, colors=pie_colors,
                        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
                ax4.set_title("Error Type Distribution (All Models)")
            else:
                ax4.text(0.5, 0.5, "No errors diagnosed", ha="center", va="center")
                ax4.set_title("Error Type Distribution")
        else:
            ax4.text(0.5, 0.5, "Run claude_reflector_v2\nfor diagnosis", ha="center", va="center", fontsize=10)
            ax4.set_title("Error Type Distribution")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = str(self.out / output_filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  综合总结图: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    # 一键生成全部图表
    # ──────────────────────────────────────────

    def generate_all(
        self,
        result_paths: Dict[str, str],       # {model_name: result_json_path}
        diagnosis_paths: Dict[str, str] = None  # {model_name: diagnosis_json_path}
    ) -> Dict[str, str]:
        """
        一键生成所有图表。

        Args:
            result_paths:    各模型的评测结果路径
            diagnosis_paths: 各模型的诊断报告路径（可选）

        Returns:
            生成的图表路径字典
        """
        print("\n[Analytics V2] 开始生成可视化图表...")

        results   = {m: self.load_result(p) for m, p in result_paths.items()}
        diagnoses = None
        if diagnosis_paths:
            diagnoses = {m: self.load_diagnosis(p) for m, p in diagnosis_paths.items()
                         if Path(p).exists()}

        generated = {}
        generated["radar"]      = self.radar_chart(results)
        generated["difficulty"] = self.difficulty_bar(results)
        generated["summary"]    = self.summary_figure(results, diagnoses)
        if diagnoses:
            generated["error_pie"] = self.error_pie(diagnoses)

        print(f"\n✅ 共生成 {len(generated)} 张图表，保存在: {self.out}")
        return generated


def main():
    import argparse, glob

    parser = argparse.ArgumentParser(description="Analytics Viz V2")
    parser.add_argument("--results",   nargs="+", help="模型结果文件：model_name:path ...")
    parser.add_argument("--diagnoses", nargs="*", help="诊断文件：model_name:path ...")
    parser.add_argument("--output-dir", default="outputs/visualizations")
    args = parser.parse_args()

    result_paths = {}
    if args.results:
        for item in args.results:
            name, path = item.split(":", 1)
            result_paths[name] = path

    diagnosis_paths = {}
    if args.diagnoses:
        for item in args.diagnoses:
            name, path = item.split(":", 1)
            diagnosis_paths[name] = path

    viz = AnalyticsVizV2(output_dir=args.output_dir)
    viz.generate_all(result_paths, diagnosis_paths or None)


if __name__ == "__main__":
    main()
