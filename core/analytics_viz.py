"""
可视化分析层 (Analytics Visualizer)
负责生成雷达图、饼图、热力图等可视化报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class AnalyticsVisualizer:
    """可视化分析层"""

    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_radar_chart(
        self,
        analyses_paths: Dict[str, str],
        output_filename: str = "radar_chart.png"
    ) -> str:
        """
        生成雷达图（各维度错误率对比）

        Args:
            analyses_paths: {model_name: analysis_file_path}
            output_filename: 输出文件名

        Returns:
            图片保存路径
        """
        # 错误类型维度
        error_dimensions = [
            "physical_misalignment",
            "spatial_topology_error",
            "fov_boundary_issue",
            "memory_decay",
            "object_hallucination",
            "occlusion_misunderstanding"
        ]

        dimension_labels = [
            "Physical\nMisalignment",
            "Spatial\nTopology",
            "FOV\nBoundary",
            "Memory\nDecay",
            "Object\nHallucination",
            "Occlusion\nMisunderstanding"
        ]

        # 收集每个模型的错误分布
        model_data = {}

        for model_name, analysis_path in analyses_paths.items():
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)

            # 统计错误类型
            error_counts = Counter([a['error_type'] for a in analyses])
            total = len(analyses) if analyses else 1

            # 计算每个维度的错误率
            error_rates = [error_counts.get(dim, 0) / total for dim in error_dimensions]
            model_data[model_name] = error_rates

        # 绘制雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(error_dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 绘制每个模型
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for i, (model_name, error_rates) in enumerate(model_data.items()):
            values = error_rates + error_rates[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Error Type Distribution by Model', size=16, pad=20)

        output_path = self.output_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Saved] Radar chart: {output_path}")
        return str(output_path)

    def generate_forgetting_curve(
        self,
        analyses_path: str,
        questions_path: str,
        output_filename: str = "forgetting_curve.png"
    ) -> str:
        """
        生成遗忘曲线（时间间隔 vs 准确率）

        Args:
            analyses_path: 错误分析文件路径
            questions_path: 问题文件路径
            output_filename: 输出文件名

        Returns:
            图片保存路径
        """
        # 加载数据
        with open(analyses_path, 'r', encoding='utf-8') as f:
            analyses = json.load(f)

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # 构建问题字典
        questions_dict = {q['question_id']: q for q in questions}

        # 收集时间间隔和错误率
        time_gaps = []
        for analysis in analyses:
            question = questions_dict.get(analysis['question_id'])
            if question and question.get('time_gap') is not None:
                time_gaps.append(question['time_gap'])

        # 统计每个时间间隔的错误数
        if not time_gaps:
            print("[Warning] No time gap data available")
            return ""

        # 分桶统计
        max_gap = max(time_gaps)
        bins = np.linspace(0, max_gap, 10)
        bin_errors = np.histogram(time_gaps, bins=bins)[0]

        # 计算总问题数（假设均匀分布）
        total_questions = len(questions)
        bin_totals = [total_questions // len(bins)] * len(bins)

        # 计算错误率
        error_rates = [errors / max(total, 1) for errors, total in zip(bin_errors, bin_totals)]

        # 绘制曲线
        fig, ax = plt.subplots(figsize=(10, 6))

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, error_rates, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
        ax.fill_between(bin_centers, error_rates, alpha=0.3, color='#FF6B6B')

        ax.set_xlabel('Time Gap (frames)', fontsize=12)
        ax.set_ylabel('Error Rate', fontsize=12)
        ax.set_title('Spatial Memory Forgetting Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        output_path = self.output_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[Saved] Forgetting curve: {output_path}")
        return str(output_path)

    def generate_heatmap(
        self,
        analyses_path: str,
        questions_path: str,
        output_filename: str = "heatmap.png"
    ) -> str:
        """
        生成热力图（空间层级 vs 时间间隔）

        Args:
            analyses_path: 错误分析文件路径
            questions_path: 问题文件路径
            output_filename: 输出文件名

        Returns:
            图片保存路径
        """
        # 加载数据
        with open(analyses_path, 'r', encoding='utf-8') as f:
            analyses = json.load(f)

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # 构建问题字典
        questions_dict = {q['question_id']: q for q in questions}

        # 空间层级
        spatial_levels = ["room", "area", "object", "attribute"]

        # 时间间隔分桶
        time_bins = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]
        time_labels = ["0-2", "2-5", "5-10", "10+"]

        # 构建矩阵
        matrix = np.zeros((len(spatial_levels), len(time_bins)))

        for analysis in analyses:
            question = questions_dict.get(analysis['question_id'])
            if not question:
                continue

            spatial_level = question.get('spatial_level')
            time_gap = question.get('time_gap', 0)

            if spatial_level in spatial_levels:
                level_idx = spatial_levels.index(spatial_level)

                # 找到时间桶
                for bin_idx, (low, high) in enumerate(time_bins):
                    if low <= time_gap < high:
                        matrix[level_idx, bin_idx] += 1
                        break

        # 归一化（转换为错误率）
        # 这里简化处理，实际应该除以每个格子的总问题数
        matrix = matrix / (matrix.sum() + 1e-6)

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            xticklabels=time_labels,
            yticklabels=spatial_levels,
            cbar_kws={'label': 'Error Rate'},
            ax=ax
        )

        ax.set_xlabel('Time Gap (frames)', fontsize=12)
        ax.set_ylabel('Spatial Level', fontsize=12)
        ax.set_title('Error Distribution: Spatial Level vs Time Gap', fontsize=14)

        output_path = self.output_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[Saved] Heatmap: {output_path}")
        return str(output_path)

    def generate_summary_report(
        self,
        analyses_paths: Dict[str, str],
        questions_path: str,
        output_filename: str = "summary_report.png"
    ) -> str:
        """
        生成综合报告（多子图）

        Args:
            analyses_paths: {model_name: analysis_file_path}
            questions_path: 问题文件路径
            output_filename: 输出文件名

        Returns:
            图片保存路径
        """
        fig = plt.figure(figsize=(16, 10))

        # 子图1: 总体准确率对比
        ax1 = plt.subplot(2, 3, 1)
        model_names = []
        accuracy_rates = []

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        total_questions = len(questions)

        for model_name, analysis_path in analyses_paths.items():
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)

            error_count = len(analyses)
            accuracy = (total_questions - error_count) / total_questions if total_questions > 0 else 0

            model_names.append(model_name)
            accuracy_rates.append(accuracy)

        ax1.bar(model_names, accuracy_rates, color=['#4ECDC4', '#FF6B6B', '#45B7D1'])
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Overall Accuracy by Model', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)

        # 子图2: 错误类型分布（饼图）
        ax2 = plt.subplot(2, 3, 2)
        all_errors = []
        for analysis_path in analyses_paths.values():
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
            all_errors.extend([a['error_type'] for a in analyses])

        error_counts = Counter(all_errors)
        ax2.pie(
            error_counts.values(),
            labels=[e.replace('_', '\n') for e in error_counts.keys()],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Error Type Distribution', fontsize=12)

        # 子图3: 难度分布
        ax3 = plt.subplot(2, 3, 3)
        difficulties = [q['difficulty'] for q in questions]
        difficulty_counts = Counter(difficulties)
        ax3.bar(difficulty_counts.keys(), difficulty_counts.values(), color=['#90EE90', '#FFD700', '#FF6347'])
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('Question Difficulty Distribution', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)

        # 子图4-6: 每个模型的错误类型分布
        for idx, (model_name, analysis_path) in enumerate(analyses_paths.items()):
            if idx >= 3:
                break

            ax = plt.subplot(2, 3, 4 + idx)

            with open(analysis_path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)

            error_counts = Counter([a['error_type'] for a in analyses])

            ax.barh(
                list(error_counts.keys()),
                list(error_counts.values()),
                color='#4ECDC4'
            )
            ax.set_xlabel('Count', fontsize=10)
            ax.set_title(f'{model_name} Error Types', fontsize=12)
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Saved] Summary report: {output_path}")
        return str(output_path)


def main():
    """主函数：生成可视化报告"""
    visualizer = AnalyticsVisualizer()

    # 示例路径
    analyses_paths = {
        "gpt-4o-mini": "outputs/reports/error_analysis_20260526_120000.json"
    }
    questions_path = "outputs/exams/exam_episode_001.json"

    # 检查文件是否存在
    for path in list(analyses_paths.values()) + [questions_path]:
        if not Path(path).exists():
            print(f"[Error] File not found: {path}")
            print("Please run the previous steps first.")
            return

    try:
        # 生成雷达图
        visualizer.generate_radar_chart(analyses_paths, "radar_chart.png")

        # 生成遗忘曲线
        visualizer.generate_forgetting_curve(
            analyses_paths["gpt-4o-mini"],
            questions_path,
            "forgetting_curve.png"
        )

        # 生成热力图
        visualizer.generate_heatmap(
            analyses_paths["gpt-4o-mini"],
            questions_path,
            "heatmap.png"
        )

        # 生成综合报告
        visualizer.generate_summary_report(
            analyses_paths,
            questions_path,
            "summary_report.png"
        )

        print("\n[Success] All visualizations generated!")

    except Exception as e:
        print(f"[Error] Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
