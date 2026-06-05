# SelfPhy-Agent-System 快速开始指南（已更新）

## 重要更新

**长官模型已从 GPT-4o 更换为 Claude Sonnet 4.6**

## 1. 环境配置

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# Claude (Supervisor Agent - 必需)
ANTHROPIC_API_KEY=sk-ant-xxxxx

# 可选：其他被测模型
OPENAI_API_KEY=sk-xxxxx
MOONSHOT_API_KEY=xxxxx
```

## 2. 创建测试数据

如果没有真实的 Habitat 数据集，可以创建示例数据：

```bash
# 创建 3 个 episode（默认）
python create_sample_data.py

# 或指定数量
python create_sample_data.py --num-episodes 5 --num-frames 40
```

## 3. 运行评测

### 3.1 单个 Episode 评测

```bash
python run_pipeline.py
```

这会处理第一个 episode 并生成完整报告。

### 3.2 批量 Episode 评测

```bash
python run_pipeline_batch.py
```

这会处理所有 episode 并生成汇总报告。

## 4. 查看结果

### 4.1 文件输出

评测完成后，结果保存在以下位置：

```
outputs/
├── trajectories/          # 轨迹数据
│   └── episode_001.json
├── exams/                 # 生成的考题
│   └── exam_episode_001.json
├── answers/               # 模型回答
│   └── responses_claude-sonnet-4-6_*.json
├── reports/               # 错误分析报告
│   ├── error_analysis_*.json
│   └── basic_report_*.txt
└── visualizations/        # 可视化图表
    ├── radar_chart.png
    ├── forgetting_curve.png
    └── heatmap.png
```

### 4.2 启动 Web 看板

```bash
streamlit run app.py
```

在浏览器中打开 http://localhost:8501 查看交互式看板。

## 5. 常见问题

### Q1: API 调用失败（ConnectionError）

**原因**：API 密钥未配置或网络问题

**解决**：
1. 检查 `.env` 文件中的 `ANTHROPIC_API_KEY` 是否正确
2. 确认网络可以访问 Anthropic API
3. 如果在国内，可能需要配置代理

### Q2: 没有生成报告

**原因**：所有测试都失败了，没有有效数据

**解决**：
1. 检查 API 调用是否成功
2. 查看 `outputs/reports/basic_report_*.txt` 了解失败原因
3. 确保至少有一些成功的回答才能生成完整报告

### Q3: 只处理了一个 episode

**原因**：使用了 `run_pipeline.py`（单 episode 模式）

**解决**：使用 `run_pipeline_batch.py` 处理所有 episode

## 6. 模型配置

### 6.1 使用不同的 Supervisor 模型

编辑 `core/gpt4o_reflector.py` 的初始化参数：

```python
inspector = InspectorAgent(
    supervisor_model="claude-opus-4",  # 或 "claude-sonnet-4-6"
    use_claude=True
)
```

### 6.2 测试不同的被测模型

在 `run_pipeline.py` 中修改模型名称：

```python
model_name = "kimi"  # 或 "gpt-4o", "minimax", "doubao"
```

## 7. 评测指标说明

### 7.1 题型分布

- **recall**: 空间记忆回溯（测试模型是否记得之前看到的位置）
- **reasoning**: 空间推理（测试模型能否计算直线距离和方向）
- **counterfactual**: 反事实推理（测试模型的因果推理能力）

### 7.2 错误类型

- **physical_misalignment**: 违反物理定律
- **spatial_topology_error**: 空间拓扑错误
- **fov_boundary_issue**: 视场边界问题
- **memory_decay**: 空间记忆衰减
- **object_hallucination**: 物体幻觉
- **occlusion_misunderstanding**: 遮挡理解错误

## 8. 下一步

1. 查看生成的报告，分析模型表现
2. 调整评测参数（难度、题目数量等）
3. 测试不同的模型并对比结果
4. 如果有真实 Habitat 数据，替换 `data/raw/habitat/` 目录

## 9. 技术支持

如有问题，请查看：
- `REFACTOR_SUMMARY.md` - 系统架构说明
- `technical_blueprint.py` - 技术蓝图
- `CLAUDE.md` - 项目指导文档
