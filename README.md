# SelfPhy-Agent-System

**基于第一人称感知演化与自反思的空间推理自动化评测系统**

> 系统化评测大模型在第一人称动态物理世界中的空间记忆与推理能力（Egocentric Spatial Recall）

---

## 项目背景

现有多模态大模型 Benchmark 大多基于静态第三人称视角，缺乏对第一人称连续运动中"物体遮挡、视场盲区、空间回溯"的深度考核。SelfPhy-Agent-System 填补这一学术空白，构建了一套完整的自动化闭环评测流水线。

系统同时扮演四个角色：
- **自动化出题平台**：基于 Habitat 6-DoF 轨迹数据自动生成空间推理考题
- **自动化监考平台**：统一派发考题给多个被测大模型并收集回答
- **错误诊断平台**：通过四步排除法 + RAG 知识库进行根因分析
- **可视化分析平台**：雷达图、遗忘曲线、热力图等多维度看板

---

## 非对称式长官-被测智能体架构

```
┌─────────────────────────────────────────────────────────┐
│              Supervisor Agent（长官模型）                │
│         claude-sonnet-4-6  ← 出题 + 错题诊断             │
└──────────────────────┬──────────────────────────────────┘
                       │ 派发考题 / 收集回答
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Kimi (Moonshot)  MiniMax       豆包 (Doubao)
   长序列空间记忆   多模态辨识度   空间几何计算
   遗忘边界测试     物体幻觉率     方位转换正确率
```

---

## 系统架构与数据流

```
视频序列 (Habitat 3D)
    │
    ▼
[cv2_processor.py]  数据感知层
    关键帧提取（yaw 阈值 + 光流运动分数）
    6-DoF 位姿解析 → TrajectorySegment JSON
    │
    ▼
[claude_examiner.py]  考试生成器（Supervisor Agent）
    Claude Sonnet 4.6 自动出题
    四类能力：方向记忆 / 遮挡推理 / 路径回溯 / 距离估算
    输出 exam_raw.json
    │
    ▼
[evaluate_runner.py]  测试执行器
    统一适配器接口 → 派发给 Kimi / MiniMax / 豆包
    记录响应时间 + token 消耗
    │
    ▼  （答案错误时触发）
[claude_reflector.py]  错题分析器（Inspector Agent）
    四步排除法 + ChromaDB RAG 物理规则库
    输出 ErrorAnalysis JSON
    │
    ▼
[analytics_viz.py]  可视化层
    雷达图 / 遗忘曲线 / 热力图 / 综合报告
    │
    ▼
[app.py]  Streamlit 交互看板
```

---

## 快速开始

### 1. 环境配置

```bash
conda create -n selfphy python=3.10
conda activate selfphy
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# 长官模型（必填）
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_BASE_URL=https://api.anthropic.com   # 国内可替换为代理地址

# 被测模型（按需填写）
MOONSHOT_API_KEY=xxxxx          # Kimi
MINIMAX_API_KEY=xxxxx           # MiniMax
DOUBAO_API_KEY=xxxxx            # 豆包
DOUBAO_ENDPOINT=xxxxx
```

### 3. 准备数据

如无真实 Habitat 数据，可生成示例数据：

```bash
python create_sample_data.py                        # 默认 3 个 episode
python create_sample_data.py --num-episodes 5 --num-frames 40
```

### 4. 运行评测

```bash
# 单 episode 完整流程
python run_pipeline.py

# 批量处理所有 episode
python run_pipeline_batch.py
```

### 5. 分步运行（高级）

```bash
python core/cv2_processor.py       # 提取关键帧与轨迹
python core/claude_examiner.py     # 生成考题
python core/evaluate_runner.py     # 派发给被测模型
python core/claude_reflector.py    # 错题诊断
python core/analytics_viz.py       # 生成可视化
```

### 6. 查看结果

```bash
streamlit run app.py               # 启动 Web 看板
cat outputs/reports/DIAGNOSIS_*.md # 查看诊断报告
```

---

## 模块说明

### `core/cv2_processor.py` — 数据感知层

- 从 Habitat 帧序列中提取 6-DoF 位姿（position + quaternion）
- 基于 yaw 偏转角（默认阈值 20°）和光流运动分数（默认阈值 15）筛选关键帧
- 输出 `TrajectorySegment` 结构，含自然语言轨迹描述

### `core/claude_examiner.py` — 考试生成器

- 长官模型：`claude-sonnet-4-6`，启用 adaptive thinking 提升推理质量
- 四类考题能力维度：
  - **Relative Direction Memory**：转身后判断物体相对方向
  - **Occlusion Reasoning**：物体被遮挡后的位置推断
  - **Spatial Backtracking**：回溯路径中的空间关系
  - **Distance Estimation**：运动距离与物体远近估算
- 难度与 yaw 幅度正相关（<45° 简单 / 45-90° 中等 / >90° 困难）

### `core/evaluate_runner.py` — 测试执行器

- 统一适配器接口，支持多模态输入（图片 + 文本）
- 支持模型：Claude / Kimi / MiniMax / 豆包 / GPT-4o
- 自动重试 + 错误处理，记录响应时间和 token 消耗

### `core/claude_reflector.py` — 错题分析器（四步排除法）

| 步骤 | 名称 | 检测内容 |
|------|------|----------|
| Step 1 | 物理及语义对齐检查 | 方向矛盾、物理定律违反 |
| Step 2 | 空间位置重塑验证 | 空间拓扑重建错误、距离估算偏差 |
| Step 3 | 视场边界校验 | 遮挡问题、视野盲区、时间间隔过大 |
| Step 4 | 根因分类归纳 | 自动归类为 6 种错误类型 |

错误类型枚举：`physical_misalignment` / `spatial_topology_error` / `fov_boundary_issue` / `memory_decay` / `object_hallucination` / `occlusion_misunderstanding`

### `core/rag_manager.py` — RAG 知识库

- 基于 ChromaDB 的物理规则向量库
- 内置 8 条空间物理规则，支持语义检索和标签过滤
- 支持规则导入导出

### `core/analytics_viz.py` — 可视化层

- 雷达图：多模型错误类型对比
- 遗忘曲线：时间间隔 vs 准确率
- 热力图：空间层级 vs 时间间隔
- 综合报告：多子图汇总

### `app.py` — Streamlit 交互看板

- 实时展示评测进度与结果
- 支持多 episode、多模型对比视图

---

## 数据格式规范

### 轨迹片段 `TrajectorySegment`

```json
{
  "segment_id": "episode_001",
  "video_source": "habitat/episode_001",
  "start_time": 0.0,
  "end_time": 10.0,
  "spatial_narrative": "Agent moved forward 2.3m, then turned right 90°",
  "keyframes": [
    {
      "frame_id": 42,
      "image_path": "outputs/frames/frame_042.jpg",
      "pose": {
        "frame_id": 42,
        "timestamp": 1.4,
        "position": [1.2, 0.0, -3.5],
        "orientation": [0.924, 0.0, 0.383, 0.0],
        "euler_angles": [0.0, 0.0, 45.0]
      },
      "detected_objects": []
    }
  ]
}
```

### 考题 `ExamItem`

```json
{
  "question": "我站在走廊中，正前方是一扇门，左侧是书架。我向右转了90°后，请问书架现在在我的哪个方向？",
  "answer": "左后方。向右转90°后，原来的左侧变为后方偏左。",
  "capability": "Relative Direction Memory",
  "difficulty": "简单",
  "reasoning_chain": "初始：门=前，书架=左。右转90°后坐标系旋转，书架相对新朝向为左后方。",
  "frame_name": "frame_042.jpg",
  "yaw": 90.0,
  "motion_score": 28.5,
  "image_path": "outputs/frames/frame_042.jpg"
}
```

### 错误分析 `ErrorAnalysis`

```json
{
  "question_id": "Q001_recall",
  "model_name": "kimi",
  "error_type": "spatial_topology_error",
  "causal_trace": [
    {
      "step_name": "Physical Alignment Check",
      "hypothesis": "Model's answer violates physical laws",
      "evidence": ["Model mentioned opposite direction: right"],
      "conclusion": "Physical misalignment detected",
      "confidence": 0.8
    }
  ],
  "root_cause": "Model failed to reconstruct spatial topology after rotation",
  "confidence": 0.72,
  "timestamp": "2026-06-01T10:30:00"
}
```

---

## 输出目录结构

```
outputs/
├── frames/                    # 关键帧图片
│   └── frame_042.jpg
├── metadata/                  # 帧元数据
│   └── metadata.json
├── trajectories/              # 轨迹 JSON
│   └── episode_001.json
├── exams/                     # 生成的考题
│   ├── exam_raw.json
│   └── exam_episode_001.json
├── answers/                   # 模型回答
│   └── responses_kimi_20260601_103000.json
├── reports/                   # 分析报告
│   ├── error_analysis_20260601_103000.json
│   ├── basic_report_20260601_103000.txt
│   └── DIAGNOSIS_20260601_103000.md
└── visualizations/            # 可视化图表
    ├── radar_chart.png
    ├── forgetting_curve.png
    ├── heatmap.png
    └── comprehensive_report.png
```

---

## 项目结构

```
SelfPhy-Agent-System/
├── core/
│   ├── cv2_processor.py        # 数据感知层：关键帧提取 + 6-DoF 解析
│   ├── habitat_loader.py       # Habitat 数据加载器
│   ├── claude_examiner.py      # 考试生成器（Supervisor Agent）
│   ├── claude_adapter.py       # Claude API 适配器
│   ├── claude_reflector.py     # 错题分析器（Inspector Agent）
│   ├── evaluate_runner.py      # 测试执行器（多模型适配）
│   ├── exam_formatter.py       # 考题格式化器
│   ├── prompt_generator.py     # Prompt 构建工具
│   ├── rag_manager.py          # RAG 知识库（ChromaDB）
│   ├── analytics_viz.py        # 可视化层
│   └── metadata_exporter.py    # 元数据导出器
├── config/
│   ├── model_config.json       # 模型配置（含 API 端点）
│   └── model_config.example.json
├── data/
│   ├── raw/habitat/            # 原始 Habitat 数据
│   └── chroma_db/              # ChromaDB 向量数据库
├── outputs/                    # 评测输出（自动生成）
├── app.py                      # Streamlit 交互看板
├── run_pipeline.py             # 单 episode 完整流程
├── run_pipeline_batch.py       # 批量处理脚本
├── create_sample_data.py       # 示例数据生成器
├── test_claude_quick.py        # Claude API 快速测试
├── requirements.txt
├── .env.example
└── README.md
```

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 仿真环境 | Meta Habitat 3D + Ego4D |
| 数据处理 | OpenCV-headless, NumPy, SciPy |
| AI 编排 | Anthropic Claude (claude-sonnet-4-6) |
| 被测模型 | Kimi (Moonshot), MiniMax, 豆包 (Doubao) |
| 向量数据库 | ChromaDB |
| 可视化 | Matplotlib, Seaborn, Plotly |
| 前端看板 | Streamlit |
| 运行环境 | Python 3.10, Conda, Ubuntu (阿里云) |

---

## 常见问题

**Q: API 调用失败（ConnectionError）**

检查 `.env` 中的 `ANTHROPIC_API_KEY` 是否正确，国内环境可在 `.env` 中设置 `ANTHROPIC_BASE_URL` 为代理地址。

**Q: 没有生成报告**

确认 API 调用成功后查看 `outputs/reports/basic_report_*.txt`，至少需要有成功回答才能生成完整报告。

**Q: 只处理了一个 episode**

使用 `run_pipeline_batch.py` 批量处理所有 episode。

**Q: 如何添加新的被测模型**

在 `config/model_config.json` 中添加模型配置，并在 `core/evaluate_runner.py` 中实现对应的适配器类。

---

## 更新日志

### 2026-06-01
- 将长官模型统一为 `claude-sonnet-4-6`，启用 adaptive thinking
- 新增 `claude_examiner.py`（替代旧版 `gpt4o_examiner.py`）
- 新增 `claude_reflector.py`（替代旧版 `gpt4o_reflector.py`）
- 新增 `claude_adapter.py` 统一 Claude API 调用层
- 新增批量处理支持（`run_pipeline_batch.py`）
- 完善报告生成逻辑（失败时也生成基础报告）

### 2026-05-28（初始版本）
- 实现完整评测流水线
- 支持多模型适配器接口
- 四步排除法错误诊断
- Streamlit 可视化看板

---

## 许可证

MIT License
