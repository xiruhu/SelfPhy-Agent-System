# SelfPhy-Agent-System

**基于第一人称感知的多模态大模型空间推理自动化评测系统**

> 系统化评测多模态大模型在第一人称动态物理世界中的空间记忆与推理能力（Egocentric Spatial Reasoning）

---

## 项目背景

现有多模态大模型 Benchmark 大多基于静态第三人称视角，缺乏对第一人称连续运动中"物体遮挡、视场盲区、空间回溯"的深度考核。SelfPhy-Agent-System 填补这一学术空白，构建了一套完整的自动化闭环评测流水线。

**核心设计原则**：被测模型只能看到连续帧图像序列，不接收任何场景文字描述，必须通过"看视频"来推断空间关系。这真正测试的是视觉定位 + 自运动感知 + 空间记忆映射能力。

---

## 四 Agent 架构（V2.2）

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent 1 — Claude Sonnet 4.6（Examiner）                        │
│  输入：关键帧图像 + 精确位姿轨迹                                  │
│  输出：exam_v2.json（5 题 / 场景，含 reasoning_trace）            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 2 — GPT-4o（Quality Checker）                            │
│  输入：同一批关键帧图像 + 轨迹数据 + Claude 的题目               │
│  职责：① 独立验算答案数学正确性                                   │
│        ② 判断题干是否真正考察了视觉空间推理                       │
│  输出：exam_refined_v2.json（删错题 / 更正答案 / 优化题干）       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 精炼考卷
           ┌────────────────┴────────────────┐
           ▼                                  ▼
┌──────────────────┐              ┌──────────────────────┐
│  Agent 3a        │              │  Agent 3b             │
│  Kimi (kimi-k2.5)│              │  豆包 (Doubao)        │
│  Moonshot API    │              │  火山引擎 Ark          │
└────────┬─────────┘              └──────────┬───────────┘
         └────────────────┬──────────────────┘
                          │ 作答结果
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 4 — Claude Sonnet 4.6（Reflector）                       │
│  输入：错题 + 精炼考卷 + 位姿轨迹 + 关键帧图像                   │
│  职责：三步排除法根因诊断                                         │
│  输出：diagnosis_*.json（错误类型 + 置信度 + 推导过程）           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 数据流

```
VL-LN-Bench 数据集（10 场景 × tar.gz）
    ↓  tools/extract_all_episodes.py
data/VL-LN-Bench/extracted/<SCENE>/episode_000000/
    trajectory.json / rgb/ / depth/
    ↓  core/habitat_metadata_builder.py
episode_000000/metadata.json
（关键帧标注 + 偏航角 + 位移增量）
    ↓  [Step 3]  core/claude_examiner.py
outputs/runs/<RUN_ID>/exams/exam_<SCENE>_episode_000000.json
    ↓  [Step 3.5] core/gpt_quality_checker.py
outputs/runs/<RUN_ID>/exams/exam_refined_<SCENE>_episode_000000.json
    ↓  [Step 4]  core/evaluate_runner.py（Kimi + 豆包）
outputs/runs/<RUN_ID>/answers/result_<MODEL>_<SCENE>_episode_000000.json
    ↓  [Step 5]  core/claude_reflector.py（三步排除法）
outputs/runs/<RUN_ID>/reports/diagnosis_<MODEL>_<SCENE>_episode_000000.json
    ↓  [Step 6]  core/analytics_viz.py
outputs/runs/<RUN_ID>/visualizations/
```

---

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env 填写各模型 API 密钥
```

`.env` 变量说明：

```env
# Agent 1: Claude 出题模型（必填）
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_BASE_URL=https://api.anthropic.com   # 支持代理地址

# Agent 2: GPT-4o 质检模型（必填）
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1       # 支持代理地址
OPENAI_MODEL=gpt-4o

# Agent 3a: Kimi 被测模型
MOONSHOT_API_KEY=sk-xxxxx
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=kimi-k2.5

# Agent 3b: 豆包被测模型
DOUBAO_API_KEY=ark-xxxxx
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=ep-xxxxxxxxxxxxxxxx   # 火山引擎视觉 endpoint ID
```

### 3. 运行完整评测

```bash
# 全新运行（自动生成 RUN_ID）
nohup bash run_full_pipeline.sh > /tmp/pipeline.log 2>&1 &

# 指定 RUN_ID
RUN_ID=run2 nohup bash run_full_pipeline.sh > /tmp/pipeline.log 2>&1 &

# 复用已有考卷（跳过 Claude 出题，直接从 GPT 质检开始）
RUN_ID=run2 EXAM_SOURCE_RUN=run1 nohup bash run_full_pipeline.sh > /tmp/pipeline.log 2>&1 &
```

Pipeline 自动执行以下步骤（已完成的步骤自动 skip）：

| 步骤 | 模块 | 说明 |
|------|------|------|
| Step 1 | `tools/extract_all_episodes.py` | 从 tar.gz 解压场景数据 |
| Step 2 | `core/habitat_metadata_builder.py` | 生成位姿 metadata |
| Step 3 | `core/claude_examiner.py` | Claude 出题（每场景 5 题）|
| Step 3.5 | `core/gpt_quality_checker.py` | GPT-4o 质检（验算答案 + 判断有效性）|
| Step 4 | `core/evaluate_runner.py` | Kimi + 豆包答题 |
| Step 5 | `core/claude_reflector.py` | Claude 错题根因诊断 |
| Step 6 | `core/analytics_viz.py` | 生成可视化图表 |

### 4. 查看结果

```bash
streamlit run streamlit_dashboard_final.py --server.port 8501
# 本地访问：ssh -N -L 8501:localhost:8501 <server>  →  http://localhost:8501
```

---

## 五维能力评测体系

| 能力维度 | 中文 | 题目示例 |
|---------|------|---------|
| `egocentric_memory` | 空间记忆 | "最初在你左边的物体现在在哪里？" |
| `spatial_transformation` | 空间变换 | "转身后，原来左边的物体在哪里？" |
| `occlusion_reasoning` | 遮挡推理 | "被遮挡前桌上有几件物体？" |
| `trajectory_backtracking` | 轨迹回溯 | "退回 2 米后会回到初始门口吗？" |
| `distance_estimation` | 距离估算 | "你距离初始沙发有多远？" |

**答案判断策略**：
- 方向类（4 种）：同义词归一化 + 最长匹配（支持中英文别名、角度表示）
- 距离类：允许 ±20% 数值误差范围

---

## 六类错误根因（Claude Reflector 三步排除法）

| 错误类型 | 说明 |
|---------|------|
| `direction_calc_error` | 方向计算错误（如左右混淆） |
| `rotation_sense_error` | 旋转方向错误（顺逆时针判断失误） |
| `rotation_translation_confusion` | 旋转/平移混淆 |
| `memory_decay` | 空间记忆衰减（长序列后遗忘初始状态） |
| `object_hallucination` | 物体幻觉（凭空出现未见过的物体） |
| `fov_misunderstanding` | 视野误解（误判视场范围内外） |

---

## 输出目录结构

```
outputs/
└── runs/
    └── <RUN_ID>/
        ├── exams/
        │   ├── exam_<SCENE>_episode_000000.json          # Claude 原始考卷
        │   └── exam_refined_<SCENE>_episode_000000.json  # GPT-4o 精炼考卷
        ├── answers/
        │   └── result_<MODEL>_<SCENE>_episode_000000.json
        ├── reports/
        │   └── diagnosis_<MODEL>_<SCENE>_episode_000000.json
        ├── visualizations/
        │   ├── radar_chart.png
        │   ├── difficulty_accuracy.png
        │   ├── error_pie.png
        │   └── summary.png
        └── pipeline.log
```

---

## 模块说明

| 模块 | 职责 |
|------|------|
| `core/habitat_metadata_builder.py` | Habitat 位姿解析，提取偏航角/位移，标注关键帧 |
| `core/claude_examiner.py` | Agent 1：出题，5 类能力，extended thinking，3 次重试 |
| `core/gpt_quality_checker.py` | Agent 2：逐题质检，图像+轨迹双验证，输出精炼考卷 |
| `core/evaluate_runner.py` | Agent 3：多模型评测，支持 kimi / doubao |
| `core/claude_reflector.py` | Agent 4：三步排除法错因诊断，6 类根因分类 |
| `core/analytics_viz.py` | 生成静态图表（雷达图、柱状图、饼图） |
| `schema/question.py` | Pydantic 数据模型：QuestionV2 / ExamPaperV2 / EvaluationResultV2 |
| `streamlit_dashboard_final.py` | 交互式仪表盘，支持 RUN_ID 批次切换 |
| `tools/extract_all_episodes.py` | VL-LN-Bench tar.gz 解析 |
| `run_full_pipeline.sh` | 一键 Pipeline，含 skip 逻辑和 RUN_ID 管理 |

---

## 项目结构

```
SelfPhy-Agent-System/
├── core/
│   ├── habitat_metadata_builder.py   # Habitat 数据处理
│   ├── claude_examiner.py            # Agent 1: 出题
│   ├── gpt_quality_checker.py        # Agent 2: 质检
│   ├── evaluate_runner.py            # Agent 3: 多模型评测
│   ├── claude_reflector.py           # Agent 4: 错因诊断
│   └── analytics_viz.py              # 静态可视化
├── schema/
│   └── question.py                   # Pydantic 数据模型
├── tools/
│   └── extract_all_episodes.py       # VL-LN-Bench 解析
├── data/
│   └── VL-LN-Bench/
│       ├── traj_data/mp3d_split2/    # 原始 tar.gz
│       └── extracted/                # 提取后的场景数据
├── outputs/
│   └── runs/                         # 每次运行独立目录
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_REVIEW.md
│   └── QUICKSTART.md
├── streamlit_dashboard_final.py      # Streamlit 仪表盘
├── run_full_pipeline.sh              # 一键 Pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## 常见问题

**Q: 如何复用已有考卷跳过 Claude 出题？**

```bash
RUN_ID=run2 EXAM_SOURCE_RUN=run1 bash run_full_pipeline.sh
```
Pipeline 会从 run1/exams/ 复制原始考卷到 run2/exams/，然后从 Step 3.5（GPT 质检）开始执行。

**Q: 数据已提取过，会重新解压吗？**

不会。每个场景目录下有 `.extracted` 标记，pipeline 检测到后直接 skip。

**Q: 如何只重跑某个步骤？**

删除对应输出文件后重跑：
- 重新质检：`rm outputs/runs/<RUN_ID>/exams/exam_refined_*.json`
- 重新答题：`rm outputs/runs/<RUN_ID>/answers/*.json`
- 全新一轮：指定新 `RUN_ID` 运行

**Q: GPT-4o 质检失败了怎么办？**

质检失败时自动降级：原始考卷直接复制为精炼考卷，pipeline 继续执行，不中断。

---

## 更新日志

### 2026-06-17（V2.2）
- 新增 GPT-4o Quality Checker（Step 3.5）：基于关键帧图像 + 轨迹数据逐题验证
  - ① 独立验算 ground_truth_answer 数学正确性
  - ② 判断题干是否真正考察了视觉空间推理
  - 支持 delete / fix_answer / rewrite / keep 四种操作
- Pipeline 新增 `EXAM_SOURCE_RUN` 参数，支持复用已有考卷
- 架构升级为四 Agent 结构

### 2026-06-06（V2.1）
- 引入 RUN_ID 输出版本管理
- 新增 Streamlit 仪表盘批次选择器
- MiniMax 替换为豆包（MiniMax 无图像能力）
- 修复 Claude Examiner 偶发 JSON 解析错误

### 2026-06-04（V2.0）
- V1 → V2 核心重构：被测模型只能通过图像序列推理，不再接收文字场景描述
- 接入 VL-LN-Bench 真实数据集（10 场景）
- 实现五维能力评测体系 + 六类错误根因分类

---

## 许可证

MIT License
