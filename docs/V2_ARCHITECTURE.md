# SelfPhy-Agent-System V2 Architecture Documentation

**Version**: V2.0  
**Last Updated**: 2026-06-04

## Core Design Philosophy

### The Critical Problem in V1 (Fixed)

**V1 Approach (Wrong)**:
```
Claude generates: "I was facing a door, then turned right 90 degrees, now I see a table. Where is the door?"

Target model: Only needs text reasoning -> "The door is on my left"

Testing: Reading comprehension + Geometry reasoning (NOT embodied understanding)
```

**V2 Approach (Correct)**:
```
Claude generates: "Where is the object that was initially on your left side?"

Target model receives:
- Pure question (no scene description)
- Multiple frames [frame_0, frame_15, frame_30, frame_45]
- Trajectory data [position, rotation, ...]

Target model must:
1. Look at frame_0 to identify "sofa on the left"
2. Look at frames 15-45 to understand "I am turning"
3. Build spatial memory map -> "sofa is now at right-back"

Testing: Visual localization + Self-motion perception + Spatial memory mapping
```

## Data Flow

```
Habitat Dataset (RGB + Depth + Pose Matrix)
    ↓
extract_all_episodes.py
    ↓
trajectory.json (precise Habitat data)
    ↓
habitat_metadata_builder.py (extract yaw/displacement)
    ↓
metadata.json (keyframes + statistics)
    ↓
claude_examiner_v2.py (supervisor generates pure questions)
    ↓
exam_v2.json (QuestionV2 format)
    ↓
evaluate_runner_v2.py (send pure question + multimodal data)
    ↓
result_v2.json (EvaluationResultV2 format)
```

## Core Modules

### 1. schema/question_v2.py
Data model definitions using Pydantic

**Key Classes**:
- `Pose6DoF`: 6-DOF pose representation
- `QuestionV2`: Question without scene description
- `ExamPaperV2`: Complete exam with multimodal evidence
- `EvaluationResultV2`: Evaluation results with metrics

### 2. core/habitat_metadata_builder.py
Extract metadata from Habitat trajectory

**Functions**:
- `quaternion_to_euler()`: Convert quaternion to Euler angles
- `compute_displacement()`: Calculate position displacement
- `extract_keyframes()`: Extract key motion moments
- `build_metadata()`: Generate complete metadata

**Usage**:
```bash
python3 core/habitat_metadata_builder.py \
    trajectory.json \
    -o metadata.json \
    --yaw-threshold 20.0 \
    --displacement-threshold 0.5
```

### 3. core/claude_examiner_v2.py
Supervisor agent for question generation

**Key Principle**: Generate questions WITHOUT scene descriptions

**Question Categories**:
1. egocentric_memory: "Where is the object that was initially on your left?"
2. spatial_transformation: "After turning, where is the object that was on your left?"
3. occlusion_reasoning: "How many objects were on the table before it was occluded?"
4. trajectory_backtracking: "If you move back 2 meters, will you return to the initial door?"
5. distance_estimation: "How far are you from the initial sofa?"

**Usage**:
```bash
python3 core/claude_examiner_v2.py \
    metadata.json \
    --data-dir episode_dir \
    -n 5 \
    -o exam_v2.json
```

### 4. core/evaluate_runner_v2.py
Send multimodal input to target models

**Supported Models**:
- Kimi (Moonshot)
- MiniMax
- Doubao (partial support)

**Input Format**:
```json
{
  "question": "Where is the object that was initially on your left?",
  "frames": [
    {"frame_id": 0, "rgb": "<base64>", "depth": "<base64>"},
    {"frame_id": 15, "rgb": "<base64>", "depth": "<base64>"}
  ],
  "trajectory": [
    {"frame_id": 0, "position": [0, 0, 0], "rotation": [1, 0, 0, 0]},
    {"frame_id": 45, "position": [2.3, 0, 1.1], "rotation": [0.7, 0, 0.7, 0]}
  ]
}
```

**Usage**:
```bash
python3 core/evaluate_runner_v2.py \
    exam_v2.json \
    --model kimi \
    -o result_v2.json
```

## V1 vs V2 Comparison

| Dimension | V1 | V2 |
|-----------|----|----|
| **Testing Capability** | Reading + Geometry | Visual Memory + Spatial Mapping |
| **Model Input** | Scene description + images | Pure question + frames + trajectory |
| **Data Source** | Optical flow (±5° error) | Habitat Pose Matrix (±0.01° error) |
| **Question Example** | "I face door, turn 90°, where is door?" | "Where is the initially left-side object?" |
| **Difficulty** | Easy (text reasoning) | Hard (video understanding required) |
| **Academic Value** | Incremental improvement | Novel evaluation paradigm |

## Question Design Principles

### 5 Capability Dimensions

**1. egocentric_memory (First-person memory)**
- Question: "Where is X initially?" / "Where is X now?"
- Testing: Remember initial state, understand relative changes after self-motion

**2. spatial_transformation (Spatial transformation understanding)**
- Question: "After turning, where is the object that was on your left?"
- Testing: Understand coordinate system transformation caused by self-rotation

**3. occlusion_reasoning (Occlusion reasoning)**
- Question: "How many objects were on the table before occlusion?"
- Testing: Build memory before occlusion, reason after occlusion

**4. trajectory_backtracking (Trajectory backtracking)**
- Question: "If you move back 2 meters, will you return to the initial door?"
- Testing: Understand complete trajectory, perform reverse reasoning

**5. distance_estimation (Distance estimation)**
- Question: "How far are you from the initial sofa?"
- Testing: Estimate distance from displacement accumulation

## Quick Start

### Single Episode Test

```bash
# Step 1: Build metadata
python3 core/habitat_metadata_builder.py \
    data/VL-LN-Bench/D7N2EKCX4Sj/data_test/episode_000001/trajectory.json \
    -o data/VL-LN-Bench/D7N2EKCX4Sj/data_test/episode_000001/metadata.json

# Step 2: Generate exam
python3 core/claude_examiner_v2.py \
    data/VL-LN-Bench/D7N2EKCX4Sj/data_test/episode_000001/metadata.json \
    --data-dir data/VL-LN-Bench/D7N2EKCX4Sj/data_test/episode_000001 \
    -n 5

# Step 3: Evaluate model
python3 core/evaluate_runner_v2.py \
    data/VL-LN-Bench/D7N2EKCX4Sj/data_test/episode_000001/exam_v2.json \
    --model kimi
```

### Batch Testing

```bash
python3 test_v2_pipeline.py
```

## Expected Results

### Discrimination Improvement

**V1 Results (Wrong)**:
- Kimi: 85%
- MiniMax: 82%
- Doubao: 80%

Small difference because all doing text reasoning

**V2 Expected (Correct)**:
- Kimi: 65% (long context advantage)
- MiniMax: 55% (medium visual understanding)
- Doubao: 45% (weaker spatial reasoning)

Large difference reveals true capability boundaries

### Academic Value

| Dimension | V1 | V2 |
|-----------|----|----|
| **Innovation** | Incremental | Novel paradigm |
| **Paper Target** | Workshop | Main conference |
| **Citation Value** | Low | High |
| **Contribution** | Engineering | Academic innovation |

## Troubleshooting

### Q1: Claude API call fails
Check API key and network:
```bash
echo $ANTHROPIC_API_KEY
curl -I https://api.anthropic.com
```

### Q2: Image files not found
Ensure trajectory.json directory has rgb/ and depth/ subdirectories

### Q3: Low question quality
Adjust temperature in claude_examiner_v2.py:
- Current: 0.7
- More stable: 0.3
- More diverse: 0.9

## Next Steps

1. **Phase 1 Complete**: Core architecture + data flow refactoring
2. **Phase 2**: Enhanced RAG (3-level index system)
3. **Phase 3**: 5D diagnostic matrix (Evidence Chain)
4. **Phase 4**: Visualization + paper materials

---

**Documentation Version**: V2.0  
**Last Updated**: 2026-06-04
