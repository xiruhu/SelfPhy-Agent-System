"""
错题分析器 (Inspector Agent)
负责四步排除法错误诊断和根因分析
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np


class ErrorType(str, Enum):
    """错误类型"""
    PHYSICAL_MISALIGNMENT = "physical_misalignment"
    SPATIAL_TOPOLOGY_ERROR = "spatial_topology_error"
    FOV_BOUNDARY_ISSUE = "fov_boundary_issue"
    MEMORY_DECAY = "memory_decay"
    OBJECT_HALLUCINATION = "object_hallucination"
    OCCLUSION_MISUNDERSTANDING = "occlusion_misunderstanding"


@dataclass
class CausalTraceStep:
    """因果追踪步骤"""
    step_name: str
    hypothesis: str
    evidence: List[str]
    conclusion: str
    confidence: float
    supporting_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorAnalysis:
    """错误分析报告"""
    question_id: str
    model_name: str
    error_type: ErrorType
    causal_trace: List[CausalTraceStep]
    retrieved_rules: List[str]
    root_cause: str
    confidence: float
    visualization_data: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class InspectorAgent:
    """错题分析器 - 四步排除法"""

    def __init__(
        self,
        supervisor_api_key: str = None,
        supervisor_model: str = "claude-sonnet-4-6",
        use_claude: bool = True
    ):
        self.supervisor_model = supervisor_model
        self.use_claude = use_claude
        self.output_dir = Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Claude 或 OpenAI 客户端
        if use_claude:
            self.supervisor_api_key = supervisor_api_key or os.getenv("ANTHROPIC_API_KEY")
            base_url = os.getenv("ANTHROPIC_BASE_URL")
            if self.supervisor_api_key:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.supervisor_api_key, base_url=base_url)
                print(f"[Inspector] Using Claude model: {supervisor_model}")
            else:
                self.client = None
                print("[Warning] No ANTHROPIC_API_KEY provided, using rule-based analysis only")
        else:
            self.supervisor_api_key = supervisor_api_key or os.getenv("OPENAI_API_KEY")
            if self.supervisor_api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.supervisor_api_key)
                print(f"[Inspector] Using OpenAI model: {supervisor_model}")
            else:
                self.client = None
                print("[Warning] No supervisor API key provided, using rule-based analysis only")

    def analyze_errors(
        self,
        responses_path: str,
        questions_path: str,
        trajectory_path: str
    ) -> List[ErrorAnalysis]:
        """
        批量分析错误

        Args:
            responses_path: 模型响应文件路径
            questions_path: 问题文件路径
            trajectory_path: 轨迹文件路径

        Returns:
            错误分析报告列表
        """
        # 加载数据
        with open(responses_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        with open(trajectory_path, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)

        # 构建问题字典
        questions_dict = {q['question_id']: q for q in questions}

        analyses = []

        print(f"\n[Analysis] Processing {len(responses)} responses...")

        for response in responses:
            question = questions_dict.get(response['question_id'])
            if not question:
                continue

            # 判断是否错误
            is_correct = self._check_correctness(response, question)

            if not is_correct:
                print(f"\n[Analyzing] {response['question_id']}...")

                analysis = self._analyze_single_error(
                    response,
                    question,
                    trajectory
                )

                analyses.append(analysis)

        return analyses

    def _check_correctness(self, response: Dict, question: Dict) -> bool:
        """检查答案是否正确"""
        ground_truth = question['ground_truth']
        parsed_answer = response['parsed_answer']

        # 如果有错误，直接判定为错误
        if 'error' in parsed_answer:
            return False

        # 如果响应为空或太短，判定为错误
        raw_response = response.get('raw_response', '')
        if not raw_response or len(raw_response.strip()) < 10:
            return False

        # 检查方向
        if 'direction' in ground_truth:
            gt_direction = ground_truth['direction'].lower()
            response_text = raw_response.lower()

            # 检查关键词是否匹配
            direction_keywords = gt_direction.split()
            matches = sum(1 for kw in direction_keywords if kw in response_text)

            if matches < len(direction_keywords) * 0.5:
                return False

        # 检查距离
        if 'distance_meters' in ground_truth:
            gt_distance = ground_truth['distance_meters']
            extracted_distance = parsed_answer.get('extracted_distance')

            if extracted_distance is not None:
                error_ratio = abs(extracted_distance - gt_distance) / max(gt_distance, 0.1)
                if error_ratio > 0.3:  # 30% 误差阈值
                    return False

        return True

    def _analyze_single_error(
        self,
        response: Dict,
        question: Dict,
        trajectory: Dict
    ) -> ErrorAnalysis:
        """
        四步排除法分析单个错误

        步骤：
        1. 物理及语义对齐检查
        2. 空间位置重塑验证
        3. 视场边界校验
        4. 根因分类归纳
        """
        causal_trace = []

        # Step 1: 物理及语义对齐
        step1 = self._step1_physical_alignment(response, question, trajectory)
        causal_trace.append(step1)

        # Step 2: 空间位置重塑
        step2 = self._step2_spatial_reconstruction(response, question, trajectory)
        causal_trace.append(step2)

        # Step 3: 视场边界校验
        step3 = self._step3_fov_verification(response, question, trajectory)
        causal_trace.append(step3)

        # Step 4: 根因分类
        error_type, root_cause = self._step4_root_cause_classification(causal_trace)

        # 计算总体置信度
        confidence = np.mean([step.confidence for step in causal_trace])

        analysis = ErrorAnalysis(
            question_id=question['question_id'],
            model_name=response['model_name'],
            error_type=error_type,
            causal_trace=causal_trace,
            retrieved_rules=[],  # TODO: 集成 RAG
            root_cause=root_cause,
            confidence=confidence
        )

        return analysis

    def _step1_physical_alignment(
        self,
        response: Dict,
        question: Dict,
        trajectory: Dict
    ) -> CausalTraceStep:
        """步骤 1: 物理及语义对齐检查"""
        hypothesis = "Model's answer violates physical laws or semantic understanding"
        evidence = []
        confidence = 0.5

        # 检查物理一致性
        ground_truth = question['ground_truth']
        parsed_answer = response['parsed_answer']

        # 检查方向是否合理
        if 'direction' in ground_truth:
            gt_direction = ground_truth['direction']
            response_text = response['raw_response'].lower()

            # 检查是否提到了相反的方向
            opposite_directions = {
                'left': 'right', 'right': 'left',
                'front': 'behind', 'behind': 'front'
            }

            for direction, opposite in opposite_directions.items():
                if direction in gt_direction.lower() and opposite in response_text:
                    evidence.append(f"Model mentioned opposite direction: {opposite}")
                    confidence = 0.8

        if evidence:
            conclusion = "Physical misalignment detected"
        else:
            conclusion = "No obvious physical violations"
            confidence = 0.3

        return CausalTraceStep(
            step_name="Physical Alignment Check",
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            confidence=confidence
        )

    def _step2_spatial_reconstruction(
        self,
        response: Dict,
        question: Dict,
        trajectory: Dict
    ) -> CausalTraceStep:
        """步骤 2: 空间位置重塑验证"""
        hypothesis = "Model failed to reconstruct spatial topology correctly"
        evidence = []
        confidence = 0.5

        # 检查空间推理
        ground_truth = question['ground_truth']

        if 'relative_position' in ground_truth:
            gt_pos = np.array(ground_truth['relative_position'])
            distance = np.linalg.norm(gt_pos)

            # 检查模型是否提到了距离
            response_text = response['raw_response'].lower()
            has_distance = any(word in response_text for word in ['meter', 'distance', 'far', 'close'])

            if not has_distance:
                evidence.append("Model did not mention distance")
                confidence = 0.7

            # 检查距离估计是否合理
            extracted_distance = response['parsed_answer'].get('extracted_distance')
            if extracted_distance is not None:
                error_ratio = abs(extracted_distance - distance) / max(distance, 0.1)
                if error_ratio > 0.5:
                    evidence.append(f"Distance estimation error: {error_ratio:.1%}")
                    confidence = 0.8

        if evidence:
            conclusion = "Spatial reconstruction error detected"
        else:
            conclusion = "Spatial topology appears correct"
            confidence = 0.3

        return CausalTraceStep(
            step_name="Spatial Reconstruction",
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            confidence=confidence
        )

    def _step3_fov_verification(
        self,
        response: Dict,
        question: Dict,
        trajectory: Dict
    ) -> CausalTraceStep:
        """步骤 3: 视场边界校验"""
        hypothesis = "Object was outside field of view"
        evidence = []
        confidence = 0.5

        # 检查是否是视场问题
        response_text = response['raw_response'].lower()

        # 检查模型是否提到看不见
        visibility_keywords = ['cannot see', 'not visible', 'out of view', 'behind']
        has_visibility_issue = any(kw in response_text for kw in visibility_keywords)

        if has_visibility_issue:
            evidence.append("Model mentioned visibility issues")
            confidence = 0.7

        # 检查时间间隔
        time_gap = question.get('time_gap', 0)
        if time_gap > 5:
            evidence.append(f"Large time gap: {time_gap} frames")
            confidence = 0.6

        if evidence:
            conclusion = "Possible FOV boundary issue"
        else:
            conclusion = "FOV appears adequate"
            confidence = 0.3

        return CausalTraceStep(
            step_name="FOV Verification",
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            confidence=confidence
        )

    def _step4_root_cause_classification(
        self,
        causal_trace: List[CausalTraceStep]
    ) -> tuple[ErrorType, str]:
        """步骤 4: 根因分类归纳"""
        # 基于前三步的置信度决定错误类型
        confidences = {step.step_name: step.confidence for step in causal_trace}

        max_confidence_step = max(confidences, key=confidences.get)

        if max_confidence_step == "Physical Alignment Check":
            return ErrorType.PHYSICAL_MISALIGNMENT, "Model violated physical laws or semantic understanding"
        elif max_confidence_step == "Spatial Reconstruction":
            return ErrorType.SPATIAL_TOPOLOGY_ERROR, "Model failed to reconstruct spatial topology"
        elif max_confidence_step == "FOV Verification":
            return ErrorType.FOV_BOUNDARY_ISSUE, "Object was outside field of view or occluded"
        else:
            return ErrorType.MEMORY_DECAY, "Model forgot spatial information over time"

    def save_analyses(
        self,
        analyses: List[ErrorAnalysis],
        output_filename: str = None
    ) -> str:
        """保存分析报告"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"error_analysis_{timestamp}.json"

        output_path = self.output_dir / output_filename

        analyses_dict = []
        for analysis in analyses:
            analysis_dict = asdict(analysis)
            # 转换 Enum
            analysis_dict['error_type'] = analysis.error_type.value
            analyses_dict.append(analysis_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analyses_dict, f, indent=2, ensure_ascii=False)

        print(f"\n[Saved] {len(analyses)} error analyses to {output_path}")

        return str(output_path)


def main():
    """主函数：运行错误分析"""
    inspector = InspectorAgent()

    # 示例路径
    responses_path = "outputs/answers/responses_claude-sonnet-4-6_20260526_120000.json"
    questions_path = "outputs/exams/exam_episode_001.json"
    trajectory_path = "outputs/trajectories/episode_001.json"

    # 检查文件是否存在
    for path in [responses_path, questions_path, trajectory_path]:
        if not Path(path).exists():
            print(f"[Error] File not found: {path}")
            print("Please run the previous steps first.")
            return

    try:
        analyses = inspector.analyze_errors(
            responses_path=responses_path,
            questions_path=questions_path,
            trajectory_path=trajectory_path
        )

        # 保存分析
        inspector.save_analyses(analyses)

        # 统计
        error_types = {}
        for analysis in analyses:
            error_type = analysis.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1

        print(f"\n[Summary] Total errors analyzed: {len(analyses)}")
        print("Error type distribution:")
        for error_type, count in error_types.items():
            print(f"  - {error_type}: {count}")

    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()