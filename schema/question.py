"""
Question Schema V2
------------------
第二代考题数据模型：纯问题 + 多模态证据
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime


class Pose6DoF(BaseModel):
    """6自由度位姿"""
    frame_id: int
    position: List[float]
    rotation: List[float]
    euler_angles: Optional[List[float]] = None
    timestamp: float


class QuestionV2(BaseModel):
    """第二代考题格式：纯问题，无场景描述"""
    question_id: str
    question_text: str
    ground_truth_answer: str
    capability: Literal["egocentric_memory", "spatial_transformation", "occlusion_reasoning", "trajectory_backtracking", "distance_estimation"]
    evidence_frame_ids: List[int]
    trajectory_window: List[Pose6DoF]
    depth_frame_ids: Optional[List[int]] = None
    reasoning_trace: str
    difficulty: Literal["easy", "medium", "hard"]
    spatial_transform_type: Optional[Literal["rotation_only", "translation_only", "rotation_translation", "complex_trajectory"]] = None
    rotation_degree: Optional[float] = None
    displacement_meters: Optional[float] = None
    expected_error_patterns: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MultimodalEvidence(BaseModel):
    """多模态证据数据"""
    rgb_frames: Dict[str, str]
    depth_frames: Optional[Dict[str, str]] = None
    trajectory: List[Pose6DoF]
    scene_id: str
    episode_id: str


class ExamPaperV2(BaseModel):
    """第二代考卷：纯问题 + 多模态证据"""
    exam_id: str
    trajectory_id: str
    scene_id: str
    episode_id: str
    questions: List[QuestionV2]
    multimodal_evidence: MultimodalEvidence
    created_at: datetime = Field(default_factory=datetime.now)
    supervisor_model: str = "claude-sonnet-4-6"
    total_questions: int
    difficulty_distribution: Dict[str, int]
    capability_distribution: Dict[str, int]
    statistics: Optional[Dict[str, Any]] = None


class ModelInputV2(BaseModel):
    """被测模型的输入格式"""
    question_id: str
    question_text: str
    frames: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    has_depth: bool = False


class EvaluationResultV2(BaseModel):
    """第二代评测结果"""
    exam_id: str
    model_name: str
    responses: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
