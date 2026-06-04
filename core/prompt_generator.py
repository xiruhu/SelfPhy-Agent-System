"""
考试生成器 (Exam Generator)
负责根据轨迹自动生成空间推理考题
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class QuestionType(str, Enum):
    """问题类型"""
    RECALL = "recall"  # 空间记忆回忆
    REASONING = "reasoning"  # 空间推理
    COUNTERFACTUAL = "counterfactual"  # 反事实推理
    ADVERSARIAL = "adversarial"  # 对抗式探测


@dataclass
class Question:
    """生成的考题"""
    question_id: str
    question_type: QuestionType
    prompt: str
    context_frames: List[str]  # 图片路径列表
    ground_truth: Dict[str, Any]
    difficulty: Literal["easy", "medium", "hard"]
    spatial_level: Literal["room", "area", "object", "attribute"]
    time_gap: Optional[int] = None  # 距离最后一帧的间隔（帧数）
    is_adversarial: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ExamGenerator:
    """考试生成器"""

    def __init__(self, output_dir: str = "outputs/exams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_questions(
        self,
        trajectory_path: str,
        num_questions: int = 10,
        difficulty_distribution: Dict[str, float] = None,
        question_type_distribution: Dict[QuestionType, float] = None,
        enable_adversarial: bool = False
    ) -> List[Question]:
        """
        自动生成考题

        Args:
            trajectory_path: 轨迹 JSON 文件路径
            num_questions: 生成题目数量
            difficulty_distribution: 难度分布
            question_type_distribution: 题型分布
            enable_adversarial: 是否启用对抗样本

        Returns:
            问题列表
        """
        # 加载轨迹数据
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)

        # 默认分布
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        if question_type_distribution is None:
            question_type_distribution = {
                QuestionType.RECALL: 0.4,
                QuestionType.REASONING: 0.4,
                QuestionType.COUNTERFACTUAL: 0.1,
                QuestionType.ADVERSARIAL: 0.1 if enable_adversarial else 0.0
            }

        questions = []
        keyframes = trajectory['keyframes']

        if len(keyframes) < 2:
            print("[Warning] Not enough keyframes to generate questions")
            return questions

        for i in range(num_questions):
            # 随机选择题型
            q_type = self._sample_from_distribution(question_type_distribution)

            # 随机选择难度
            difficulty = self._sample_from_distribution(difficulty_distribution)

            # 根据题型生成问题
            if q_type == QuestionType.RECALL:
                question = self._generate_recall_question(
                    keyframes, trajectory, i, difficulty
                )
            elif q_type == QuestionType.REASONING:
                question = self._generate_reasoning_question(
                    keyframes, trajectory, i, difficulty
                )
            elif q_type == QuestionType.COUNTERFACTUAL:
                question = self._generate_counterfactual_question(
                    keyframes, trajectory, i, difficulty
                )
            elif q_type == QuestionType.ADVERSARIAL:
                question = self._generate_adversarial_question(
                    keyframes, trajectory, i, difficulty
                )
            else:
                continue

            if question:
                questions.append(question)

        return questions

    def _sample_from_distribution(self, distribution: Dict) -> Any:
        """从分布中采样"""
        items = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _generate_recall_question(
        self,
        keyframes: List[Dict],
        trajectory: Dict,
        question_idx: int,
        difficulty: str
    ) -> Optional[Question]:
        """生成空间记忆回忆题"""
        if len(keyframes) < 3:
            return None

        # 选择一个早期关键帧作为"记忆点"
        if difficulty == "easy":
            memory_idx = len(keyframes) - 2  # 倒数第二帧
            time_gap = 1
        elif difficulty == "medium":
            memory_idx = len(keyframes) // 2  # 中间帧
            time_gap = len(keyframes) - memory_idx
        else:  # hard
            memory_idx = min(2, len(keyframes) - 2)  # 早期帧
            time_gap = len(keyframes) - memory_idx

        memory_frame = keyframes[memory_idx]
        current_frame = keyframes[-1]

        # 构造问题
        prompt = f"""You are navigating through an environment.

At frame {memory_frame['frame_id']}, you observed the scene shown in the first image.

After that, you {self._describe_movement(keyframes[memory_idx:])}.

Now you are at frame {current_frame['frame_id']} (shown in the last image).

Question: Based on your memory, what was the approximate position of the camera at frame {memory_frame['frame_id']} relative to your current position?

Please answer with a direction (e.g., "behind and to the left", "in front and to the right") and an approximate distance."""

        # 计算相对位置
        memory_pos = np.array(memory_frame['pose']['position'])
        current_pos = np.array(current_frame['pose']['position'])
        relative_pos = memory_pos - current_pos

        # 判断方向
        direction = self._position_to_direction(relative_pos)
        distance = np.linalg.norm(relative_pos)

        ground_truth = {
            "direction": direction,
            "distance_meters": round(distance, 1),
            "relative_position": relative_pos.tolist()
        }

        question = Question(
            question_id=f"Q{question_idx:03d}_recall",
            question_type=QuestionType.RECALL,
            prompt=prompt,
            context_frames=[memory_frame['image_path'], current_frame['image_path']],
            ground_truth=ground_truth,
            difficulty=difficulty,
            spatial_level="area",
            time_gap=time_gap,
            metadata={"memory_frame_id": memory_frame['frame_id']}
        )

        return question

    def _generate_reasoning_question(
        self,
        keyframes: List[Dict],
        trajectory: Dict,
        question_idx: int,
        difficulty: str
    ) -> Optional[Question]:
        """生成空间推理题"""
        if len(keyframes) < 3:
            return None

        # 选择三个关键帧：起点、中间点、终点
        start_idx = 0
        mid_idx = len(keyframes) // 2
        end_idx = len(keyframes) - 1

        start_frame = keyframes[start_idx]
        mid_frame = keyframes[mid_idx]
        end_frame = keyframes[end_idx]

        # 构造问题
        prompt = f"""You are navigating through an environment.

Image 1: Your starting position (frame {start_frame['frame_id']})
Image 2: An intermediate position (frame {mid_frame['frame_id']})
Image 3: Your current position (frame {end_frame['frame_id']})

Your movement: {trajectory['spatial_narrative']}

Question: If you were to walk directly from your starting position (frame {start_frame['frame_id']}) to your current position (frame {end_frame['frame_id']}) in a straight line, approximately how far would you travel?

Also, what is the general direction from start to end?"""

        # 计算直线距离
        start_pos = np.array(start_frame['pose']['position'])
        end_pos = np.array(end_frame['pose']['position'])
        straight_distance = np.linalg.norm(end_pos - start_pos)
        direction = self._position_to_direction(end_pos - start_pos)

        ground_truth = {
            "straight_line_distance": round(straight_distance, 1),
            "direction": direction,
            "start_position": start_pos.tolist(),
            "end_position": end_pos.tolist()
        }

        question = Question(
            question_id=f"Q{question_idx:03d}_reasoning",
            question_type=QuestionType.REASONING,
            prompt=prompt,
            context_frames=[
                start_frame['image_path'],
                mid_frame['image_path'],
                end_frame['image_path']
            ],
            ground_truth=ground_truth,
            difficulty=difficulty,
            spatial_level="area",
            metadata={"requires_spatial_integration": True}
        )

        return question

    def _generate_counterfactual_question(
        self,
        keyframes: List[Dict],
        trajectory: Dict,
        question_idx: int,
        difficulty: str
    ) -> Optional[Question]:
        """生成反事实推理题"""
        if len(keyframes) < 2:
            return None

        # 选择一个转折点
        turn_idx = len(keyframes) // 2
        turn_frame = keyframes[turn_idx]
        end_frame = keyframes[-1]

        # 获取转折时的旋转角度
        turn_euler = turn_frame['pose']['euler_angles']
        yaw_change = turn_euler[2] if turn_euler else 0

        prompt = f"""You are navigating through an environment.

At frame {turn_frame['frame_id']} (shown in image 1), you made a turn.

You then continued moving and reached frame {end_frame['frame_id']} (shown in image 2).

Counterfactual Question: If you had NOT made that turn at frame {turn_frame['frame_id']}, but instead continued straight, where would you approximately be now relative to your actual current position?

Please describe the direction and approximate distance."""

        # 计算反事实位置（简化：假设继续直行）
        turn_pos = np.array(turn_frame['pose']['position'])
        end_pos = np.array(end_frame['pose']['position'])

        # 估算如果不转弯会在哪里
        displacement = end_pos - turn_pos
        distance_traveled = np.linalg.norm(displacement)

        # 假设原方向（简化）
        counterfactual_direction = self._position_to_direction(displacement)

        ground_truth = {
            "counterfactual_scenario": "continued_straight",
            "actual_turn_angle": round(yaw_change, 1),
            "estimated_offset": "would be in a different direction",
            "reasoning": "The turn changed the trajectory"
        }

        question = Question(
            question_id=f"Q{question_idx:03d}_counterfactual",
            question_type=QuestionType.COUNTERFACTUAL,
            prompt=prompt,
            context_frames=[turn_frame['image_path'], end_frame['image_path']],
            ground_truth=ground_truth,
            difficulty=difficulty,
            spatial_level="area",
            metadata={"turn_frame_id": turn_frame['frame_id']}
        )

        return question

    def _generate_adversarial_question(
        self,
        keyframes: List[Dict],
        trajectory: Dict,
        question_idx: int,
        difficulty: str
    ) -> Optional[Question]:
        """生成对抗式探测题"""
        # TODO: 实现对抗样本生成（需要图像编辑）
        # 这里返回一个占位问题
        return None

    def _describe_movement(self, keyframes: List[Dict]) -> str:
        """描述一段轨迹的运动"""
        if len(keyframes) < 2:
            return "remained stationary"

        movements = []
        for i in range(1, len(keyframes)):
            prev_frame = keyframes[i-1]
            curr_frame = keyframes[i]

            action = curr_frame['pose'].get('action_label', 'moved')
            movements.append(action.replace('_', ' '))

        return ", then ".join(movements)

    def _position_to_direction(self, relative_pos: np.ndarray) -> str:
        """将相对位置向量转换为方向描述"""
        x, y, z = relative_pos

        # 主要看 x 和 z（假设 y 是高度）
        direction_parts = []

        # 前后方向
        if abs(z) > 0.5:
            direction_parts.append("front" if z > 0 else "behind")

        # 左右方向
        if abs(x) > 0.5:
            direction_parts.append("right" if x > 0 else "left")

        if not direction_parts:
            return "approximately at the same location"

        return " and to the ".join(direction_parts)

    def save_questions(
        self,
        questions: List[Question],
        output_filename: str = "exam.json"
    ) -> str:
        """保存问题到文件"""
        output_path = self.output_dir / output_filename

        questions_dict = [
            {
                "question_id": q.question_id,
                "question_type": q.question_type.value,
                "prompt": q.prompt,
                "context_frames": q.context_frames,
                "ground_truth": q.ground_truth,
                "difficulty": q.difficulty,
                "spatial_level": q.spatial_level,
                "time_gap": q.time_gap,
                "is_adversarial": q.is_adversarial,
                "metadata": q.metadata
            }
            for q in questions
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_dict, f, indent=2, ensure_ascii=False)

        print(f"[Saved] {len(questions)} questions to {output_path}")

        return str(output_path)


def main():
    """主函数：生成示例考题"""
    generator = ExamGenerator()

    # 示例：从轨迹生成考题
    trajectory_path = "outputs/trajectories/episode_001.json"

    if not Path(trajectory_path).exists():
        print(f"[Error] Trajectory file not found: {trajectory_path}")
        print("Please run cv2_processor.py first to generate trajectory data.")
        return

    try:
        questions = generator.generate_questions(
            trajectory_path=trajectory_path,
            num_questions=10,
            enable_adversarial=False
        )

        # 保存问题
        generator.save_questions(questions, "exam_episode_001.json")

        print(f"\n[Success] Generated {len(questions)} questions")
        for q in questions[:3]:  # 显示前3个问题
            print(f"\n{q.question_id} ({q.difficulty}):")
            print(q.prompt[:200] + "...")

    except Exception as e:
        print(f"[Error] Failed to generate questions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
