"""
model_adapters.py
-----------------
统一的多模态模型适配器接口

设计原则：
1. 每个模型一个适配器类，继承自 BaseAdapter
2. 统一的输入格式：MultimodalInput
3. 统一的输出格式：ModelResponse
4. 自动处理图像编码、API 重试、错误处理
"""

import os
import time
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MultimodalFrame:
    """单帧多模态数据"""
    frame_id: int
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    rgb_base64: Optional[str] = None
    depth_base64: Optional[str] = None


@dataclass
class PoseData:
    """位姿数据"""
    frame_id: int
    position: List[float]
    rotation: List[float]
    euler_angles: Optional[List[float]] = None
    timestamp: float = 0.0


@dataclass
class MultimodalInput:
    """统一的多模态输入"""
    question_text: str
    frames: List[MultimodalFrame]
    trajectory: List[PoseData]
    system_prompt: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass
class ModelResponse:
    """统一的模型响应"""
    answer: str
    response_time_ms: int
    model_name: str
    raw_response: Optional[Dict] = None
    error: Optional[str] = None
    token_usage: Optional[Dict] = None


class BaseAdapter(ABC):
    """模型适配器基类"""

    def __init__(self):
        self.model_name = self.__class__.__name__.replace("Adapter", "")

    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def prepare_frames(self, frames: List[MultimodalFrame]) -> List[MultimodalFrame]:
        """准备帧数据（自动编码图像）"""
        prepared_frames = []

        for frame in frames:
            prepared_frame = MultimodalFrame(
                frame_id=frame.frame_id,
                rgb_path=frame.rgb_path,
                depth_path=frame.depth_path
            )

            # 编码 RGB
            if frame.rgb_path and Path(frame.rgb_path).exists():
                prepared_frame.rgb_base64 = self.encode_image(frame.rgb_path)
            elif frame.rgb_base64:
                prepared_frame.rgb_base64 = frame.rgb_base64

            # 编码 Depth
            if frame.depth_path and Path(frame.depth_path).exists():
                prepared_frame.depth_base64 = self.encode_image(frame.depth_path)
            elif frame.depth_base64:
                prepared_frame.depth_base64 = frame.depth_base64

            prepared_frames.append(prepared_frame)

        return prepared_frames

    @abstractmethod
    def call_api(self, input_data: MultimodalInput) -> ModelResponse:
        """调用模型 API（子类实现）"""
        pass

    def __call__(self, input_data: MultimodalInput, retry: int = 2) -> ModelResponse:
        """
        调用模型（带重试）

        Args:
            input_data: 多模态输入
            retry: 重试次数

        Returns:
            ModelResponse
        """
        # 准备帧数据
        input_data.frames = self.prepare_frames(input_data.frames)

        # 尝试调用 API
        for attempt in range(retry + 1):
            try:
                response = self.call_api(input_data)

                if response.error is None:
                    return response

                # 如果有错误但还有重试机会
                if attempt < retry:
                    time.sleep(1)  # 等待 1 秒后重试
                    continue

                return response

            except Exception as e:
                if attempt < retry:
                    time.sleep(1)
                    continue

                # 最后一次尝试失败
                return ModelResponse(
                    answer="",
                    response_time_ms=0,
                    model_name=self.model_name,
                    error=f"API call failed after {retry + 1} attempts: {str(e)}"
                )


class KimiAdapter(BaseAdapter):
    """Kimi (Moonshot) 适配器"""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = os.getenv("MOONSHOT_MODEL", "moonshot-v1-128k")

        if not self.api_key:
            raise ValueError("未找到 MOONSHOT_API_KEY，请检查 .env 文件")

    def call_api(self, input_data: MultimodalInput) -> ModelResponse:
        """调用 Kimi API"""
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 构建消息内容
        message_content = []

        # 系统提示
        if input_data.system_prompt:
            message_content.append({
                "type": "text",
                "text": input_data.system_prompt
            })

        # 问题
        message_content.append({
            "type": "text",
            "text": f"问题：{input_data.question_text}\n\n请基于以下第一人称视觉轨迹回答。"
        })

        # 图像帧
        for frame in input_data.frames:
            if frame.rgb_base64:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame.rgb_base64}"
                    }
                })

        # 轨迹数据
        trajectory_text = "\n【轨迹数据】\n"
        for pose in input_data.trajectory:
            trajectory_text += (
                f"Frame {pose.frame_id}: "
                f"位置={[round(p, 2) for p in pose.position]}, "
                f"旋转={[round(r, 2) for r in pose.rotation]}\n"
            )

        message_content.append({
            "type": "text",
            "text": trajectory_text
        })

        # 调用 API
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message_content}],
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            return ModelResponse(
                answer=response.choices[0].message.content,
                response_time_ms=response_time_ms,
                model_name=self.model,
                raw_response=response.model_dump(),
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            )

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return ModelResponse(
                answer="",
                response_time_ms=response_time_ms,
                model_name=self.model,
                error=str(e)
            )


class MiniMaxAdapter(BaseAdapter):
    """MiniMax 适配器"""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
        self.model = os.getenv("MINIMAX_MODEL", "abab6.5-chat")

        if not self.api_key:
            raise ValueError("未找到 MINIMAX_API_KEY，请检查 .env 文件")

    def call_api(self, input_data: MultimodalInput) -> ModelResponse:
        """调用 MiniMax API"""
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 构建消息（类似 Kimi）
        message_content = []

        if input_data.system_prompt:
            message_content.append({
                "type": "text",
                "text": input_data.system_prompt
            })

        message_content.append({
            "type": "text",
            "text": f"问题：{input_data.question_text}\n\n请基于以下视觉证据回答。"
        })

        # 图像
        for frame in input_data.frames:
            if frame.rgb_base64:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame.rgb_base64}"
                    }
                })

        # 轨迹
        trajectory_text = "\n【轨迹信息】\n"
        for pose in input_data.trajectory:
            trajectory_text += f"帧 {pose.frame_id}: 位置 {pose.position}\n"

        message_content.append({
            "type": "text",
            "text": trajectory_text
        })

        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message_content}],
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            return ModelResponse(
                answer=response.choices[0].message.content,
                response_time_ms=response_time_ms,
                model_name=self.model,
                raw_response=response.model_dump(),
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            )

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return ModelResponse(
                answer="",
                response_time_ms=response_time_ms,
                model_name=self.model,
                error=str(e)
            )


class DoubaoAdapter(BaseAdapter):
    """豆包 (Doubao) 适配器"""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DOUBAO_API_KEY")
        self.endpoint = os.getenv("DOUBAO_ENDPOINT")

        if not self.api_key or not self.endpoint:
            raise ValueError("未找到 DOUBAO_API_KEY 或 DOUBAO_ENDPOINT")

    def call_api(self, input_data: MultimodalInput) -> ModelResponse:
        """调用豆包 API（待实现具体接口）"""
        # TODO: 实现豆包的实际 API 调用
        return ModelResponse(
            answer="豆包多模态 API 尚未实现",
            response_time_ms=0,
            model_name="doubao",
            error="Not implemented"
        )


# 适配器工厂
def get_adapter(model_name: str) -> BaseAdapter:
    """
    获取模型适配器

    Args:
        model_name: 模型名称 (kimi/minimax/doubao)

    Returns:
        BaseAdapter 实例
    """
    adapters = {
        "kimi": KimiAdapter,
        "moonshot": KimiAdapter,
        "minimax": MiniMaxAdapter,
        "doubao": DoubaoAdapter
    }

    model_name_lower = model_name.lower()

    if model_name_lower not in adapters:
        raise ValueError(f"未知模型: {model_name}. 支持的模型: {list(adapters.keys())}")

    return adapters[model_name_lower]()
