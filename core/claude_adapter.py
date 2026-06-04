"""
Claude API 适配器
用于调用 Anthropic Claude 模型
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Any
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential


class ClaudeAdapter:
    """Claude (Anthropic) 适配器"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-6",
        base_url: str = None
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=base_url
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """
        调用 Claude API

        Args:
            prompt: 文本提示
            images: 图片路径列表
            **kwargs: 其他参数

        Returns:
            包含 response, token_count 等的字典
        """
        # 构造多模态消息
        content = []

        # 添加图片
        for img_path in images:
            if not Path(img_path).exists():
                print(f"[Warning] Image not found: {img_path}")
                continue

            # 读取图片并编码
            with open(img_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.standard_b64encode(img_data).decode("utf-8")

            # 检测图片格式
            img_suffix = Path(img_path).suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(img_suffix, 'image/jpeg')

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_base64,
                },
            })

        # 添加文本
        content.append({
            "type": "text",
            "text": prompt
        })

        # 调用 API
        message = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.3),
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        # 提取响应文本
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        return {
            "response": response_text,
            "token_count": message.usage.input_tokens + message.usage.output_tokens,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens
        }


def main():
    """测试 Claude 适配器"""
    adapter = ClaudeAdapter()

    # 测试简单调用
    result = adapter.call(
        prompt="Hello! Please respond with 'OK' if you can see this message.",
        images=[],
        temperature=0.3
    )

    print("Response:", result['response'])
    print("Token count:", result['token_count'])


if __name__ == "__main__":
    main()
