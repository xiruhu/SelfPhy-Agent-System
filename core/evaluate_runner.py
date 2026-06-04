"""
测试执行器 (Evaluation Runner)
负责派发问题给目标模型并收集回答
"""

import json
import os
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# 导入 Claude 适配器
from core.claude_adapter import ClaudeAdapter


@dataclass
class ModelResponse:
    """模型回答"""
    question_id: str
    model_name: str
    raw_response: str
    parsed_answer: Dict[str, Any]
    confidence: Optional[float] = None
    response_time: float = 0.0
    token_count: Optional[int] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ModelAdapter(ABC):
    """模型适配器基类"""

    @abstractmethod
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """
        调用模型 API

        Args:
            prompt: 文本提示
            images: 图片路径列表
            **kwargs: 其他参数

        Returns:
            包含 response, token_count 等的字典
        """
        pass


class OpenAIAdapter(ModelAdapter):
    """OpenAI / GPT-4o 适配器"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """调用 OpenAI API"""
        # 构造多模态消息
        content = []

        # 添加图片
        for img_path in images:
            if not Path(img_path).exists():
                print(f"[Warning] Image not found: {img_path}")
                continue

            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

        # 添加文本
        content.append({"type": "text", "text": prompt})

        # 调用 API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        return {
            "response": result['choices'][0]['message']['content'],
            "token_count": result.get('usage', {}).get('total_tokens', 0)
        }


class KimiAdapter(ModelAdapter):
    """Kimi (Moonshot) 适配器 — 支持多模态，OpenAI 兼容接口"""

    def __init__(self, api_key: str, model: str = "kimi-k2.5"):
        self.api_key  = api_key
        self.model    = os.getenv("MOONSHOT_MODEL", model)
        self.base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """调用 Kimi API（文本 + 可选图片）"""
        content: List[Dict] = []

        # 图片放在文本前面（Kimi 推荐顺序）
        for img_path in images:
            if not img_path or not Path(img_path).exists():
                print(f"  [WARN] 图片不存在，跳过: {img_path}")
                continue
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            # 根据扩展名判断 MIME
            suffix   = Path(img_path).suffix.lower()
            mime     = "image/png" if suffix == ".png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"}
            })

        content.append({"type": "text", "text": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=120
        )
        resp.raise_for_status()
        result = resp.json()

        return {
            "response":    result["choices"][0]["message"]["content"],
            "token_count": result.get("usage", {}).get("total_tokens", 0),
        }


class MiniMaxAdapter(ModelAdapter):
    """
    MiniMax (MiniMax-M2.7) 适配器 — 纯文本模型，不支持图片。
    图片参数会被忽略；exam_formatter 已生成纯文本版 prompt。
    """

    def __init__(self, api_key: str, model: str = "MiniMax-M2.7"):
        self.api_key  = api_key
        self.model    = os.getenv("MINIMAX_MODEL", model)
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.laozhang.ai/v1")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """调用 MiniMax API（纯文本，忽略 images 参数）"""
        if images:
            print("  [INFO] MiniMax 不支持图片，已忽略图片输入，使用纯文本 prompt。")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        result = resp.json()

        return {
            "response":    result["choices"][0]["message"]["content"],
            "token_count": result.get("usage", {}).get("total_tokens", 0),
        }


class DoubaoAdapter(ModelAdapter):
    """豆包 (Doubao-Vision) 适配器"""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """调用豆包 API"""
        # 豆包的 API 格式（需根据官方文档调整）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.3)
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        return {
            "response": result.get('response', result.get('output', '')),
            "token_count": result.get('token_count', 0)
        }


class EvaluationRunner:
    """测试执行器"""

    def __init__(self, config_path: str = "config/model_config.json"):
        self.config_path = Path(config_path)
        self.adapters: Dict[str, ModelAdapter] = {}
        self.output_dir = Path("outputs/answers")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载模型配置"""
        # 先加载 .env 文件
        from dotenv import load_dotenv
        load_dotenv(override=True)

        if not self.config_path.exists():
            print(f"[Warning] Config file not found: {self.config_path}")
            print("Using environment variables for API keys")
            self._load_from_env()
            return

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        # 初始化适配器
        for model_name, model_config in config.items():
            self._init_adapter(model_name, model_config)

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 加载 .env 文件
        from dotenv import load_dotenv
        load_dotenv(override=True)

        # Claude (Anthropic) - 优先使用
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
        if anthropic_key:
            self.adapters["claude-sonnet-4-6"] = ClaudeAdapter(anthropic_key, "claude-sonnet-4-6", anthropic_base_url)
            self.adapters["claude-opus-4"] = ClaudeAdapter(anthropic_key, "claude-opus-4", anthropic_base_url)
            self.adapters["claude-haiku-4"] = ClaudeAdapter(anthropic_key, "claude-haiku-4", anthropic_base_url)
            print(f"[Config] Loaded Claude adapters from .env (base_url: {anthropic_base_url})")

        # OpenAI / GPT-4o
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.adapters["gpt-4o-mini"] = OpenAIAdapter(openai_key, "gpt-4o-mini")
            self.adapters["gpt-4o"] = OpenAIAdapter(openai_key, "gpt-4o")

        # Kimi
        kimi_key = os.getenv("MOONSHOT_API_KEY")
        if kimi_key:
            self.adapters["kimi"] = KimiAdapter(kimi_key)

        # MiniMax
        minimax_key = os.getenv("MINIMAX_API_KEY")
        if minimax_key:
            self.adapters["minimax"] = MiniMaxAdapter(
                minimax_key
            )

    def _init_adapter(self, model_name: str, config: Dict[str, Any]):
        """初始化模型适配器"""
        adapter_type = config.get("type", "openai")

        # 替换环境变量占位符
        def replace_env_vars(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            return value

        # 处理所有配置项
        processed_config = {k: replace_env_vars(v) for k, v in config.items()}

        if adapter_type == "claude":
            api_key = processed_config.get("api_key")
            if not api_key or api_key.startswith("${"):
                print(f"[Warning] Skipping {model_name}: ANTHROPIC_API_KEY not set")
                return
            self.adapters[model_name] = ClaudeAdapter(
                api_key=api_key,
                model=processed_config.get("model", "claude-sonnet-4-6"),
                base_url=processed_config.get("base_url")
            )
        elif adapter_type == "openai":
            api_key = processed_config.get("api_key")
            if not api_key or api_key.startswith("${"):
                print(f"[Warning] Skipping {model_name}: OPENAI_API_KEY not set")
                return
            self.adapters[model_name] = OpenAIAdapter(
                api_key=api_key,
                model=processed_config.get("model", "gpt-4o-mini"),
                base_url=processed_config.get("base_url")
            )
        elif adapter_type == "kimi":
            api_key = processed_config.get("api_key")
            if not api_key or api_key.startswith("${"):
                print(f"[Warning] Skipping {model_name}: MOONSHOT_API_KEY not set")
                return
            self.adapters[model_name] = KimiAdapter(
                api_key=api_key,
                model=processed_config.get("model", "moonshot-v1-128k")
            )
        elif adapter_type == "minimax":
            api_key = processed_config.get("api_key")
            group_id = processed_config.get("group_id")
            if not api_key or api_key.startswith("${"):
                print(f"[Warning] Skipping {model_name}: MINIMAX_API_KEY not set")
                return
            self.adapters[model_name] = MiniMaxAdapter(
                api_key=api_key,
                model=processed_config.get("model")
            )
        elif adapter_type == "doubao":
            api_key = processed_config.get("api_key")
            endpoint = processed_config.get("endpoint")
            if not api_key or api_key.startswith("${"):
                print(f"[Warning] Skipping {model_name}: DOUBAO_API_KEY not set")
                return
            self.adapters[model_name] = DoubaoAdapter(
                api_key=api_key,
                endpoint=endpoint
            )

    def run_evaluation(
        self,
        questions_path: str,
        model_name: str,
        max_retries: int = 3,
        timeout: float = 60.0
    ) -> List[ModelResponse]:
        """
        执行评测

        Args:
            questions_path: 问题 JSON 文件路径
            model_name: 模型名称
            max_retries: 最大重试次数
            timeout: 超时时间（秒）

        Returns:
            模型响应列表
        """
        # 加载问题
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        if model_name not in self.adapters:
            raise ValueError(f"Model {model_name} not configured")

        adapter = self.adapters[model_name]
        responses = []

        print(f"\n[Evaluation] Model: {model_name}, Questions: {len(questions)}")

        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Processing {question['question_id']}...")

            try:
                start_time = time.time()

                # 调用模型
                result = adapter.call(
                    prompt=question['prompt'],
                    images=question['context_frames'],
                    temperature=0.3
                )

                response_time = time.time() - start_time

                # 解析答案
                parsed_answer = self._parse_answer(
                    result['response'],
                    question
                )

                # 构建响应
                response = ModelResponse(
                    question_id=question['question_id'],
                    model_name=model_name,
                    raw_response=result['response'],
                    parsed_answer=parsed_answer,
                    response_time=response_time,
                    token_count=result.get('token_count')
                )

                responses.append(response)

                print(f"[Success] Response time: {response_time:.2f}s")

            except Exception as e:
                print(f"[Error] Failed to process question: {e}")
                # 记录失败
                responses.append(ModelResponse(
                    question_id=question['question_id'],
                    model_name=model_name,
                    raw_response=f"ERROR: {str(e)}",
                    parsed_answer={"error": str(e)},
                    response_time=0.0
                ))

        return responses

    def _parse_answer(self, raw_response: str, question: Dict) -> Dict[str, Any]:
        """
        解析模型输出

        Args:
            raw_response: 模型原始输出
            question: 问题数据

        Returns:
            解析后的答案
        """
        # 简单的答案提取（可以用更复杂的解析逻辑）
        parsed = {
            "raw_text": raw_response,
            "extracted_answer": raw_response.strip()
        }

        # 尝试提取关键信息（根据问题类型）
        if "direction" in question.get('ground_truth', {}):
            # 提取方向信息
            direction_keywords = ["left", "right", "front", "behind", "forward", "backward"]
            found_directions = [kw for kw in direction_keywords if kw in raw_response.lower()]
            parsed["extracted_direction"] = found_directions

        if "distance" in question.get('ground_truth', {}):
            # 提取距离信息（简单正则）
            import re
            distance_match = re.search(r'(\d+\.?\d*)\s*(m|meter|metre)', raw_response.lower())
            if distance_match:
                parsed["extracted_distance"] = float(distance_match.group(1))

        return parsed

    def save_responses(
        self,
        responses: List[ModelResponse],
        output_filename: str = None
    ) -> str:
        """保存响应到文件"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"responses_{responses[0].model_name}_{timestamp}.json"

        output_path = self.output_dir / output_filename

        responses_dict = [asdict(r) for r in responses]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(responses_dict, f, indent=2, ensure_ascii=False)

        print(f"\n[Saved] {len(responses)} responses to {output_path}")

        return str(output_path)


def main():
    """主函数：运行评测"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="SelfPhy 被测模型评测执行器")
    parser.add_argument(
        "--model", type=str, default="kimi",
        choices=["kimi", "minimax", "doubao"],
        help="被测模型名称"
    )
    parser.add_argument(
        "--exam", type=str, default=None,
        help="试卷路径（默认自动推断 outputs/exams/exam_<model>.json）"
    )
    args = parser.parse_args()

    model_name    = args.model
    questions_path = args.exam or f"outputs/exams/exam_{model_name}.json"

    if not Path(questions_path).exists():
        print(f"[ERROR] 试卷文件不存在: {questions_path}")
        print("请先运行 exam_formatter.py 生成各模型试卷。")
        return

    runner = EvaluationRunner()

    if model_name not in runner.adapters:
        print(f"[ERROR] 模型 {model_name} 未配置，请检查 .env 文件。")
        print(f"已配置的模型: {list(runner.adapters.keys())}")
        return

    try:
        responses = runner.run_evaluation(
            questions_path=questions_path,
            model_name=model_name,
        )
        runner.save_responses(responses)

        total  = len(responses)
        errors = sum(1 for r in responses if "error" in r.parsed_answer)
        print(f"\n[汇总] 总题数: {total}  错误: {errors}  成功: {total - errors}")

    except Exception as e:
        print(f"[ERROR] 评测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()