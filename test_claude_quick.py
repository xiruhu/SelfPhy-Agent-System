"""
快速测试 Claude API 调用
强制使用 .env 文件中的配置
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 设置 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 清除系统环境变量，强制使用 .env
if 'ANTHROPIC_API_KEY' in os.environ:
    del os.environ['ANTHROPIC_API_KEY']
if 'ANTHROPIC_BASE_URL' in os.environ:
    del os.environ['ANTHROPIC_BASE_URL']

# 加载 .env 文件
load_dotenv(override=True)

print("=" * 80)
print("Claude API 快速测试")
print("=" * 80)

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

print(f"API Key: {api_key[:20]}..." if api_key else "未设置")
print(f"Base URL: {base_url}")
print()

if not api_key:
    print("错误: 未找到 ANTHROPIC_API_KEY")
    sys.exit(1)

# 测试调用
from core.claude_adapter import ClaudeAdapter

try:
    print("正在调用 Claude API...")
    adapter = ClaudeAdapter(api_key=api_key, base_url=base_url)

    result = adapter.call(
        prompt="请用中文回答：你是什么模型？",
        images=[],
        temperature=0.3
    )

    print("✓ 调用成功!")
    print(f"响应: {result['response']}")
    print(f"Token 数: {result['token_count']}")

except Exception as e:
    print(f"✗ 调用失败: {e}")
    import traceback
    traceback.print_exc()
