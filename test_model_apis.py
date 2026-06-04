"""
测试所有模型 API 连接
验证 Claude, Kimi, MiniMax 的接口是否正确配置
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 设置控制台编码为 UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.claude_adapter import ClaudeAdapter
from core.evaluate_runner import KimiAdapter, MiniMaxAdapter


def test_claude():
    """测试 Claude API"""
    print("\n" + "="*60)
    print("测试 Claude API")
    print("="*60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    if not api_key:
        print("❌ ANTHROPIC_API_KEY 未设置")
        return False

    print(f"✓ API Key: {api_key[:20]}...")
    print(f"✓ Base URL: {base_url or '默认'}")

    try:
        adapter = ClaudeAdapter(
            api_key=api_key,
            model="claude-sonnet-4-6",
            base_url=base_url
        )

        result = adapter.call(
            prompt="请用中文回复'测试成功'",
            images=[],
            temperature=0.3
        )

        print(f"✓ 响应: {result['response'][:100]}")
        print(f"✓ Token 数: {result['token_count']}")
        print("✅ Claude API 测试通过")
        return True

    except Exception as e:
        print(f"❌ Claude API 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kimi():
    """测试 Kimi API"""
    print("\n" + "="*60)
    print("测试 Kimi (Moonshot) API")
    print("="*60)

    api_key = os.getenv("MOONSHOT_API_KEY")
    base_url = os.getenv("MOONSHOT_BASE_URL")
    model = os.getenv("MOONSHOT_MODEL", "kimi-k2.5")

    if not api_key:
        print("❌ MOONSHOT_API_KEY 未设置")
        return False

    print(f"✓ API Key: {api_key[:20]}...")
    print(f"✓ Base URL: {base_url or '默认'}")
    print(f"✓ Model: {model}")

    try:
        adapter = KimiAdapter(api_key=api_key, model=model)

        result = adapter.call(
            prompt="请用中文回复'测试成功'",
            images=[],
            temperature=0.3
        )

        print(f"✓ 响应: {result['response'][:100]}")
        print(f"✓ Token 数: {result['token_count']}")
        print("✅ Kimi API 测试通过")
        return True

    except Exception as e:
        print(f"❌ Kimi API 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimax():
    """测试 MiniMax API"""
    print("\n" + "="*60)
    print("测试 MiniMax API")
    print("="*60)

    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

    if not api_key:
        print("❌ MINIMAX_API_KEY 未设置")
        return False

    print(f"✓ API Key: {api_key[:20]}...")
    print(f"✓ Base URL: {base_url or '默认'}")
    print(f"✓ Model: {model}")

    try:
        adapter = MiniMaxAdapter(api_key=api_key, model=model)

        result = adapter.call(
            prompt="请用中文回复'测试成功'",
            images=[],
            temperature=0.3
        )

        print(f"✓ 响应: {result['response'][:100]}")
        print(f"✓ Token 数: {result['token_count']}")
        print("✅ MiniMax API 测试通过")
        return True

    except Exception as e:
        print(f"❌ MiniMax API 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("SelfPhy-Agent-System 模型 API 连接测试")
    print("="*60)

    # 加载环境变量
    load_dotenv(override=True)

    results = {
        "Claude": test_claude(),
        "Kimi": test_kimi(),
        "MiniMax": test_minimax()
    }

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for model, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 个模型测试通过")

    if passed == total:
        print("\n🎉 所有模型 API 配置正确！")
    else:
        print("\n⚠️  部分模型 API 配置有问题，请检查 .env 文件")


if __name__ == "__main__":
    main()
