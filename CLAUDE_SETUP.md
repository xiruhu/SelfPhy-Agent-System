# Claude API 配置说明

## 当前配置

| 项目 | 值 |
|------|----|
| API 提供商 | api.v3.cm（第三方代理） |
| 长官模型 | `claude-sonnet-4-6` |
| Base URL | `https://api.v3.cm`（不带 `/v1` 后缀） |

`.env` 关键字段：
```
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_BASE_URL=https://api.v3.cm
```

## 已解决的历史问题

- Base URL 末尾不能带 `/v1`，SDK 会自动拼接
- `python-dotenv` 需要在代码最早处调用 `load_dotenv()`，否则系统环境变量会覆盖 `.env`
- `config/model_config.json` 中的 `${VAR}` 占位符需要在 `evaluate_runner.py` 里手动替换

## 快速验证

```bash
python test_claude_quick.py
```

## 核心文件

| 文件 | 用途 |
|------|------|
| `core/claude_adapter.py` | Claude API 适配器（长官模型调用入口） |
| `core/claude_examiner.py` | 考试生成器（调用 claude-sonnet-4-6 出题） |
| `core/claude_reflector.py` | 错题反思器（四步排除法） |
| `config/model_config.json` | 所有模型的 API 配置 |
| `.env` | API Key 和 Base URL（不提交 git） |
