使用 API 密钥

将以下环境变量添加到您的终端配置文件或直接在终端中运行。


Claude Code

OpenCode

macOS / Linux

Windows CMD

PowerShell
Terminal
复制
export ANTHROPIC_BASE_URL="https://ai.tokencloud.ai"
export ANTHROPIC_AUTH_TOKEN="sk-bf907cb732da4390f4755aac9a25e00ca58abcf5e65cb830a0a64d162a788f68"
VSCode Claude Code

~/.claude/settings.json
复制
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://ai.tokencloud.ai",
    "ANTHROPIC_AUTH_TOKEN": "sk-bf907cb732da4390f4755aac9a25e00ca58abcf5e65cb830a0a64d162a788f68",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  }
}
这些环境变量将在当前终端会话中生效。如需永久配置，请将其添加到 ~/.bashrc、~/.zshrc 或相应的配置文件中。