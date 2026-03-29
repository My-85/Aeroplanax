from anthropic import Anthropic

client = Anthropic(
    api_key="sk-a3f77579c5ac4f7c266eadbe8779d3b1d178cd08df168224df5f4a779fc197c5",
    base_url="https://ai.tokencloud.ai"
)

try:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("✓ API 调用成功!")
    print(f"Response: {response.content[0].text}")
except Exception as e:
    print(f"✗ API 调用失败: {e}")
