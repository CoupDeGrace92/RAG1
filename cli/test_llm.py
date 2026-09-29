import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages: list[ChatCompletionUserMessageParam] = [
    {
        "role": "user",
        "content": "Why is Bootdev such a great place to learn about RAG? Use one paragraph maximum."
    }
]

response = client.chat.completions.create(
    model="openrouter/free", 
    messages = messages
    )

print(response.choices[0].message.content)
if response.usage:
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Response tokens: {response.usage.completion_tokens}")