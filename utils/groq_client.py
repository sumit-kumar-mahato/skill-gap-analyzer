from groq import Groq

def groq_call(prompt: str, api_key: str) -> str:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, deterministic assistant. "
                    "Follow instructions exactly. "
                    "If JSON is requested, return ONLY valid JSON. "
                    "Do not add explanations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=2048
    )

    return response.choices[0].message.content
