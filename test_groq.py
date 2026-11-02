import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Reply with: DocuMind Groq connection successful!"}
    ],
    temperature=0.1,
)

print(response.choices[0].message.content)