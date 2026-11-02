
import ollama

response = ollama.chat(
    model="qwen3:4b-instruct-2507-q4_K_M",
    messages=[
        {
            "role": "user",
            "content": "Explain what an Applicant Tracking System does in five points."
        }
    ]
)

print(response["message"]["content"])
