
import os
import requests
from requests.exceptions import RequestException


class OllamaService:
    def __init__(
        self,
        host="http://localhost:11434",
        model="qwen3:4b-instruct-2507-q4_K_M",
        timeout=240,
        provider="ollama",
        api_key=None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if self.provider == "groq" and not self.api_key:
            raise ValueError("GROQ_API_KEY is missing.")

    def health_check(self):
        if self.provider == "groq":
            try:
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )
                r.raise_for_status()
                return True, f"Groq API connected. Model: {self.model}"
            except RequestException as e:
                return False, f"Groq connection failed: {e}"

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=8)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]

            if self.model not in names and not any(
                n.split(":")[0] == self.model.split(":")[0] for n in names
            ):
                return False, f"Model '{self.model}' is not listed."

            return True, f"Ollama is running. Model: {self.model}"

        except Exception as e:
            return False, f"Ollama connection failed: {e}"

    def list_models(self):
        if self.provider == "groq":
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]

        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m.get("name") for m in r.json().get("models", [])]

    def generate(self, prompt, system=None, temperature=0.1):
        if self.provider == "groq":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.chat(messages, temperature)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        if system:
            payload["system"] = system

        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            answer = r.json().get("response", "").strip()

            if not answer:
                raise RuntimeError("Ollama returned an empty response.")

            return answer

        except requests.Timeout as e:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s."
            ) from e
        except RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def chat(self, messages, temperature=0.1):
        if self.provider == "groq":
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                answer = r.json()["choices"][0]["message"]["content"]
                if not answer:
                    raise RuntimeError("Groq returned an empty response.")
                return answer.strip()

            except requests.Timeout as e:
                raise RuntimeError("Groq request timed out.") from e
            except RequestException as e:
                detail = e.response.text if e.response is not None else str(e)
                raise RuntimeError(f"Groq request failed: {detail}") from e

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }

        r = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
