"""Provider-agnostic model adapters.

Every adapter is a callable ``(prompt: str) -> str`` matching the injected-model
interface used by the production Interpreter, Buyer, and Vendor. API keys are read
ONLY from environment variables and never written to files or source. OpenAI,
Groq, and OpenRouter all expose an OpenAI-compatible chat-completions API, so one
implementation covers all three via a per-provider base URL.

Hosted-LLM calls are NOT bit-deterministic even with a seed; the adapter passes a
seed where supported and records the returned system fingerprint so runs are
auditable. We never claim perfect determinism.
"""

import os

PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
}


class OpenAICompatibleClient:
    def __init__(self, provider: str, model: str, seed=None, temperature=0.0, max_tokens=1024):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider = provider
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_system_fingerprint = None
        self._cfg = PROVIDERS[provider]
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy: tests/offline runs never import this

            key = os.environ.get(self._cfg["key_env"])
            if not key:
                raise RuntimeError(f"{self._cfg['key_env']} is not set")
            self._client = OpenAI(api_key=key, base_url=self._cfg["base_url"])
        return self._client

    def __call__(self, prompt: str) -> str:
        client = self._ensure()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = client.chat.completions.create(**kwargs)
        self.last_system_fingerprint = getattr(response, "system_fingerprint", None)
        return response.choices[0].message.content

    def fingerprint(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "temperature": self.temperature,
            "system_fingerprint": self.last_system_fingerprint,
            "deterministic": False,
        }


def make_client(provider: str, model: str, seed=None, temperature=0.0):
    return OpenAICompatibleClient(provider, model, seed=seed, temperature=temperature)
