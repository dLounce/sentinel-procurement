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
    def __init__(self, provider: str, model: str, seed=None, temperature=0.0, max_tokens=1024, reasoning_effort=None):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider = provider
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.last_system_fingerprint = None
        self.last_usage = None
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.usage_calls = 0
        self._cfg = PROVIDERS[provider]
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy: tests/offline runs never import this

            key = os.environ.get(self._cfg["key_env"])
            if not key:
                raise RuntimeError(f"{self._cfg['key_env']} is not set")
            self._client = OpenAI(api_key=key, base_url=self._cfg["base_url"],  timeout=45.0, max_retries=0,)
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

        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            if self.seed is None:
                raise
            kwargs.pop("seed", None)
            response = client.chat.completions.create(**kwargs)
        self.last_system_fingerprint = getattr(response, "system_fingerprint", None)
        usage = getattr(response, "usage", None)
        self.last_usage = (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            if usage is not None
            else None
        )

        if self.last_usage is not None:
            for key in self.total_usage:
                value = self.last_usage.get(key)
                if isinstance(value, int):
                    self.total_usage[key] += value
            self.usage_calls += 1

        return response.choices[0].message.content

    def fingerprint(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "system_fingerprint": self.last_system_fingerprint,
            "deterministic": False,
            "last_usage": self.last_usage,
            "total_usage": self.total_usage,
            "usage_calls": self.usage_calls,
        }


def make_client(provider: str, model: str, seed=None, temperature=0.0, reasoning_effort=None):
    return OpenAICompatibleClient(provider, model, seed=seed, temperature=temperature, reasoning_effort=reasoning_effort,)
