"""Live-provider configuration for adversarial evaluation.

Loads local .env configuration and creates the real model bundle.
No API keys are stored in code.
"""

import os

from dotenv import load_dotenv

from eval.adversarial.harness import Bundle
from eval.adversarial.providers import make_client


load_dotenv()


BUYER_PROVIDER = "openai"
BUYER_MODEL = "gpt-4.1-mini"

INTERPRETER_PROVIDER = "groq"
INTERPRETER_MODEL = "openai/gpt-oss-120b"

VENDOR_PROVIDER = "groq"
VENDOR_MODEL = "openai/gpt-oss-20b"


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"{name} is not set")


def make_live_bundle(seed: int) -> Bundle:
    """Create one real-model bundle for a matched trial."""

    _require_env("OPENAI_API_KEY")
    _require_env("GROQ_API_KEY")
    _require_env("OPENROUTER_API_KEY")

    buyer = make_client(
        BUYER_PROVIDER,
        BUYER_MODEL,
        seed=seed,
        temperature=0.0,
    )

    interpreter = make_client(
        INTERPRETER_PROVIDER,
        INTERPRETER_MODEL,
        seed=seed,
        temperature=0.0,
    )

    vendor = make_client(
        VENDOR_PROVIDER,
        VENDOR_MODEL,
        seed=seed,
        temperature=0.0,
        reasoning_effort="low",
    )

    attacker = make_live_attacker(seed)

    return Bundle(
    buyer=buyer,
        interpreter=interpreter,
        vendor=vendor,
        attacker=attacker,
        fingerprint={
            "buyer": buyer.fingerprint(),
            "interpreter": interpreter.fingerprint(),
            "vendor": vendor.fingerprint(),
            "attacker": attacker.fingerprint(),
        },
    )


def make_live_attacker(seed: int):
    """Create the OpenRouter attacker client for the future live run."""

    _require_env("OPENROUTER_API_KEY")

    return make_client(
        "openrouter",
        "openrouter/free",
        seed=seed,
        temperature=0.0,
    )