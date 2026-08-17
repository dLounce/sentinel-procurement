"""Bedrock model client for the deployed runtimes.

UNDEPLOYED / NOT YET VERIFIED. Produces the injected ``model(prompt) -> str``
callable that the Interpreter and Buyer use in production. Credentials come from
the runtime's AgentCore execution role (the default boto3 chain); NO API keys or
secrets are read from source.

The model id is passed explicitly (never hard-coded). The runtimes resolve it from
an environment variable if set, otherwise from the invocation payload supplied by
the trusted caller — so a deployment can always provide the model id without
depending on a runtime env-var mechanism that the current starter-toolkit docs do
not pin down. A model id is a model name, not a secret.
"""

import boto3

_client = None


def _runtime_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime")
    return _client


def make_bedrock_model(model_id: str):
    if not model_id:
        raise ValueError("model_id is required")

    def model(prompt: str) -> str:
        response = _runtime_client().converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 1024},
        )
        return response["output"]["message"]["content"][0]["text"]

    return model
