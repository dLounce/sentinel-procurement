"""Interpreter AgentCore Runtime entrypoint.

UNDEPLOYED / NOT YET VERIFIED.

Quarantined runtime: reads one untrusted vendor message and returns exactly one
schema-validated VendorOffer (or an error marker). It holds no privileged tools,
no place_order, and no Buyer access. vendor_id is stamped from the trusted
transport argument, not the model output. Reservation prices and evaluation
ground truth never exist in this runtime.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402

from agents.interpreter import ExtractionError, make_interpreter  # noqa: E402
from deploy.agentcore.bedrock_model import make_bedrock_model  # noqa: E402
from guards.schema_validate import SchemaValidationError  # noqa: E402

app = BedrockAgentCoreApp()
_interpreters = {}


def _interpreter_for(model_id):
    if model_id not in _interpreters:
        _interpreters[model_id] = make_interpreter(make_bedrock_model(model_id))
    return _interpreters[model_id]


@app.entrypoint
def invoke(payload):
    model_id = os.environ.get("INTERPRETER_MODEL_ID") or payload.get("model_id")
    if not model_id:
        return {"error": "model_id_not_configured"}
    try:
        offer = _interpreter_for(model_id)(
            payload["raw_text"],
            vendor_id=payload["vendor_id"],
            quantity=int(payload["quantity"]),
        )
    except (SchemaValidationError, ExtractionError) as exc:
        return {"error": type(exc).__name__}
    return {"offer": offer}


if __name__ == "__main__":
    app.run()
