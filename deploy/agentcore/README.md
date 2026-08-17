# AgentCore deployment — UNDEPLOYED / NOT YET VERIFIED

Everything in `deploy/agentcore/` is **configuration and code to be run on an
AWS-enabled machine**. Nothing here has been deployed or verified on AWS. A file
existing here is **not** evidence of any AWS security property. Run the procedure
below and collect the real evidence before claiming anything.

## Topology (the deployed M8 architecture)

```
Vendor model (local driver, private reservation clamp)
  -> raw vendor text
  -> Interpreter Runtime (LLM)      -> validated VendorOffer
  -> Buyer Runtime (price guard + LangGraph Buyer LLM -> BuyerDecision
                    -> deterministic authorization -> gated place_order)
```

- **Interpreter Runtime** (`interpreter_runtime.py`): reads untrusted vendor text,
  returns only a schema-validated `VendorOffer`. No tools, no `place_order`, no
  Buyer access. `vendor_id` is stamped from the trusted transport argument.
- **Buyer Runtime** (`buyer_runtime.py`): re-validates every offer at the boundary
  (rejecting raw text), runs the price guard, runs the LangGraph Buyer (LLM) to get
  a `BuyerDecision`, and calls the gated `place_order` only for an accept that
  clears the deterministic gate. The Buyer LLM never calls `place_order`.
- **Driver** (`driver.py`): runs locally (NOT a runtime). Hosts the Vendors and
  their private reservation prices, and invokes the two runtimes over the AWS SDK.
  It is the only identity holding `bedrock-agentcore:InvokeAgentRuntime`.

## Verified vs TBD (AgentCore interface)

Verified against current AWS docs (starter toolkit quickstart + IAM permissions
reference, fetched during preparation):

- CLI: `pip install bedrock-agentcore bedrock-agentcore-starter-toolkit boto3`;
  `agentcore configure -e <file> -r <region> --execution-role <ARN> --disable-memory`;
  `agentcore deploy`; `agentcore invoke`; `agentcore status`; `agentcore destroy`.
- Entrypoint contract: `BedrockAgentCoreApp()` + `@app.entrypoint def invoke(payload)`
  + `app.run()` (serves `/invocations` on 8080).
- Programmatic invoke: `boto3.client("bedrock-agentcore").invoke_agent_runtime(
  agentRuntimeArn=…, runtimeSessionId=…, payload=…, qualifier="DEFAULT")`.
- IAM: execution-role policy (logs/xray/metrics/workload-token + `bedrock:InvokeModel`)
  and the deploy caller policy — see `iam/`. Execution roles have **no**
  `InvokeAgentRuntime` (the isolation lever). Trust principal
  `bedrock-agentcore.amazonaws.com`.

**Model id (resolved — no env-var dependency):** the runtimes read the model id
from the `INTERPRETER_MODEL_ID` / `BUYER_MODEL_ID` environment variable **if set**,
otherwise from the invocation **payload** supplied by the trusted driver
(`model_id` / `buyer_model_id`). So a deployment can always provide the model id
even if you do not use the runtime env-var mechanism. The driver forwards the ids
it reads from its own environment. A model id is a model name, not a secret.

**Packaging (locally verified for the repo-tree layout):** the entrypoints add the
repo root to `sys.path` from `__file__`, and this was verified locally by running
each entrypoint as a script from a foreign working directory with only the repo
tree present — the app imports (`agents`, `guards`, `rfq`, `deploy.agentcore.*`)
resolve. **Requirement:** run `agentcore configure` **from the repo root** so the
whole repo tree is packaged (the entrypoint stays at `deploy/agentcore/…`).
Confirm the packaged contents with `agentcore deploy --dry-run` (or inspect the
CodeZip) before the real deploy.

**TBD (confirm on your machine; do not assume):**
- Concrete **Bedrock model ids** — set to models access-enabled in your region.
- **Alternative CLI:** the npm `@aws/agentcore` CLI (`agentcore create --framework
  LangChain_LangGraph`, CDK-based) is also current; it scaffolds a different project
  layout, so adapting these entrypoints to it is TBD.
- Live network reachability between runtimes is AgentCore-managed; it is verified
  at the IAM layer here (see isolation section) and marked UNVERIFIED UNTIL
  DEPLOYMENT at the network layer.

## Isolation: is "no InvokeAgentRuntime" sufficient?

The Interpreter runtime cannot reach the Buyer's privileged action surface because
**all** of the following hold, not just the single IAM action:

- Its execution role has **no** `bedrock-agentcore:InvokeAgentRuntime` (verified by
  `iam:SimulatePrincipalPolicy` in `verify_isolation.py`).
- The two runtimes use **separate execution roles** — no shared role or credentials.
- The Interpreter runtime is never given the **Buyer runtime ARN** or any Buyer
  credential/OAuth token in its environment or payload, so it has no address or
  token to reach the Buyer even over HTTPS Inbound Auth.
- `place_order` is **internal deterministic code** inside the Buyer runtime, not a
  separately addressable endpoint/tool — there is no action surface to reach.
- No Lambda / API Gateway / queue / shared service is deployed (M7 scope is exactly
  the two runtimes + two roles), so there is no indirect path.

The IAM half is verifiable now (simulate). Network-layer non-reachability is
AgentCore-managed and is marked **UNVERIFIED UNTIL DEPLOYMENT**.

## Deployment procedure (run on your AWS-enabled machine)

```bash
# 1-2. clone the exact commit and verify git state
git clone https://github.com/dLounce/sentinel-procurement.git && cd sentinel-procurement
git log --oneline -1            # confirm the intended M7 commit

# 3-6. verify AWS readiness (do NOT paste secrets anywhere)
aws sts get-caller-identity
aws configure get region
aws bedrock list-foundation-models --query "modelSummaries[?contains(modelId,'claude')].modelId"
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit boto3 && agentcore --help

# 7. create the two least-privilege execution roles from iam/ (substitute placeholders),
#    each with iam/trust_policy.json as its trust relationship. Neither gets InvokeAgentRuntime.

# 8. deploy the Interpreter FIRST (isolation proven before the Buyer exists)
cp deploy/agentcore/requirements-interpreter.txt requirements.txt
agentcore configure -e deploy/agentcore/interpreter_runtime.py -r <REGION> \
  --execution-role <INTERPRETER_EXECUTION_ROLE_ARN> --name interpreter --disable-memory
agentcore deploy         # record the Interpreter runtime ARN
# set INTERPRETER_MODEL_ID on the runtime (mechanism TBD — see above)

# 9. verify Interpreter isolation (see below)

# 10. deploy the Buyer
cp deploy/agentcore/requirements-buyer.txt requirements.txt
agentcore configure -e deploy/agentcore/buyer_runtime.py -r <REGION> \
  --execution-role <BUYER_EXECUTION_ROLE_ARN> --name buyer --disable-memory
agentcore deploy         # record the Buyer runtime ARN
# set BUYER_MODEL_ID on the runtime

# 11-14. connect + run end-to-end (driver, local)
export INTERPRETER_RUNTIME_ARN=... BUYER_RUNTIME_ARN=... VENDOR_MODEL_ID=...
python deploy/agentcore/driver.py

# isolation / security verification (actual tests)
export INTERPRETER_EXECUTION_ROLE_ARN=...
python deploy/agentcore/verify_isolation.py
```

## Isolation verification (actual authorization/behaviour tests)

`verify_isolation.py` performs, and prints PASS/FAIL/UNVERIFIED for, each of:

1. **Interpreter role cannot invoke the Buyer runtime** — `iam:SimulatePrincipalPolicy`
   on `bedrock-agentcore:InvokeAgentRuntime` against the Buyer ARN → expect *not*
   `allowed`. (Effective-permission evaluation, not JSON inspection.)
2. **Buyer rejects raw vendor text** — invoke Buyer with a raw string offer → rejected.
3. **Buyer blocks a below-floor offer** — guards active → no order.
4. **Interpreter returns only a VendorOffer** for an injected message → no action fields.

Do not convert "the config looks right" into "verified." Only a printed PASS from a
real run is evidence, and only for that one property.

## Rollback / cleanup

```bash
agentcore destroy        # per runtime project (starter toolkit)
# or, npm CLI: agentcore remove all && agentcore deploy
# then delete the two execution roles if no longer needed
```

## Security notes

- No secrets in any file here. Model ids come from environment variables; AWS
  credentials come from the runtime execution role / local caller chain.
- `place_order` is internal deterministic code (no real payment rail, per project
  non-goals); the Buyer runtime needs no payment/marketplace permissions.
- The M6 scripted evaluation baseline (Metric A = 0, Metric B = 1) is unchanged.
  Any live-model run must be recorded as a SEPARATE, clearly-labelled result.
- The delivery-value limitation is unchanged and NOT fixed here: a well-formed but
  false `delivery_days` can still reach an anomalous in-budget order. The delivery
  channel is not claimed secure.
