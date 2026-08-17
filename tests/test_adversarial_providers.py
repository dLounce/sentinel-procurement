from eval.adversarial.providers import OpenAICompatibleClient


class FakeCompletions:
    def create(self, **kwargs):
        if "seed" in kwargs:
            raise TypeError("seed is not supported")

        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message})()
        return type(
            "Response",
            (),
            {
                "system_fingerprint": "test-fingerprint",
                "choices": [choice],
            },
        )()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_provider_retries_without_seed_when_provider_rejects_it():
    client = OpenAICompatibleClient("openai", "test-model", seed=123)
    client._client = FakeClient()

    assert client("hello") == "ok"
    assert client.last_system_fingerprint == "test-fingerprint"
