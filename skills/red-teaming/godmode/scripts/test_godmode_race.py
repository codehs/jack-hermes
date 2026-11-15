"""Offline request-contract tests. No SDK, credentials, or network required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import godmode_race


class CapturingClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Test response."))]
        )


class RequestContractTests(unittest.TestCase):
    def test_sampling_is_omitted_only_for_replacement_models(self):
        for model in (
            "google/gemini-3.5-flash-lite",
            "google/gemini-3.8-flash",
            "google/gemini-2.5-flash-lite",
            "google/gemini-2.5-flash-image",
            "google/gemma-3-27b-it",
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "deepseek/deepseek-chat",
        ):
            with self.subTest(model=model):
                client = CapturingClient()
                messages = [{"role": "user", "content": "Say hello."}]
                result = godmode_race._query_model(client, model, messages, timeout=17)
                expected = dict(model=model, messages=messages, max_tokens=4096, timeout=17)
                if model not in (
                    "google/gemini-3.5-flash-lite", "google/gemini-3.8-flash"
                ):
                    expected["temperature"] = 0.7
                self.assertEqual(client.requests, [expected])
                self.assertEqual(result[0:2], (model, "Test response."))
                self.assertIsNone(result[3])

    def test_standard_selection_and_final_user_with_optional_prefill(self):
        for append_directive in (False, True):
            for prefill in (None, [{"role": "assistant", "content": "Earlier reply."}]):
                with self.subTest(append_directive=append_directive, prefill=prefill):
                    client = CapturingClient()
                    with patch.object(godmode_race, "OpenAI", return_value=client) as factory:
                        godmode_race.race_models(
                            "Say hello.", api_key="offline-test", system_prompt="Be concise.",
                            prefill=prefill, append_directive=append_directive, max_workers=1,
                        )
                    factory.assert_called_once_with(
                        api_key="offline-test", base_url="https://openrouter.ai/api/v1"
                    )
                    models = [request["model"] for request in client.requests]
                    self.assertEqual(len(models), 24)
                    self.assertEqual(models[0], "google/gemini-3.5-flash-lite")
                    self.assertEqual(models[15], "google/gemini-3.8-flash")
                    self.assertNotIn("google/gemini-2.5-flash", models)
                    self.assertNotIn("google/gemini-2.5-pro", models)
                    expected = [{"role": "system", "content": "Be concise."}]
                    expected.extend(prefill or [])
                    expected.append({
                        "role": "user",
                        "content": "Say hello." + (
                            godmode_race.DEPTH_DIRECTIVE if append_directive else ""
                        ),
                    })
                    for request in client.requests:
                        self.assertEqual(request["messages"], expected)

    def test_classic_selection_uses_replacement_and_final_user(self):
        client = CapturingClient()
        with patch.object(godmode_race, "OpenAI", return_value=client):
            result = godmode_race.race_godmode_classic("Say hello.", api_key="offline-test")
        google_requests = [
            request for request in client.requests if request["model"].startswith("google/")
        ]
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(len(google_requests), 1)
        self.assertEqual(google_requests[0]["model"], "google/gemini-3.5-flash-lite")
        self.assertNotIn("temperature", google_requests[0])
        self.assertEqual(google_requests[0]["messages"][-1], {
            "role": "user", "content": "Z=Say hello."
        })
        google_results = [r for r in result["all_results"] if r["model"].startswith("google/")]
        self.assertEqual(google_results[0]["codename"], "GEMINI 3.5 FLASH LITE")
        for request in client.requests:
            if not request["model"].startswith("google/"):
                self.assertEqual(request["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
