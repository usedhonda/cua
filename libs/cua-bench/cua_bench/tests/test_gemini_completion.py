"""Offline completion-loop tests; runnable directly without SDK dependencies."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def load_agent():
    # Load the real agent and base under an isolated package name, avoiding the
    # SDK's top-level imports and optional desktop/provider dependencies.
    package = ModuleType("_gemini_completion_tests")
    package.__path__ = []
    package.register_agent = lambda name: lambda cls: cls
    modules = {package.__name__: package}
    agents_dir = Path(__file__).resolve().parents[1] / "agents"
    with patch.dict(sys.modules, modules):
        for name in ("base", "gemini"):
            module_name = f"{package.__name__}.{name}"
            spec = importlib.util.spec_from_file_location(module_name, agents_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    return modules["gemini"].GeminiAgent, modules["base"].FailureMode


GeminiAgent, FailureMode = load_agent()


class FakePart(SimpleNamespace):
    @classmethod
    def from_bytes(cls, **kwargs):
        return cls(**kwargs)


def response(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


class TestGeminiCompletion(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, responses, max_steps=2):
        generate = Mock(side_effect=responses)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        genai = SimpleNamespace(Client=Mock(return_value=client))
        types = SimpleNamespace(
            Part=FakePart,
            Content=SimpleNamespace,
            Tool=SimpleNamespace,
            GenerateContentConfig=SimpleNamespace,
            FunctionResponse=SimpleNamespace,
        )
        agent = GeminiAgent(thinking_level=None, media_resolution=None, max_steps=max_steps)
        agent._lazy_import_genai = Mock(return_value=(genai, types))
        agent._build_custom_function_declarations = Mock(return_value=[])
        agent._map_function_call_to_action = AsyncMock(return_value=object())
        session = SimpleNamespace(screenshot=AsyncMock(return_value=b"fake screenshot"))
        result = await agent.perform_task("Finish the task", session)
        return result, generate, agent._map_function_call_to_action

    async def test_mentions_do_not_stop_remaining_actions(self):
        for text in ("NOT DONE yet", "Type UNDONE into the field", "The next step is DONE later"):
            with self.subTest(text=text):
                result, generate, execute = await self.run_agent(
                    [
                        response(
                            FakePart(text=text),
                            FakePart(
                                function_call=SimpleNamespace(
                                    name="type_text", args={"text": "UNDONE"}
                                )
                            ),
                        ),
                        response(FakePart(text="DONE")),
                    ]
                )
                self.assertEqual(generate.call_count, 2)
                execute.assert_awaited_once_with("type_text", {"text": "UNDONE"}, unittest.mock.ANY)
                self.assertEqual(result.failure_mode, FailureMode.NONE)

    async def test_mentions_without_actions_do_not_report_completion(self):
        result, _, _ = await self.run_agent([response(FakePart(text="NOT DONE"))])
        self.assertEqual(result.failure_mode, FailureMode.UNKNOWN)

    async def test_thinking_done_does_not_stop_remaining_actions(self):
        result, generate, _ = await self.run_agent(
            [
                response(
                    FakePart(text="DONE", thought=True),
                    FakePart(function_call=SimpleNamespace(name="click", args={"x": 1, "y": 2})),
                ),
                response(FakePart(text="DONE")),
            ]
        )
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.failure_mode, FailureMode.NONE)

    async def test_standalone_done_completes(self):
        result, generate, execute = await self.run_agent([response(FakePart(text=" \nDONE\n "))])
        self.assertEqual(result.failure_mode, FailureMode.NONE)
        self.assertEqual(generate.call_count, 1)
        execute.assert_not_awaited()

    async def test_done_tool_completes(self):
        result, generate, execute = await self.run_agent(
            [response(FakePart(function_call=SimpleNamespace(name="done", args={})))]
        )
        self.assertEqual(result.failure_mode, FailureMode.NONE)
        self.assertEqual(generate.call_count, 1)
        execute.assert_not_awaited()

    async def test_unfinished_action_at_limit_reports_max_steps(self):
        result, _, execute = await self.run_agent(
            [
                response(
                    FakePart(
                        text="NOT DONE",
                        function_call=SimpleNamespace(name="click", args={"x": 1, "y": 2}),
                    )
                )
            ],
            max_steps=1,
        )
        self.assertEqual(result.failure_mode, FailureMode.MAX_STEPS_EXCEEDED)
        self.assertEqual(execute.await_count, 1)


if __name__ == "__main__":
    unittest.main()
