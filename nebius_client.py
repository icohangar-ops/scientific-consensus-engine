"""Nebius Token Factory client wrapper for Scientific Consensus Engine."""

import os
import time
from typing import Optional

from openai import OpenAI

from cubiczan_resilience import resilient

# --- Datadog LLM Observability (no-op unless DD_LLMOBS_ENABLED) ---
from observability import init_observability
from prism_observability import build_prism_telemetry, trace_llm as trace_prism_llm

init_observability("scientific-consensus-engine")

_PRISM = build_prism_telemetry()

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
ORCHESTRATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEBATE_MODEL = "deepseek-ai/DeepSeek-V3.2"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable must be set")
        _client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key, timeout=60)
    return _client


class NebiusAgent:
    """Nebius-powered agent with structured JSON and optional tool calling."""

    def __init__(self, model: str = ORCHESTRATOR_MODEL, system_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt
        self.client = get_client()

    @resilient(timeout=60, max_attempts=3)
    def chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> dict:
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = getattr(response, "usage", None)
        token_count_input = int(getattr(usage, "prompt_tokens", 0) or 0)
        token_count_output = int(getattr(usage, "completion_tokens", 0) or 0)

        result = {"content": message.content, "role": message.role}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "type": tc.type,
                }
                for tc in message.tool_calls
            ]
        trace_prism_llm(
            _PRISM,
            model=self.model,
            input_messages=full_messages,
            output=message.content or "",
            latency_ms=latency_ms,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            metadata={
                "source": "scientific-consensus-engine",
                "operation": "chat",
                "has_tool_calls": bool(message.tool_calls),
            },
        )
        return result

    @resilient(timeout=60, max_attempts=3)
    def embed(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = getattr(response, "usage", None)
        token_count_input = int(getattr(usage, "prompt_tokens", 0) or 0)
        token_count_output = int(getattr(usage, "completion_tokens", 0) or 0)
        trace_prism_llm(
            _PRISM,
            model=EMBEDDING_MODEL,
            input_messages=[{"role": "user", "content": "\n\n".join(texts[:3])}],
            output=f"{len(response.data)} embeddings",
            latency_ms=latency_ms,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            metadata={
                "source": "scientific-consensus-engine",
                "operation": "embed",
                "input_count": len(texts),
            },
        )
        return [item.embedding for item in response.data]
