"""PRISMtrace wiring for Scientific Consensus Engine."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from prismtrace import PRISMtrace
from prismtrace._config import resolve_host


@dataclass(slots=True)
class PrismTelemetry:
    client: PRISMtrace
    trace_id: str
    agent_name: str = "Scientific Consensus Engine"
    agent_id: str = "scientific-consensus-engine"


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def build_prism_telemetry(agent_name: str = "Scientific Consensus Engine") -> PrismTelemetry | None:
    api_key = _env("PRISMTRACE_API_KEY")
    project_id = _env("PRISMTRACE_PROJECT_ID")
    if not (api_key and project_id):
        return None

    host = resolve_host(_env("PRISMTRACE_HOST"), _env("PRISMTRACE_ENDPOINT"))
    client = PRISMtrace(api_key=api_key, host=host, project_id=project_id)
    return PrismTelemetry(
        client=client,
        trace_id=str(uuid.uuid4()),
        agent_name=agent_name,
    )


def trace_llm(
    telemetry: PrismTelemetry | None,
    *,
    model: str,
    input_messages: list,
    output: str,
    latency_ms: int,
    token_count_input: int = 0,
    token_count_output: int = 0,
    metadata: dict | None = None,
) -> None:
    if telemetry is None:
        return
    telemetry.client.trace_llm(
        model=model,
        input_messages=input_messages,
        output=output,
        latency_ms=latency_ms,
        token_count_input=token_count_input,
        token_count_output=token_count_output,
        trace_id=telemetry.trace_id,
        agent_id=telemetry.agent_id,
        agent_name=telemetry.agent_name,
        metadata=metadata,
    )
