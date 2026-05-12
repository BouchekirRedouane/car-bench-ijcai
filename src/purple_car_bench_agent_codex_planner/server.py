"""Server entry point for the Codex planner/executor CAR-bench purple agent."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

if __package__:
    from .planner_agent import (
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_REASONING_EFFORT,
        PlannerExecutorCARBenchAgentExecutor,
    )
else:
    from planner_agent import (
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_REASONING_EFFORT,
        PlannerExecutorCARBenchAgentExecutor,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="agent", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    """Return an environment value, treating unset and blank the same."""
    return os.getenv(name) or default


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the Codex planner/executor purple agent."""
    card = AgentCard(
        name="car_bench_agent_codex_planner",
        description="In-car voice assistant using private Codex planning and Spark execution",
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant (Codex Planner/Executor)"
    skill.description = "Privately plans with Codex and returns CAR-bench A2A text or tool calls"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "codex", "planner"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Codex planner/executor purple agent."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help=f"Planner model. Defaults to CODEX_PLANNER_MODEL or {DEFAULT_PLANNER_MODEL}.",
    )
    parser.add_argument(
        "--executor-model",
        type=str,
        default=None,
        help=(
            "Executor model. Defaults to CODEX_EXECUTOR_MODEL, CODEX_MODEL, "
            f"or {DEFAULT_EXECUTOR_MODEL}."
        ),
    )
    parser.add_argument(
        "--planner-reasoning-effort",
        type=str,
        default=None,
        help=(
            "Planner reasoning effort. Defaults to CODEX_PLANNER_REASONING_EFFORT "
            f"or {DEFAULT_PLANNER_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--executor-reasoning-effort",
        type=str,
        default=None,
        help=(
            "Executor reasoning effort. Defaults to CODEX_EXECUTOR_REASONING_EFFORT, "
            f"CODEX_REASONING_EFFORT, or {DEFAULT_EXECUTOR_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Codex turn timeout. Defaults to CODEX_TIMEOUT_SECONDS or 180.",
    )
    parser.add_argument(
        "--malformed-retries",
        type=int,
        default=None,
        help="Retry budget for malformed Codex JSON. Defaults to CODEX_MALFORMED_RETRIES or 1.",
    )
    args = parser.parse_args()

    planner_model = (
        args.planner_model
        if args.planner_model is not None
        else _env_or_default("CODEX_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
    )
    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default(
            "CODEX_EXECUTOR_MODEL",
            _env_or_default("CODEX_MODEL", DEFAULT_EXECUTOR_MODEL),
        )
    )
    planner_reasoning_effort = (
        args.planner_reasoning_effort
        if args.planner_reasoning_effort is not None
        else _env_or_default(
            "CODEX_PLANNER_REASONING_EFFORT",
            DEFAULT_PLANNER_REASONING_EFFORT,
        )
    )
    executor_reasoning_effort = (
        args.executor_reasoning_effort
        if args.executor_reasoning_effort is not None
        else _env_or_default(
            "CODEX_EXECUTOR_REASONING_EFFORT",
            _env_or_default("CODEX_REASONING_EFFORT", DEFAULT_EXECUTOR_REASONING_EFFORT),
        )
    )
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else float(_env_or_default("CODEX_TIMEOUT_SECONDS", "180"))
    )
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else int(_env_or_default("CODEX_MALFORMED_RETRIES", "1"))
    )

    logger.info(
        "Starting CAR-bench agent (Codex planner/executor)",
        planner_model=planner_model,
        executor_model=executor_model,
        planner_reasoning_effort=planner_reasoning_effort,
        executor_reasoning_effort=executor_reasoning_effort,
        timeout_seconds=timeout_seconds,
        malformed_retries=malformed_retries,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=PlannerExecutorCARBenchAgentExecutor(
            planner_model=planner_model,
            executor_model=executor_model,
            planner_reasoning_effort=planner_reasoning_effort,
            executor_reasoning_effort=executor_reasoning_effort,
            timeout_seconds=timeout_seconds,
            malformed_retries=malformed_retries,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)
    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
