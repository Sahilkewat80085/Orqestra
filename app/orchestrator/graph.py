"""
app/orchestrator/graph.py
══════════════════════════
LangGraph state machine for Orqestra.

Architecture:
  - State = serialized SharedContext dict
  - Entry point = orchestrator node (sets routing_decision)
  - All subsequent edges are CONDITIONAL — driven by routing_decision
  - No static A → B → C chains anywhere

Node execution order is determined at runtime by the orchestrator's plan.
Agents that are not selected are never invoked.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.synthesis import SynthesisAgent
from app.orchestrator.context_budget import ContextBudgetManager
from app.orchestrator.router import OrchestratorRouter
from app.schemas.context import ExecutionStatus, SharedContext

# Type alias for the LangGraph state dict
GraphState = Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(state: GraphState) -> SharedContext:
    """Deserialize SharedContext from LangGraph state dict."""
    return SharedContext.model_validate(state["context"])


def _dump(ctx: SharedContext) -> GraphState:
    """Serialize SharedContext back into LangGraph state dict."""
    return {"context": ctx.model_dump(mode="json")}


# ── Node Functions ─────────────────────────────────────────────────────────────

async def orchestrator_node(state: GraphState) -> GraphState:
    """Entry point: routes the query dynamically."""
    ctx = _load(state)
    ctx = ctx.mark_timestamp("started").model_copy(
        update={"status": ExecutionStatus.RUNNING}
    )

    # Initialize token budgets
    budget_mgr = ContextBudgetManager(ctx)
    ctx = budget_mgr.initialize_budgets()

    # Produce routing decision
    router = OrchestratorRouter()
    ctx = await router.route(ctx)
    return _dump(ctx)


async def decomposition_node(state: GraphState) -> GraphState:
    ctx = _load(state)
    agent = DecompositionAgent()
    ctx = await agent.invoke(ctx)
    return _dump(ctx)


async def retrieval_node(state: GraphState) -> GraphState:
    ctx = _load(state)
    agent = RetrievalAgent()
    ctx = await agent.invoke(ctx)
    return _dump(ctx)


async def critique_node(state: GraphState) -> GraphState:
    ctx = _load(state)
    agent = CritiqueAgent()
    ctx = await agent.invoke(ctx)
    return _dump(ctx)


async def synthesis_node(state: GraphState) -> GraphState:
    ctx = _load(state)
    agent = SynthesisAgent()
    ctx = await agent.invoke(ctx)
    ctx = ctx.mark_timestamp("completed").model_copy(
        update={"status": ExecutionStatus.COMPLETED}
    )
    return _dump(ctx)


# ── Conditional Edge Functions ────────────────────────────────────────────────

def _get_plan_agents(state: GraphState) -> list[str]:
    """Extract ordered agent names from the routing decision."""
    ctx = _load(state)
    if not ctx.routing_decision:
        return ["synthesis"]
    plan = ctx.routing_decision.get("execution_plan", [])
    # Sort by priority and return agent names
    sorted_steps = sorted(plan, key=lambda s: s.get("priority", 99))
    return [s["agent"] for s in sorted_steps]


def route_after_orchestrator(state: GraphState) -> str:
    """
    Dynamic routing: returns the first agent in the execution plan.
    The graph's conditional edges map each possible first-agent name
    to the correct node.
    """
    agents = _get_plan_agents(state)
    if not agents:
        return "synthesis"
    first = agents[0]
    # Map to valid node names
    valid_nodes = {"decomposition", "retrieval", "critique", "synthesis"}
    return first if first in valid_nodes else "synthesis"


def route_after_decomposition(state: GraphState) -> str:
    agents = _get_plan_agents(state)
    # Find what comes after decomposition
    if "retrieval" in agents:
        return "retrieval"
    if "critique" in agents:
        return "critique"
    return "synthesis"


def route_after_retrieval(state: GraphState) -> str:
    agents = _get_plan_agents(state)
    if "critique" in agents:
        return "critique"
    return "synthesis"


def route_after_critique(state: GraphState) -> str:
    return "synthesis"


# ── Graph Construction ────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct and compile the LangGraph state machine.

    The graph has conditional edges everywhere — no edge is static.
    The orchestrator's routing_decision drives all transitions.
    """
    builder = StateGraph(dict)  # State is a plain dict wrapping SharedContext

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("decomposition", decomposition_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("critique", critique_node)
    builder.add_node("synthesis", synthesis_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")

    # ── Dynamic routing from orchestrator ─────────────────────────────────────
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "decomposition": "decomposition",
            "retrieval": "retrieval",
            "critique": "critique",
            "synthesis": "synthesis",
        },
    )

    # ── Conditional transitions between agents ─────────────────────────────────
    builder.add_conditional_edges(
        "decomposition",
        route_after_decomposition,
        {
            "retrieval": "retrieval",
            "critique": "critique",
            "synthesis": "synthesis",
        },
    )

    builder.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "critique": "critique",
            "synthesis": "synthesis",
        },
    )

    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"synthesis": "synthesis"},
    )

    # Synthesis always terminates
    builder.add_edge("synthesis", END)

    return builder.compile()


# Module-level compiled graph (import and use directly)
graph = build_graph()
