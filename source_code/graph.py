from langgraph.graph import StateGraph, END
from source_code.state import AgentState
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent
from source_code.agents.agent_2_field_cleaner import make_field_cleaner_agent
from source_code.agents.executor import rename_executor_agent, cleaning_executor_agent
from source_code.reclassify import reclassify_columns_node   # [AUDIT M8] top-level, not agents/


def build_graph(config: PipelineConfig, checkpointer=None):
    """
    Build and compile the DS Machine LangGraph workflow.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration. Injected into agent factory closures.
        All LLM construction happens at factory call time (graph compile time).

    Returns
    -------
    CompiledStateGraph
        Ready-to-stream graph. Call .stream(initial_input) or .invoke(initial_input).

    Graph topology:
        agent1_renamer -> executor1 -> reclassify_columns -> agent2_cleaner -> executor2 -> END

    Node descriptions:
    - agent1_renamer    : make_field_renamer_agent(config)  — LLM call, produces column_map
    - executor1         : rename_executor_agent             — applies composite rename map
    - reclassify_columns: reclassify_columns_node           — re-classifies columns post-rename
    - agent2_cleaner    : make_field_cleaner_agent(config)  — LLM call, produces cleaning_code
    - executor2         : cleaning_executor_agent           — runs cleaning code
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent1_renamer",     make_field_renamer_agent(config))
    workflow.add_node("executor1",          rename_executor_agent)
    workflow.add_node("reclassify_columns", reclassify_columns_node)
    workflow.add_node("agent2_cleaner",     make_field_cleaner_agent(config))
    workflow.add_node("executor2",          cleaning_executor_agent)

    workflow.set_entry_point("agent1_renamer")
    workflow.add_edge("agent1_renamer",     "executor1")
    workflow.add_edge("executor1",          "reclassify_columns")
    workflow.add_edge("reclassify_columns", "agent2_cleaner")
    workflow.add_edge("agent2_cleaner",     "executor2")
    workflow.add_edge("executor2",          END)

    return workflow.compile(checkpointer=checkpointer)
