# graph.py
from langgraph.graph import StateGraph, END
from source_code.state import AgentState
from source_code.agents.agent_1_field_renamer import field_renamer_agent
from source_code.agents.agent_2_field_cleaner import field_cleaner_agent
from source_code.agents.executor import code_executor_agent

# 1. Initialize the graph with our schema
workflow = StateGraph(AgentState)

# 2. Add nodes
workflow.add_node("agent1_renamer",  field_renamer_agent)
workflow.add_node("executor1",       code_executor_agent)
workflow.add_node("agent2_cleaner",  field_cleaner_agent)
workflow.add_node("executor2",       code_executor_agent)

# 3. Define edges
workflow.set_entry_point("agent1_renamer")

workflow.add_edge("agent1_renamer", "executor1")   # Agent 1 writes code → Executor runs it
workflow.add_edge("executor1",      "agent2_cleaner")  # Cleaned df passes to Agent 2
workflow.add_edge("agent2_cleaner", "executor2")   # Agent 2 writes code → Executor runs it
workflow.add_edge("executor2",      END)

# 4. Compile
ds_machine = workflow.compile()