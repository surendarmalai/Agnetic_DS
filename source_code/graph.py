# graph.py
from langgraph.graph import StateGraph, END
from source_code.state import AgentState
from source_code.agents.agents import field_standardization_agent
from source_code.tools import code_executor_agent

# 1. Initialize the graph with our schema
workflow = StateGraph(AgentState)

# 2. Add our "Work Stations" (Nodes)
workflow.add_node("cleaner", field_standardization_agent)
workflow.add_node("executor", code_executor_agent)

# 3. Define the Assembly Line (Edges)
workflow.set_entry_point("cleaner")

# Pass the state from the AI Brain directly to the Executor Hands
workflow.add_edge("cleaner", "executor") 

# Once the Executor is done, finish the pipeline
workflow.add_edge("executor", END)

# 4. Compile the machine
ds_machine = workflow.compile()