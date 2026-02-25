from typing import TypedDict, Optional

# class passed between agents.    
class AgentState(TypedDict):
    file_path: str                      # Path to CSV
    target_column: str                  # target column
    metadata_summary: str               # Description of columns
    cleaning_code: Optional[str]        # The Python code the Agent will generate
    iteration_count: Optional[int]
    error_log: Optional[str]            # To track self-correction loops