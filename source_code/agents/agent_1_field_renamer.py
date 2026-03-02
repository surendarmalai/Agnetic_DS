import os
import json
from langchain_groq import ChatGroq
from source_code.state import AgentState
from source_code.utils import preprocess_column_names, load_prompts
import re

PROMPTS_CONFIG = load_prompts(r'source_code/prompts/agent_1.json')



def field_renamer_agent(state: AgentState) -> dict:

    print("Agent 1 Field Renamer - Starting")

    raw_columns = state.get('df_columns', [])
    pre_cleaned = preprocess_column_names(raw_columns)
    cleaned_cols = list(pre_cleaned.values())

    # Build the prompt using the JSON template
    kb = PROMPTS_CONFIG['telecom_knowledge']
    template = PROMPTS_CONFIG['field_renamer_agent']['instructions']
    
    prompt = (template
                .replace("{sql_query}", state.get('sql_query', 'Not provided'))
                .replace("{cleaned_cols}", str(cleaned_cols))
                .replace("{metadata_summary}", state.get('metadata_summary', 'Not provided'))
                .replace("{special_rules}", state.get('special_rules', 'None'))
                .replace("{abbreviation_map}", kb['abbreviation_map'])
                .replace("{unit_rules}", kb['unit_rules'])
                .replace("{period_rules}", kb['period_rules'])
                .replace("{naming_style_rules}", kb['naming_style_rules'])
                .replace("{date_rules}", kb['date_rules'])
                .replace("{target_schema}", kb['target_schema'])
            )

    llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY"), # type: ignore
                    max_tokens=8000
                )

    print("[Agent 1] Sending to LLM...")
    response = llm.invoke(prompt)
    raw_response = response.content

    # ── Parse code blocks ──
    code_blocks = re.findall(r'```(?:python)?(.*?)```', raw_response, re.DOTALL) # type: ignore
    code_blocks = [block.strip() for block in code_blocks if block.strip()]

    if not code_blocks:
        print("[Agent 1] ERROR: LLM returned no parseable code blocks.")
        return {**state, "cleaning_code": "", "ambiguous_fields": [], "column_map": {}}

    rename_code = code_blocks[0]
    ambiguous_code = code_blocks[1] if len(code_blocks) >= 2 else ""

    # ── Safely extract ambiguous_fields ──
    ambiguous_fields = []
    if ambiguous_code:
        try:
            local_ns = {}
            exec(ambiguous_code, {}, local_ns)
            ambiguous_fields = local_ns.get("ambiguous_fields", [])
        except Exception as e:
            print(f"[Agent 1] WARNING: Could not parse ambiguous_fields: {e}")

    # ── Safely extract rename_map ──
    column_map = {}
    try:
        map_code = rename_code.split("df = df.rename")[0]
        local_ns = {}
        exec(map_code, {}, local_ns)
        column_map = local_ns.get("rename_map", {})
    except Exception as e:
        print(f"[Agent 1] WARNING: Could not parse rename_map: {e}")

    return {
        "cleaning_code": rename_code,
        "ambiguous_fields": ambiguous_fields,
        "column_map": column_map,
    }