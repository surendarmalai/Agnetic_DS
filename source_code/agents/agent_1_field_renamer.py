import json
import re
from langgraph.types import interrupt
from source_code.state import AgentState
from source_code.utils import preprocess_column_names, load_prompts
from source_code.config.llm_config import PipelineConfig
from source_code.config.llm_factory import LLMFactory

PROMPTS_CONFIG = load_prompts(r'source_code/prompts/agent_1.json')


def make_field_renamer_agent(config: PipelineConfig):
    """
    Factory function. Returns a LangGraph-compatible node function
    with the LLM resolved from config and closed over.

    Parameters
    ----------
    config : PipelineConfig
        Provides LLM parameters. The LLM is instantiated once at factory
        call time, then closed over in the returned function.

    Returns
    -------
    Callable[[AgentState], dict]
        LangGraph node function. Identical to old field_renamer_agent
        except the LLM comes from config instead of being hardcoded.
    """
    llm_cfg = config.get_llm_config("agent1")
    llm = LLMFactory.create(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        llm_instance=llm_cfg.llm_instance,   # [AUDIT H4]
    )

    def field_renamer_agent(state: AgentState) -> dict:
        """LangGraph node: field renamer. Logic identical to original."""

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

        print("[Agent 1] Sending to LLM...")
        response = llm.invoke(prompt)
        raw_response = response.content

        # ── Parse code blocks ──
        code_blocks = re.findall(r'```(?:python)?(.*?)```', raw_response, re.DOTALL)  # type: ignore
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
                exec(ambiguous_code, {"__builtins__": __builtins__}, local_ns)  # [AUDIT L12]
                ambiguous_fields = local_ns.get("ambiguous_fields", [])
            except Exception as e:
                print(f"[Agent 1] WARNING: Could not parse ambiguous_fields: {e}")

        # ── Safely extract rename_map ──
        column_map = {}
        try:
            map_code = rename_code.split("df = df.rename")[0]
            local_ns = {}
            exec(map_code, {"__builtins__": __builtins__}, local_ns)  # [AUDIT L12]
            column_map = local_ns.get("rename_map", {})
        except Exception as e:
            print(f"[Agent 1] WARNING: Could not parse rename_map: {e}")

        if ambiguous_fields:
            try:
                # interrupt() only works inside a LangGraph runnable context.
                # When called directly (e.g. unit tests), RuntimeError is raised
                # and we skip the pause — behaviour is identical to pre-interrupt.
                user_decisions = interrupt({
                    "type":   "ambiguous_fields",
                    "fields": ambiguous_fields,
                })
                if isinstance(user_decisions, dict):
                    column_map.update(user_decisions)
            except RuntimeError:
                pass

        return {
            "cleaning_code": rename_code,
            "ambiguous_fields": ambiguous_fields,
            "column_map": column_map,
        }

    return field_renamer_agent
