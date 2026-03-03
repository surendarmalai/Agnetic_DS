import re
from langgraph.types import interrupt
from source_code.state import AgentState
from source_code.utils import load_prompts
from source_code.config.llm_config import PipelineConfig
from source_code.config.llm_factory import LLMFactory

PROMPTS_CONFIG = load_prompts(r'source_code/prompts/agent_2.json')


def make_field_cleaner_agent(config: PipelineConfig):
    """
    Factory function. Returns a LangGraph-compatible node function
    with the LLM resolved from config and closed over.

    Parameters
    ----------
    config : PipelineConfig
        Provides LLM parameters for Agent 2.

    Returns
    -------
    Callable[[AgentState], dict]
    """
    llm_cfg = config.get_llm_config("agent2")
    llm = LLMFactory.create(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        llm_instance=llm_cfg.llm_instance,
    )

    def field_cleaner_agent(state: AgentState) -> dict:
        """
        Agent 2 — Field Cleaner.

        Reads from state:
            metadata_summary      : df.info, df.describe, df.head
            categorical_cols      : genuinely categorical object columns
            numeric_like_cols     : object columns that are actually numeric but dirty
            value_counts_summary  : top 20 value_counts for categorical columns
            null_summary          : isna().sum() for all columns
            special_rules         : optional human overrides

        Writes to state:
            cleaning_code         : executable Python cleaning code
            flagged_columns       : columns that could not be safely cleaned
            output_path           : where executor should save the result
        """
        print()
        print("[Agent 2] Field Cleaning — Starting")
        print()

        categorical_cols  = state.get('categorical_cols',  [])
        numeric_like_cols = state.get('numeric_like_cols', [])

        print(f"[Agent 2] Categorical columns  : {len(categorical_cols)}")  # type: ignore
        print(f"[Agent 2] Numeric-like (dirty) : {len(numeric_like_cols)}")  # type: ignore

        # ── Build prompt from JSON template ─────────────────────────────────────────
        kb       = PROMPTS_CONFIG['field_cleaning_agent']
        template = kb['instructions']

        prompt = (template
            .replace("{metadata_summary}",           state.get("metadata_summary",     "Not provided"))
            .replace("{null_summary}",               state.get("null_summary",          "Not provided"))
            .replace("{value_counts_summary}",       state.get("value_counts_summary",  "Not provided"))
            .replace("{categorical_cols}",           str(categorical_cols))
            .replace("{numeric_like_cols}",          str(numeric_like_cols))
            .replace("{special_rules}",              state.get("special_rules",         "None"))
            .replace("{categorical_cleaning_rules}", kb["categorical_cleaning_rules"])
            .replace("{numeric_like_cleaning_rules}",kb["numeric_like_cleaning_rules"])
            .replace("{flagging_rule}",              kb["flagging_rule"])
        )

        print("[Agent 2] Sending to LLM...")
        response     = llm.invoke(prompt)

        print("\n--- RAW LLM RESPONSE ---")
        print(response.content)
        print("------------------------\n")

        raw_response = response.content

        # ── Parse code blocks ────────────────────────────────────────────────────────
        code_blocks = re.findall(r'```(?:python)?(.*?)```', raw_response, re.DOTALL)  # type: ignore
        code_blocks = [block.strip() for block in code_blocks if block.strip()]

        if not code_blocks:
            print("[Agent 2] ERROR: LLM returned no parseable code blocks.")
            return {
                **state,
                "cleaning_code"  : "",
                "flagged_columns": [],
                "output_path"    : "output_agent2_cleaned.csv"
            }

        cleaning_code = code_blocks[0] if len(code_blocks) >= 1 else ""
        flagged_code  = code_blocks[1] if len(code_blocks) >= 2 else ""

        # ── Safely extract flagged_columns ──────────────────────────────────────────
        flagged_columns = []
        if flagged_code:
            try:
                local_ns = {}
                exec(flagged_code, {"__builtins__": __builtins__}, local_ns)  # [AUDIT L12]
                flagged_columns = local_ns.get("flagged_columns", [])
            except Exception as e:
                print(f"[Agent 2] WARNING: Could not parse flagged_columns: {e}")

        # ── Surface flagged columns ──────────────────────────────────────────────────
        if flagged_columns:
            print(f"\n[Agent 2] {len(flagged_columns)} FLAGGED COLUMN(S) — Could not clean safely:\n")
            for f in flagged_columns:
                print(f"  Column : {f.get('column')}")
                print(f"  Reason : {f.get('reason')}\n")
            try:
                # interrupt() only works inside a LangGraph runnable context.
                # When called directly (e.g. unit tests), RuntimeError is raised
                # and we skip the pause — behaviour is identical to pre-interrupt.
                interrupt({
                    "type":    "flagged_columns",
                    "columns": flagged_columns,
                })
                # return value discarded — user just acknowledges, pipeline continues
            except RuntimeError:
                pass

        print(f"[Agent 2] Cleaning code ready ({len(cleaning_code)} chars).\n")

        return {
            "cleaning_code"  : cleaning_code,
            "flagged_columns": flagged_columns,
            "output_path"    : "output_agent2_cleaned.csv",
        }

    return field_cleaner_agent
