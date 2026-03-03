from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class LLMConfig:
    """
    Configuration for a single LLM instance.
    Passed to LLMFactory.create() to produce a BaseChatModel.

    The llm_instance field (no underscore) is the DI hook for tests.
    When set, LLMFactory.create() returns it directly without instantiation.
    It is included in repr so debugging shows the injected mock.
    compare=False: two configs with different injected mocks compare equal
    if all other fields match (correct semantics for config deduplication).
    """
    provider    : str           = "groq"
    model       : str           = "llama-3.3-70b-versatile"
    api_key     : Optional[str] = None
    base_url    : Optional[str] = None
    temperature : float         = 0.0
    max_tokens  : int           = 8000
    llm_instance: Any           = field(default=None, repr=True, compare=False)
    # [AUDIT H4] Renamed from _llm_instance to llm_instance (no underscore).
    # repr=True so debugging shows the injected mock.


@dataclass
class PipelineConfig:
    """
    Top-level configuration object for a pipeline session.
    Constructed once in main.py (or test) and injected into agent factories.
    Never stored in AgentState. Never holds live LLM instances directly.

    agent_overrides keys: "agent1", "agent2" (stable logical identifiers).
    """
    default_llm    : LLMConfig                = field(default_factory=LLMConfig)
    agent_overrides: dict                     = field(default_factory=dict)

    def get_llm_config(self, agent_name: str) -> LLMConfig:
        """
        Return the LLMConfig for the named agent.

        Parameters
        ----------
        agent_name : str
            Logical agent identifier: "agent1" or "agent2".

        Returns
        -------
        LLMConfig
            Agent-specific override if registered, otherwise default_llm.
        """
        return self.agent_overrides.get(agent_name, self.default_llm)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Construct PipelineConfig from environment variables.

        Environment variables (all optional):
            LLM_PROVIDER  : defaults to "groq"
            LLM_MODEL     : defaults to "llama-3.3-70b-versatile"
            LLM_API_KEY   : explicit key; falls back to GROQ_API_KEY
            LLM_BASE_URL  : used for Ollama; optional

        Returns
        -------
        PipelineConfig
            Config backed by environment. For test injection post-construction:
                cfg = PipelineConfig.from_env()
                cfg.default_llm.llm_instance = mock
            Or use the convenience factory:
                cfg = PipelineConfig.with_mock(mock)
        """
        return cls(default_llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "groq"),
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        ))

    @classmethod
    def with_mock(cls, mock: Any) -> "PipelineConfig":
        """
        Convenience factory for tests. Returns a PipelineConfig where
        ALL agents receive the given mock LLM instance.

        Parameters
        ----------
        mock : Any
            A MockLLM instance (or any object with a .invoke() method).

        Returns
        -------
        PipelineConfig
            Config whose default_llm.llm_instance is set to mock.

        Usage
        -----
            cfg = PipelineConfig.with_mock(MockLLM(content="..."))
            agent_fn = make_field_renamer_agent(cfg)

        [AUDIT H4] This is the recommended injection path for tests.
        """
        return cls(default_llm=LLMConfig(llm_instance=mock))
