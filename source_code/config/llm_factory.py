from __future__ import annotations
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """
    Single factory point for all LLM instantiation.
    Adding a new provider requires changes only in this class.

    All provider imports are deferred (inside if-branches) so that only the
    installed provider's LangChain package is required at runtime.
    """

    @staticmethod
    def create(
        provider    : str,
        model       : str,
        api_key     : Optional[str] = None,
        base_url    : Optional[str] = None,
        temperature : float = 0.0,
        max_tokens  : int   = 8000,
        llm_instance: Any   = None,
    ) -> BaseChatModel:
        """
        Instantiate and return a LangChain BaseChatModel.

        Parameters
        ----------
        provider     : One of "groq", "openai", "anthropic", "ollama" (case-insensitive).
        model        : Provider-specific model name.
        api_key      : API key. None is valid only for "ollama".
        base_url     : Base URL override. Required for "ollama".
        temperature  : Sampling temperature. Defaults to 0.
        max_tokens   : Maximum response tokens. Defaults to 8000.
        llm_instance : [AUDIT H4] If not None, return this object directly.
                       Used to inject MockLLM or pre-built LLM in tests.
                       The caller (agent factory) passes llm_cfg.llm_instance here.

        Returns
        -------
        BaseChatModel or llm_instance if provided.

        Raises
        ------
        ValueError : If provider is not in the supported list.
        """
        if llm_instance is not None:
            return llm_instance

        p = provider.lower().strip()

        if p == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )
        elif p == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )
        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )
        elif p == "ollama":
            from langchain_ollama import ChatOllama  # [AUDIT M11]
            return ChatOllama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
            )
            # Note: ChatOllama does not accept max_tokens; uses num_predict.
            # langchain-ollama package must be installed separately.
        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: groq, openai, anthropic, ollama"
            )
