from __future__ import annotations
from typing import Any, Iterator, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
try:
    from pydantic import ConfigDict
    _PYDANTIC_V2 = True
except ImportError:
    _PYDANTIC_V2 = False


class MockResponse:
    """
    Minimal stand-in for LangChain's AIMessage.
    Exposes only .content, which is what all current agent code reads.

    Parameters
    ----------
    content : str
        The pre-programmed response string.
    """
    def __init__(self, content: str) -> None:
        self.content: str = content


class MockLLM(BaseChatModel):
    """
    Test double for any LangChain BaseChatModel.
    Returns a pre-programmed response string without making network calls.

    Inherits BaseChatModel to satisfy type annotations and LangChain
    introspection. Stubs _generate and _llm_type as required by the ABC.
    [AUDIT H3]

    Attributes
    ----------
    last_prompt : Any
        The exact value passed to the most recent .invoke() call.
        None if .invoke() has not been called yet.
        Used in tests to assert that prompt context (SQL, metadata) was
        correctly injected before the LLM call.

    Usage
    -----
        mock = MockLLM(content="```python\\nrename_map = {}\\n```")
        cfg = PipelineConfig.with_mock(mock)
        agent_fn = make_field_renamer_agent(cfg)
        result = agent_fn(state)
        assert mock.last_prompt is not None
        assert "SELECT" in str(mock.last_prompt)
    """

    # Allow arbitrary attributes (e.g. last_prompt) on pydantic model
    if _PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pydantic field for the pre-programmed content.
    # BaseChatModel uses pydantic; declare the custom field here.
    _content: str = ""

    def __init__(self, content: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        content : str
            The string that .invoke() will return as MockResponse.content.
        """
        super().__init__(**kwargs)
        self._content = content
        object.__setattr__(self, "last_prompt", None)  # bypass pydantic __setattr__

    @property
    def _llm_type(self) -> str:
        """
        Required by BaseChatModel ABC.
        Returns "mock" to identify this as a test double.
        """
        return "mock"

    def _generate(
        self,
        messages  : List[BaseMessage],
        stop      : Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs  : Any,
    ) -> ChatResult:
        """
        Required by BaseChatModel ABC.
        Records the messages (for last_prompt) and returns the pre-programmed content.

        Parameters
        ----------
        messages    : List[BaseMessage]  — The prompt messages passed by LangChain.
        stop        : Optional stop sequences (ignored).
        run_manager : LangChain callback manager (ignored).

        Returns
        -------
        ChatResult
            Wraps self._content in a ChatGeneration / AIMessage.
        """
        object.__setattr__(self, "last_prompt", messages)
        message = AIMessage(content=self._content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> MockResponse:
        """
        Override invoke() to record last_prompt and return MockResponse.
        This is what agent code calls directly (llm.invoke(prompt)).

        Parameters
        ----------
        input : Any
            The prompt. May be a str (current agents) or List[BaseMessage] (future).

        Returns
        -------
        MockResponse
            Object with .content = self._content.
        """
        object.__setattr__(self, "last_prompt", input)
        return MockResponse(self._content)
