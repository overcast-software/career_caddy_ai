from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage
from pydantic_ai._utils import PeekableAsyncStream
from pydanticai_ollama.models.ollama import OllamaModel, OllamaStreamedResponse
from pydanticai_ollama.providers.ollama import OllamaProvider
from pydanticai_ollama.settings.ollama import OllamaModelSettings


# Define a Pydantic model for your tool's arguments
class GetCurrentWeatherArgs(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(
        default="celsius",
        description="The unit of temperature (celsius or fahrenheit) to return",
    )


class ConcreteOllamaProvider(OllamaProvider):
    """Concrete implementation of OllamaProvider with provider_url method."""

    def provider_url(self) -> str:
        return self.base_url


@dataclass
class ConcreteOllamaStreamedResponse(OllamaStreamedResponse):
    """Concrete implementation of OllamaStreamedResponse with provider_url method."""

    _model_name: str
    _model_profile: ModelProfile
    _response: PeekableAsyncStream[Any]
    _timestamp: datetime

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str | None:
        return "ollama"

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def provider_url(self) -> str:
        return "http://localhost:11434"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Asynchronously iterates over Ollama stream chunks and yields ModelResponseStreamEvent."""
        async for chunk in self._response:
            self._usage += RequestUsage(input_tokens=0, output_tokens=1)
            # Process chunk and yield events - simplified implementation
            if hasattr(chunk, "message") and chunk.message.content:
                text_event = self._parts_manager.handle_text_delta(
                    vendor_part_id="content",
                    content=chunk.message.content,
                    thinking_tags=self._model_profile.thinking_tags,
                    ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                )
                if text_event:
                    yield text_event


class ConcreteOllamaModel(OllamaModel):
    """Concrete implementation of OllamaModel that returns ConcreteOllamaStreamedResponse."""

    async def _process_streamed_response(
        self,
        response: AsyncIterator[Any],
        model_request_parameters: ModelRequestParameters,
    ) -> ConcreteOllamaStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()

        return ConcreteOllamaStreamedResponse(
            model_request_parameters=model_request_parameters,
            _response=peekable_response,
            _model_name=self._model_name,
            _model_profile=self.profile,
            _timestamp=datetime.now(timezone.utc),
        )


# Configure OllamaModelSettings with desired parameters
ollama_settings = OllamaModelSettings(
    temperature=0.1,
    num_predict=1024,
    num_ctx=16384,
    repeat_penalty=1.1,
    num_gpu=1,
    top_k=40,
    top_p=0.9,
)

# Initialize OllamaProvider with your Ollama server's base URL
ollama_provider = ConcreteOllamaProvider(base_url="http://localhost:11434")

voytas26_openclaw_oss = "voytas26/openclaw-oss-20b-deterministic"
voytas26_openclaw_oss_model = ConcreteOllamaModel(
    model_name=voytas26_openclaw_oss,
    provider=ollama_provider,
)
phi3 = "phi3:14b"
phi3_model = ConcreteOllamaModel(
    model_name=phi3,
    provider=ollama_provider,
)
# Create a ConcreteOllamaModel instance
voytas26_openclaw_qwen3vl = "voytas26/openclaw-qwen3vl-8b-opt"
voytas26_openclaw_qwen3vl_model = ConcreteOllamaModel(
    model_name=voytas26_openclaw_qwen3vl,
    provider=ollama_provider,
)

astrail3_tools = "60MPH/astral3-tools:12b"
astrail3_model = ConcreteOllamaModel(
    model_name=astrail3_tools,
    provider=ollama_provider,
)

# NOTE: chevalblanc/gpt-4o-mini does NOT support tool calling.
# Ollama returns: "does not support tools"
gpt_4o_mini_tools = "chevalblanc/gpt-4o-mini:latest"
gpt_40_mini_model = ConcreteOllamaModel(
    model_name=gpt_4o_mini_tools,
    provider=ollama_provider,
)

# ---------------------------------------------------------------------------
# Tool-capable models via OpenAI-compatible endpoint (/v1)
# Verified to support tool calling within 8 GB VRAM.
# Use these anywhere agents need MCP or pydantic-ai tools.
# ---------------------------------------------------------------------------
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_ollama_openai_provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# qwen3:4b-instruct — 2.5 GB, fast, good for orchestration
qwen3_4b_model = OpenAIChatModel(
    "qwen3:4b-instruct",
    provider=_ollama_openai_provider,
)

# qwen2.5-coder:7b — 4.7 GB, stronger reasoning, better for tool-heavy agents
qwen25_coder_7b_model = OpenAIChatModel(
    "qwen2.5-coder:7b",
    provider=_ollama_openai_provider,
)

llama3 = ConcreteOllamaModel(
    model_name="llama3.3",
    provider=ollama_provider,
)
# Default tool-capable Ollama model
# ollama_model = astrail3_model
# ollama_model = qwen25_coder_7b_model  # outputs tool calls as text, not tool_calls response

# ollama_model = qwen3_4b_model

ollama_model = "openai:gpt-4o-mini"
# ---------------------------------------------------------------------------
# Global model — change this one line to switch all agents at once.
# Swap in any model above, or use "openai:gpt-4o-mini" for cloud fallback.
# ---------------------------------------------------------------------------
global_model = ollama_model

# Browser-optimised model: larger context window + low temperature for
# reliable multi-step tool sequencing (create_tab → navigate → snapshot).
# num_ctx=16384 prevents tool schemas + page content from overflowing the context.
# temperature=0.1 makes tool call ordering deterministic.
browser_ollama_model = OpenAIChatModel(
    "qwen3:4b-instruct",
    provider=_ollama_openai_provider,
    settings={
        "temperature": 0.1,
        "extra_body": {
            "options": {
                "num_ctx": 16384,
                "num_predict": 1024,
                "repeat_penalty": 1.1,
            }
        },
    },
)
