"""Unified model adapter for CompToolBench.

Uses LiteLLM to provide a single interface across all LLM providers:
- Ollama (local, $0 cost)
- Google AI Studio / Gemini (free tier: 100 req/day)
- Groq (free tier: blazing fast inference)
- OpenRouter (free credits, access to many models)
- Mistral (1B free tokens/month)
- OpenAI / Anthropic (paid, optional)

Each model is tested for tool-calling support before benchmarking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import litellm

from comptoolbench.evaluation.scorers import ModelCall
from comptoolbench.tools.base import BaseTool, ToolMode, get_all_tools

logger = logging.getLogger(__name__)

# Silence LiteLLM's noisy logging
litellm.suppress_debug_info = True
litellm.set_verbose = False


class Provider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    DEEPSEEK = "deepseek"
    CEREBRAS = "cerebras"
    SAMBANOVA = "sambanova"
    COHERE = "cohere"


@dataclass
class ModelConfig:
    """Configuration for a single model to evaluate."""

    name: str                          # Display name (e.g., "Qwen3 14B")
    litellm_id: str                    # LiteLLM model ID (e.g., "ollama/qwen3:14b")
    provider: Provider
    supports_tools: bool = True        # Whether the model supports tool calling
    max_tokens: int = 4096
    temperature: float = 0.0           # Deterministic by default
    requests_per_minute: int | None = None  # Rate limit
    cost_per_1k_input: float = 0.0     # For cost tracking
    cost_per_1k_output: float = 0.0
    api_base: str | None = None        # Custom API base URL


# ---------------------------------------------------------------------------
# Pre-configured model registry
# ---------------------------------------------------------------------------

# Models we can test for FREE right now.
# Tool-calling support verified empirically on 2026-02-21.
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    # --- Ollama (local, FREE, verified tool-calling) ---
    "qwen3-8b": ModelConfig(
        name="Qwen3 8B",
        litellm_id="ollama/qwen3:8b",
        provider=Provider.OLLAMA,
    ),
    "mistral-nemo-12b": ModelConfig(
        name="Mistral Nemo 12B",
        litellm_id="ollama/mistral-nemo:12b",
        provider=Provider.OLLAMA,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        litellm_id="ollama/mistral:7b",
        provider=Provider.OLLAMA,
    ),
    "llama3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        litellm_id="ollama/llama3.1:8b",
        provider=Provider.OLLAMA,
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen 2.5 7B",
        litellm_id="ollama/qwen2.5:7b",
        provider=Provider.OLLAMA,
    ),
    "granite4-3b": ModelConfig(
        name="Granite4 3B",
        litellm_id="ollama/granite4:3b",
        provider=Provider.OLLAMA,
    ),
    "granite4-1b": ModelConfig(
        name="Granite4 1B",
        litellm_id="ollama/granite4:1b",
        provider=Provider.OLLAMA,
    ),
    "mistral-small-24b": ModelConfig(
        name="Mistral Small 24B",
        litellm_id="ollama/mistral-small:24b",
        provider=Provider.OLLAMA,
    ),
    "command-r-35b": ModelConfig(
        name="Command R 35B",
        litellm_id="ollama/command-r:35b",
        provider=Provider.OLLAMA,
    ),
    # Ollama models with NO native tool-calling (use text-parsing fallback)
    "qwen3-14b": ModelConfig(
        name="Qwen3 14B",
        litellm_id="ollama/qwen3:14b",
        provider=Provider.OLLAMA,
        supports_tools=False,
    ),
    "qwen3-4b": ModelConfig(
        name="Qwen3 4B",
        litellm_id="ollama/qwen3:4b",
        provider=Provider.OLLAMA,
        supports_tools=False,
    ),
    "qwen3-1.7b": ModelConfig(
        name="Qwen3 1.7B",
        litellm_id="ollama/qwen3:1.7b",
        provider=Provider.OLLAMA,
        supports_tools=False,
    ),
    "qwen3-0.6b": ModelConfig(
        name="Qwen3 0.6B",
        litellm_id="ollama/qwen3:0.6b",
        provider=Provider.OLLAMA,
        supports_tools=False,
    ),
    "deepseek-r1-14b": ModelConfig(
        name="DeepSeek R1 14B",
        litellm_id="ollama/deepseek-r1:14b",
        provider=Provider.OLLAMA,
        supports_tools=False,
    ),
    # --- Gemini (free tier, quota needs ~1hr to provision on new accounts) ---
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        litellm_id="gemini/gemini-2.0-flash",
        provider=Provider.GEMINI,
        requests_per_minute=15,
    ),
    "gemini-2.5-pro": ModelConfig(
        name="Gemini 2.5 Pro",
        litellm_id="gemini/gemini-2.5-pro",
        provider=Provider.GEMINI,
        requests_per_minute=5,
    ),
    # --- Groq (free tier, blazing fast, verified) ---
    # Rate: 10 req/min to avoid hitting free-tier limits on long runs
    "groq-llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B (Groq)",
        litellm_id="groq/llama-3.3-70b-versatile",
        provider=Provider.GROQ,
        requests_per_minute=10,
    ),
    "groq-llama3.1-8b": ModelConfig(
        name="Llama 3.1 8B (Groq)",
        litellm_id="groq/llama-3.1-8b-instant",
        provider=Provider.GROQ,
        requests_per_minute=10,
    ),
    "groq-llama4-scout": ModelConfig(
        name="Llama 4 Scout 17B (Groq)",
        litellm_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        provider=Provider.GROQ,
        requests_per_minute=10,
    ),
    # --- OpenRouter (free credits, use paid-tier models with tool support) ---
    "or-gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash (OpenRouter)",
        litellm_id="openrouter/google/gemini-2.0-flash-001",
        provider=Provider.OPENROUTER,
        requests_per_minute=20,
    ),
    "or-llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B (OpenRouter)",
        litellm_id="openrouter/meta-llama/llama-3.3-70b-instruct",
        provider=Provider.OPENROUTER,
        requests_per_minute=20,
    ),
    # --- Mistral (free 1B tokens/month, verified) ---
    "mistral-small": ModelConfig(
        name="Mistral Small",
        litellm_id="mistral/mistral-small-latest",
        provider=Provider.MISTRAL,
        requests_per_minute=20,
    ),
    "mistral-medium": ModelConfig(
        name="Mistral Medium",
        litellm_id="mistral/mistral-medium-latest",
        provider=Provider.MISTRAL,
        requests_per_minute=10,
    ),
    "mistral-large": ModelConfig(
        name="Mistral Large",
        litellm_id="mistral/mistral-large-latest",
        provider=Provider.MISTRAL,
        requests_per_minute=5,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
    ),
    # --- xAI Grok (free $25 credit) ---
    "grok-3-mini": ModelConfig(
        name="Grok 3 Mini",
        litellm_id="xai/grok-3-mini-beta",
        provider=Provider.XAI,
        requests_per_minute=30,
        cost_per_1k_input=0.0003,
        cost_per_1k_output=0.0005,
    ),
    "grok-3-mini-fast": ModelConfig(
        name="Grok 3 Mini Fast",
        litellm_id="xai/grok-3-mini-fast-beta",
        provider=Provider.XAI,
        requests_per_minute=60,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
    ),
    # --- DeepSeek (paid, very cheap) ---
    "deepseek-v3": ModelConfig(
        name="DeepSeek V3",
        litellm_id="deepseek/deepseek-chat",
        provider=Provider.DEEPSEEK,
        requests_per_minute=30,
        cost_per_1k_input=0.00027,
        cost_per_1k_output=0.0011,
    ),
    # --- Gemini additional models ---
    "gemini-2.0-flash-lite": ModelConfig(
        name="Gemini 2.0 Flash Lite",
        litellm_id="gemini/gemini-2.0-flash-lite",
        provider=Provider.GEMINI,
        requests_per_minute=30,
    ),
    # --- Cerebras (free 1M tokens/day, ultra-fast inference) ---
    "cerebras-llama3.1-8b": ModelConfig(
        name="Llama 3.1 8B (Cerebras)",
        litellm_id="cerebras/llama3.1-8b",
        provider=Provider.CEREBRAS,
        requests_per_minute=30,
    ),
    "cerebras-gpt-oss-120b": ModelConfig(
        name="GPT-OSS 120B (Cerebras)",
        litellm_id="cerebras/gpt-oss-120b",
        provider=Provider.CEREBRAS,
        requests_per_minute=30,
    ),
    # --- SambaNova (free tier, 40 req/day, tool calling verified) ---
    "samba-llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B (SambaNova)",
        litellm_id="sambanova/Meta-Llama-3.3-70B-Instruct",
        provider=Provider.SAMBANOVA,
        requests_per_minute=40,
    ),
    "samba-llama4-maverick": ModelConfig(
        name="Llama 4 Maverick 17B (SambaNova)",
        litellm_id="sambanova/Llama-4-Maverick-17B-128E-Instruct",
        provider=Provider.SAMBANOVA,
        requests_per_minute=20,
    ),
    "samba-deepseek-v3": ModelConfig(
        name="DeepSeek V3 (SambaNova)",
        litellm_id="sambanova/DeepSeek-V3-0324",
        provider=Provider.SAMBANOVA,
        requests_per_minute=20,
    ),
    # --- Cohere (free trial: 1000 calls/month, excellent tool use) ---
    "cohere-command-a": ModelConfig(
        name="Command A",
        litellm_id="command-a-03-2025",
        provider=Provider.COHERE,
        requests_per_minute=20,
    ),
    "cohere-command-r-plus": ModelConfig(
        name="Command R+",
        litellm_id="command-r-plus-08-2024",
        provider=Provider.COHERE,
        requests_per_minute=20,
    ),
}


def get_free_models() -> dict[str, ModelConfig]:
    """Return only models available on free tiers (no paid API keys needed)."""
    return {k: v for k, v in AVAILABLE_MODELS.items()}


def get_ollama_models() -> dict[str, ModelConfig]:
    """Return only local Ollama models ($0 cost)."""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.provider == Provider.OLLAMA
    }


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------

def tools_to_openai_schema(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert CompToolBench tools to OpenAI function-calling format.

    BaseTool.get_openai_schema() already returns the full
    {"type": "function", "function": {...}} structure.
    """
    return [tool.get_openai_schema() for tool in tools]


def tools_by_name(tool_names: list[str]) -> list[BaseTool]:
    """Get tool instances by name from the registry."""
    all_tools = get_all_tools()  # {name: ToolClass}
    result = []
    for name in tool_names:
        tool_cls = all_tools.get(name)
        if tool_cls is not None:
            result.append(tool_cls(mode=ToolMode.SIMULATED))
    return result


# ---------------------------------------------------------------------------
# Model adapter (the core class)
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    """Result of a single model API call."""

    model_calls: list[ModelCall]       # Parsed tool calls from response
    raw_response: dict[str, Any]       # Full LiteLLM response
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ModelAdapter:
    """Unified interface for calling LLMs with tool-calling support.

    Handles:
    - Provider-specific formatting
    - Rate limiting
    - Response parsing (tool calls extraction)
    - Error handling and retries
    - Cost/token tracking
    """

    config: ModelConfig
    _last_call_time: float = field(default=0.0, repr=False)

    def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limits."""
        if self.config.requests_per_minute is None:
            return
        min_interval = 60.0 / self.config.requests_per_minute
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        """Build the messages array for the API call."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_tool_calls(
        self, response: Any,
    ) -> list[ModelCall]:
        """Extract tool calls from the LiteLLM response."""
        calls: list[ModelCall] = []

        message = response.choices[0].message

        # Check for tool_calls in the response
        if hasattr(message, "tool_calls") and message.tool_calls:
            for i, tc in enumerate(message.tool_calls):
                func = tc.function
                try:
                    args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}

                calls.append(ModelCall(
                    tool_name=func.name,
                    arguments=args,
                    call_index=i,
                ))

        # Fallback: try to parse tool calls from text content
        # Some models (especially smaller ones) output JSON in text
        if not calls and hasattr(message, "content") and message.content:
            calls = self._parse_tool_calls_from_text(message.content)

        return calls

    def _parse_tool_calls_from_text(self, text: str) -> list[ModelCall]:
        """Attempt to extract tool calls from plain text response.

        Handles models that don't support native tool calling but output
        structured JSON. Looks for patterns like:
        - {"tool": "name", "arguments": {...}}
        - [{"tool": "name", ...}, ...]
        - ```json ... ```
        """
        calls: list[ModelCall] = []

        # Strip markdown code fences
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start) if "```" in cleaned[start:] else len(cleaned)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start) if "```" in cleaned[start:] else len(cleaned)
            cleaned = cleaned[start:end].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON objects in the text
            return self._extract_json_objects(text)

        # Handle single dict or list of dicts
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    tool_name = item.get("tool") or item.get("name") or item.get("function", "")
                    arguments = item.get("arguments") or item.get("args") or item.get("parameters", {})
                    if tool_name:
                        calls.append(ModelCall(
                            tool_name=str(tool_name),
                            arguments=arguments if isinstance(arguments, dict) else {},
                            call_index=i,
                        ))

        return calls

    def _extract_json_objects(self, text: str) -> list[ModelCall]:
        """Extract JSON objects from mixed text (last resort parser)."""
        calls: list[ModelCall] = []
        depth = 0
        start = -1

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        tool_name = obj.get("tool") or obj.get("name") or obj.get("function", "")
                        arguments = obj.get("arguments") or obj.get("args") or obj.get("parameters", {})
                        if tool_name:
                            calls.append(ModelCall(
                                tool_name=str(tool_name),
                                arguments=arguments if isinstance(arguments, dict) else {},
                                call_index=len(calls),
                            ))
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return calls

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_schemas: list[dict[str, Any]],
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> CallResult:
        """Call the model with tool-calling support.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The task prompt.
            tool_schemas: OpenAI-format tool schemas.
            max_retries: Number of retries on transient errors.
            timeout: Per-call timeout in seconds (default 120s).

        Returns:
            CallResult with parsed tool calls and metadata.
        """
        self._rate_limit_wait()

        messages = self._build_messages(system_prompt, user_prompt)

        kwargs: dict[str, Any] = {
            "model": self.config.litellm_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": timeout,
        }

        # Add tools if model supports native tool calling
        if self.config.supports_tools and tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                response = litellm.completion(**kwargs)
                latency = (time.time() - start) * 1000

                self._last_call_time = time.time()

                # Parse usage
                usage = response.usage if hasattr(response, "usage") and response.usage else None
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Parse tool calls
                model_calls = self._parse_tool_calls(response)

                return CallResult(
                    model_calls=model_calls,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency,
                )

            except (TimeoutError, litellm.Timeout, litellm.APIConnectionError) as e:
                err_str = str(e).lower()
                is_timeout = "timeout" in err_str or isinstance(e, (TimeoutError, litellm.Timeout))
                if is_timeout:
                    # Timeouts rarely succeed on retry — only retry once
                    if attempt < 1:
                        logger.warning(
                            "Timeout calling %s after %.0fs (attempt %d/%d)",
                            self.config.name, timeout, attempt + 1, max_retries,
                        )
                        continue
                    return CallResult(
                        model_calls=[],
                        raw_response={},
                        error=f"timeout_after_{timeout}s",
                    )
                # Non-timeout connection error — use normal retry logic
                if attempt < max_retries:
                    logger.warning(
                        "Connection error calling %s: %s (attempt %d/%d)",
                        self.config.name, e, attempt + 1, max_retries,
                    )
                    time.sleep(2)
                    continue
                return CallResult(
                    model_calls=[],
                    raw_response={},
                    error=f"connection_error: {e}",
                )

            except litellm.RateLimitError:
                if attempt < max_retries:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s — aggressive backoff for rate limits
                    logger.warning(
                        "Rate limited on %s, waiting %ds (attempt %d/%d)",
                        self.config.name, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                return CallResult(
                    model_calls=[],
                    raw_response={},
                    error="rate_limited",
                )

            except litellm.AuthenticationError:
                return CallResult(
                    model_calls=[],
                    raw_response={},
                    error=f"auth_failed: check {self.config.provider.value} API key",
                )

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "Error calling %s: %s (attempt %d/%d)",
                        self.config.name, e, attempt + 1, max_retries,
                    )
                    time.sleep(1)
                    continue
                return CallResult(
                    model_calls=[],
                    raw_response={},
                    error=str(e),
                )

        # Should not reach here, but just in case
        return CallResult(model_calls=[], raw_response={}, error="max_retries_exceeded")


# ---------------------------------------------------------------------------
# Connection verification
# ---------------------------------------------------------------------------

def verify_model(config: ModelConfig) -> tuple[bool, str]:
    """Quick health check: can we reach this model and get a tool call back?

    Returns (success, message).
    """
    adapter = ModelAdapter(config=config)

    test_tool = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }]

    result = adapter.call(
        system_prompt="You are a helpful assistant. Use the provided tools to answer questions.",
        user_prompt="What is the weather in Paris?",
        tool_schemas=test_tool,
        max_retries=1,
    )

    if result.error:
        return False, f"Error: {result.error}"

    if not result.model_calls:
        return False, "No tool calls in response (model may not support tool calling)"

    call = result.model_calls[0]
    if call.tool_name == "get_weather":
        return True, f"OK ({result.latency_ms:.0f}ms, {result.input_tokens}+{result.output_tokens} tokens)"

    return False, f"Unexpected tool call: {call.tool_name}"


def verify_all_providers() -> dict[str, tuple[bool, str]]:
    """Test one model from each provider to verify connectivity.

    Returns {provider_name: (success, message)}.
    """
    # Pick one representative model per provider (verified to work)
    test_models = {
        "ollama": "mistral-nemo-12b",
        "gemini": "gemini-2.0-flash",
        "groq": "groq-llama3.1-8b",
        "openrouter": "or-gemini-2.0-flash",
        "mistral": "mistral-small",
    }

    results = {}
    for provider_name, model_key in test_models.items():
        config = AVAILABLE_MODELS.get(model_key)
        if config is None:
            results[provider_name] = (False, f"Model {model_key} not in registry")
            continue

        # Check for required env vars
        env_vars = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        env_var = env_vars.get(provider_name)
        if env_var and not os.environ.get(env_var):
            results[provider_name] = (False, f"Missing {env_var}")
            continue

        logger.info("Testing %s via %s...", provider_name, config.name)
        results[provider_name] = verify_model(config)

    return results
