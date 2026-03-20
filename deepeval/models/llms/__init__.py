import importlib

__all__ = [
    "AzureOpenAIModel",
    "GPTModel",
    "LocalModel",
    "OllamaModel",
    "GeminiModel",
    "AnthropicModel",
    "AmazonBedrockModel",
    "LiteLLMModel",
    "DrBuddyModel",
    "HFModel",
    "LionGuardModel",
    "TogetherModel",
]

_LAZY_IMPORTS = {
    "AzureOpenAIModel": ".azure_model",
    "GPTModel": ".openai_model",
    "LocalModel": ".local_model",
    "OllamaModel": ".ollama_model",
    "GeminiModel": ".gemini_model",
    "AnthropicModel": ".anthropic_model",
    "AmazonBedrockModel": ".amazon_bedrock_model",
    "LiteLLMModel": ".litellm_model",
    "DrBuddyModel": ".drbuddy_model",
    "HFModel": ".hf_model",
    "LionGuardModel": ".lion_guard",
    "TogetherModel": ".together_model",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
