import importlib

from deepeval.models.base_model import (
    DeepEvalBaseModel,
    DeepEvalBaseLLM,
    DeepEvalBaseMLLM,
    DeepEvalBaseEmbeddingModel,
)

__all__ = [
    "DeepEvalBaseModel",
    "DeepEvalBaseLLM",
    "DeepEvalBaseMLLM",
    "DeepEvalBaseEmbeddingModel",
    "GPTModel",
    "AzureOpenAIModel",
    "LocalModel",
    "OllamaModel",
    "AnthropicModel",
    "GeminiModel",
    "AmazonBedrockModel",
    "LiteLLMModel",
    "DrBuddyModel",
    "HFModel",
    "LionGuardModel",
    "TogetherModel",
    "MultimodalOpenAIModel",
    "MultimodalOllamaModel",
    "MultimodalGeminiModel",
    "OpenAIEmbeddingModel",
    "AzureOpenAIEmbeddingModel",
    "LocalEmbeddingModel",
    "OllamaEmbeddingModel",
]

_LAZY_IMPORTS = {
    "GPTModel": ".llms",
    "AzureOpenAIModel": ".llms",
    "LocalModel": ".llms",
    "OllamaModel": ".llms",
    "AnthropicModel": ".llms",
    "GeminiModel": ".llms",
    "AmazonBedrockModel": ".llms",
    "LiteLLMModel": ".llms",
    "DrBuddyModel": ".llms",
    "HFModel": ".llms",
    "LionGuardModel": ".llms",
    "TogetherModel": ".llms",
    "MultimodalOpenAIModel": ".mlllms",
    "MultimodalOllamaModel": ".mlllms",
    "MultimodalGeminiModel": ".mlllms",
    "OpenAIEmbeddingModel": ".embedding_models",
    "AzureOpenAIEmbeddingModel": ".embedding_models",
    "LocalEmbeddingModel": ".embedding_models",
    "OllamaEmbeddingModel": ".embedding_models",
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

# TODO: uncomment out once fixed
# from deepeval.models.summac_model import SummaCModels

# TODO: uncomment out once fixed
# from deepeval.models.detoxify_model import DetoxifyModel
# from deepeval.models.unbias_model import UnBiasedModel

# TODO: restructure or delete (if model logic not needed)
# from deepeval.models.answer_relevancy_model import (
#     AnswerRelevancyModel,
#     CrossEncoderAnswerRelevancyModel,
# )
