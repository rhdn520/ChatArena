from ..config import BackendConfig
from .anthropic import Claude
from .base import BACKEND_REGISTRY, IntelligenceBackend, register_backend
from .cohere import CohereAIChat
from .human import Human
from .openai import OpenAIChat


# Load a backend from a config dictionary
def load_backend(config: BackendConfig):
    if config.backend_type == "transformers:conversational":
        from .hf_transformers import TransformersConversational
    try:
        backend_cls = BACKEND_REGISTRY[config.backend_type]
    except KeyError:
        raise ValueError(f"Unknown backend type: {config.backend_type}")

    backend = backend_cls.from_config(config)
    return backend


def __getattr__(name):
    # The legacy conversational pipeline is optional and has been removed from
    # modern Transformers. API games should not import its pipeline stack.
    if name == "TransformersConversational":
        from .hf_transformers import TransformersConversational
        return TransformersConversational
    raise AttributeError(name)
