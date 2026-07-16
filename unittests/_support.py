import os
import sys
import types
from pathlib import Path


def _install_openai_stub() -> None:
    try:
        import openai  # type: ignore
        return
    except Exception:
        pass

    module = types.ModuleType("openai")

    class _FakeEmbeddingsAPI:
        async def create(self, *args, **kwargs):
            return types.SimpleNamespace(data=[])

    class _FakeCompletionsAPI:
        async def create(self, *args, **kwargs):
            return types.SimpleNamespace(choices=[])

    class _FakeChatAPI:
        def __init__(self):
            self.completions = _FakeCompletionsAPI()

    class AsyncOpenAI:  # noqa: N801
        def __init__(self, *args, **kwargs):
            self.embeddings = _FakeEmbeddingsAPI()
            self.chat = _FakeChatAPI()

    module.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = module


def _install_pydantic_settings_stub() -> None:
    try:
        import pydantic_settings  # type: ignore
        return
    except Exception:
        pass

    module = types.ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, **kwargs):
            annotations = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))

            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    env_value = os.environ.get(name.upper())
                    if env_value is not None:
                        value = env_value
                    else:
                        class_attr = getattr(self.__class__, name, None)
                        value = getattr(class_attr, "default", class_attr)
                setattr(self, name, value)

    module.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = module


def _install_pydantic_stub() -> None:
    try:
        import pydantic  # type: ignore
        return
    except Exception:
        pass

    module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FieldInfo:
        def __init__(self, default=None, **kwargs):
            self.default = default
            self.metadata = kwargs

    def Field(default=None, **kwargs):  # noqa: N802
        return _FieldInfo(default=default, **kwargs)

    module.BaseModel = BaseModel
    module.Field = Field
    sys.modules["pydantic"] = module


def _install_qdrant_stub() -> None:
    try:
        import qdrant_client  # type: ignore
        return
    except Exception:
        pass

    module = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class AsyncQdrantClient:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        async def collection_exists(self, *args, **kwargs):
            return True

        async def create_collection(self, *args, **kwargs):
            return None

        async def upsert(self, *args, **kwargs):
            return None

        async def search(self, *args, **kwargs):
            return []

        async def retrieve(self, *args, **kwargs):
            return []

        async def delete(self, *args, **kwargs):
            return None

        async def get_collection(self, *args, **kwargs):
            return _DummyModel(points_count=0, vectors_count=0, status=_DummyModel(value="green"))

        async def close(self):
            return None

    models.Distance = _DummyModel
    models.VectorParams = _DummyModel
    models.PointStruct = _DummyModel
    models.Filter = _DummyModel
    models.FieldCondition = _DummyModel
    models.MatchValue = _DummyModel
    models.ScoredPoint = _DummyModel

    module.AsyncQdrantClient = AsyncQdrantClient
    module.models = models
    sys.modules["qdrant_client"] = module
    sys.modules["qdrant_client.models"] = models


def configure_test_environment() -> Path:
    root = Path(__file__).resolve().parent.parent
    mcp_serv = root / "mcp-serv"

    _install_openai_stub()
    _install_pydantic_stub()
    _install_pydantic_settings_stub()
    _install_qdrant_stub()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(mcp_serv) not in sys.path:
        sys.path.insert(0, str(mcp_serv))

    defaults = {
        "QDRANT_URL": "http://localhost:6333",
        "OPENAI_API_KEY": "test-openai-key",
        "FAST_LLM_API_KEY": "test-fast-key",
        "STRONG_LLM_API_KEY": "test-strong-key",
        "GEMINI_API_KEY": "test-gemini-key",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    return root