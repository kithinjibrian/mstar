from typing import Any, Dict

class Register:
    _instance = None
    _registry: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Register, cls).__new__(cls)
        return cls._instance

    def set(self, key: str, agent: Any):
        self._registry[key] = agent

    def get(self, key: str):
        return self._registry.get(key)

    def remove(self, key: str):
        if key in self._registry:
            del self._registry[key]
