# config.py
import json
from importlib import resources
from llama_github.logger import logger

# utils.py
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Config(metaclass=SingletonMeta):
    _config = None

    def __init__(self):
        if Config._config is None:
            with resources.open_text('llama_github.config', 'config.json') as file:
                Config._config = json.load(file)

    @classmethod
    def get(cls, key, default=None):
        # Ensure the singleton instance is created
        if cls._config is None:
            cls()
        return cls._config.get(key, default)

config = Config()