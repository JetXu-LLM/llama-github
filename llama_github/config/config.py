import json
from importlib.resources import files

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
            config_path = files("llama_github.config").joinpath("config.json")
            Config._config = json.loads(config_path.read_text(encoding="utf-8"))

    @classmethod
    def get(cls, key, default=None):
        if cls._config is None:
            cls()
        return cls._config.get(key, default)

config = Config()
