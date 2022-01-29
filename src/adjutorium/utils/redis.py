# stdlib
import os

# adjutorium absolute
from adjutorium.utils.pip import install

for retry in range(2):
    # third party
    try:
        # third party
        import optuna
        import redis

        break
    except ImportError:
        depends = ["optuna", "redis"]
        install(depends)


REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")


class RedisBackend:
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: str = "6379",
        auth: bool = False,
    ):
        self.url = f"redis://{host}:{port}/"

        self._optuna_storage = optuna.storages.RedisStorage(url=self.url)
        self._client = redis.Redis.from_url(self.url)

    def optuna(self) -> optuna.storages.RedisStorage:
        return self._optuna_storage

    def client(self) -> redis.Redis:
        return self._client


backend = RedisBackend()
