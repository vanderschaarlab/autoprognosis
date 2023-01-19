# stdlib
import os

# third party
from optuna.storages import JournalRedisStorage, JournalStorage
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")


class RedisBackend:
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: str = REDIS_PORT,
        auth: bool = False,
    ):
        self.url = f"redis://{host}:{port}/"

        self._optuna_storage = JournalStorage(JournalRedisStorage(url=self.url))
        self._client = redis.Redis.from_url(self.url)

    def optuna(self) -> JournalStorage:
        return self._optuna_storage

    def client(self) -> redis.Redis:
        return self._client
