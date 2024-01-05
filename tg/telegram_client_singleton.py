from telethon import TelegramClient

class TelegramClientSingleton:
    _instance = None
    _client = None

    def __new__(cls, api_id, api_hash):
        if cls._instance is None:
            cls._instance = super(TelegramClientSingleton, cls).__new__(cls)
            cls._client = TelegramClient('anon', api_id, api_hash)
        return cls._instance

    @classmethod
    def get_client(cls):
        if cls._client is None:
            raise Exception("TelegramClient has not been initialized. Call __new__ first.")
        return cls._client
