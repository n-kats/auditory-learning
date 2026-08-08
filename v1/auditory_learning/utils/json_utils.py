import json


class Bson:
    def __init__(self, data: bytes):
        self.__data = json.loads(str(data, encoding="utf-8"))

    def __getitem__(self, key):
        return self.__data[key]

    def __setitem__(self, key, value):
        self.__data[key] = value

    def as_dict(self):
        return self.__data

    def as_bytes(self):
        return bytes(json.dumps(self.__data, ensure_ascii=False), encoding="utf-8")
