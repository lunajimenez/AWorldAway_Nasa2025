# type: ignore


class Singleton(type):
    _instances = {}

    def __call__(cls, *args: object, **kwds: object):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwds)
            cls._instances[cls] = instance

        return cls._instances[cls]
