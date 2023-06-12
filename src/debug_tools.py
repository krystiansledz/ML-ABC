import time
from contextlib import contextmanager


def measure_time_decorator(func):
    def wrapper(*args, **kwargs):
        start: float = 0.0
        try:
            start = time.time()
            yield func(*args, **kwargs)
        finally:
            end: float = time.time()
            print(f"Time of '{func.__name__}()' execution: {end - start:.2f}s")

    return wrapper


@contextmanager
def measure_time_context(name: str = "computation"):
    start = time.time()
    yield
    end: float = time.time()
    print(f"Time of '{name}' execution: {(end - start):.8f}s")


@contextmanager
def block_interruption():
    try:
        yield
    except KeyboardInterrupt:
        print("Gracefully shutting down...")
