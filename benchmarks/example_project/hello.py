def greet(name: str) -> str:
    """Return a small greeting used by the offline benchmark fixture."""

    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("benchmark"))
