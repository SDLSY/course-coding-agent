from hello import greet


def test_greet() -> None:
    assert greet("benchmark") == "Hello, benchmark!"
