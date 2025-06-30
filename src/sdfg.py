import typer


def suma(a: int, b: int) -> int:
    return a + b


def main(a: int = 2, b: int = 5):
    suma(a, b)


if __name__ == "__main__":
    typer.run(main)
