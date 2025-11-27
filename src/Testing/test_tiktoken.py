from importlib.metadata import version
import tiktoken


if __name__ == "__main__":
    print(f"tiktoken version: {version("tiktoken")}")

    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)


    strings = tokenizer.decode(integers)
    print(strings)

    # ================ testing ====================

    word = "Akwirw ier"
    integers = tokenizer.encode(word)
    print(f"ints: {integers}")

    decoded = tokenizer.decode(integers)
    print(f"word: {decoded}")