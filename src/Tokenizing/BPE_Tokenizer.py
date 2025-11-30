import tiktoken


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("../the-verdict.txt", 'r') as file:
        text = file.read()

    enc_text = tokenizer.encode(text)
    print(f"length of the encoded text: {len(enc_text)}")

    enc_sample = enc_text[50:]


    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:      {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

