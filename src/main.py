from src.Tokenizing.GPTDatasetV1 import create_dataloader_v1


def main():
    with open("the-verdict.txt", 'r') as file:
        raw_text = file.read()

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

if __name__ == "__main__":
    main()
