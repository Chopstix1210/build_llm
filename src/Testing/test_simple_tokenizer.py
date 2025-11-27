from SimpleTokenizer import SimpleTokenizerV1, SimpleTokenizerV2
import re

if __name__ == "__main__": 
    
    with open("../the-verdict.txt", 'r') as file:
        all_text = file.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', all_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token:integer for integer,token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
    
    ids = tokenizer.encode(text)
    print(ids)


    print(tokenizer.decode(ids))

   
    # ==================== Testing SimpleTokenizerV2 ================================

    # need to add some tokens to handle unknown and end of line characters 
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)
 
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)


    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.decode(tokenizer.encode(text)))