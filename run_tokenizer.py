from src.tokenizer import CharTokenizer

def main():
    text = "hello world!"

    tokenizer = CharTokenizer(text)

    print("Vocab size:", tokenizer.vocab_size)
    print("Vocab:", tokenizer.vocab)

    encoded = tokenizer.encode("hello!")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()