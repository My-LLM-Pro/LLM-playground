import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I am Maohui Li"

tokens = enc.encode(text)
print(tokens)

for token in tokens:
    tranlation = enc.decode([token])
    print(tranlation)