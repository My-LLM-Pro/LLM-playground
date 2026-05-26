import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I love NLP"
tokens = enc.encode(text)

text = "I am Maohui Li"
tokens = enc.encode(text)

print(tokens)
print(len(tokens))