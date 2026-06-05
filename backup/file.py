import json

path = 'datasets/mobvoi_seq_monkey_general_open_corpus_1000.json'
with open(path, 'r') as f:
    for i, line in enumerate(f):
        if i < 2:
            print(line)
        else:
            break



