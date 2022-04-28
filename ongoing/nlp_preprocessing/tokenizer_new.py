import sys, fileinput
from sacremoses import MosesTokenizer
if __name__ == "__main__":
    tokenize = MosesTokenizer(lang='en')
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenize.tokenize(line.strip(), escape=False)
            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')
