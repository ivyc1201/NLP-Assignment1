import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import jieba

def zipf_law(corpus_dir):
    # Read text from Chinese corpus
    corpus_text = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r', encoding='gb18030') as file:
                corpus_text += file.read()

    # Tokenize the text
    words = list(jieba.cut(corpus_text))

    # Calculate word frequencies
    word_freq = Counter(words)

    # Sort word frequencies
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Extract frequencies and ranks
    freq = [item[1] for item in sorted_word_freq]
    rank = np.arange(1, len(freq) + 1)

    # Plot log-log graph for Zipf's Law
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(rank), np.log(freq), marker='o', linestyle='')
    plt.title("Zipf's Law for Chinese Corpus")
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.grid(True)
    plt.show()


corpus_dir = "C:/Users/123/Downloads/jyxstxtqj_downcc.com"
zipf_law(corpus_dir)
