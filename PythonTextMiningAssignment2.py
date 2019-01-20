import nltk
from nltk.stem import WordNetLemmatizer
import os
from nltk.corpus import words
import numpy as np
from collections import Counter

# PART ONE

moby_raw = open(os.getcwd() + '/course4_downloads/moby.txt', 'r').read()
moby_tokens = nltk.word_tokenize(moby_raw)
set_moby_tokens = set(moby_tokens)
text1 = nltk.Text(moby_tokens)


def example_one():
    return len(moby_tokens)


def example_two():
    return len(set_moby_tokens)


def example_three():
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]

    return len(set(lemmatized))


# Lexical diversity
def answer_one():
    answer = len(set_moby_tokens) / len(moby_tokens)
    return answer


# What percentage of tokens is 'whale' or 'Whale'
def answer_two():
    whales = list(filter(lambda x: x == 'whale', moby_tokens)) + list(filter(lambda x: x == 'Whale', moby_tokens))

    return (len(whales) / len(moby_tokens)) * 100


def answer_three():
    frequency = sorted(map(lambda x: (x, moby_tokens.count(x)), set_moby_tokens), key=lambda x: x[1], reverse=True)[:20]
    return frequency


def answer_four():
    frequency = [token for token in set_moby_tokens if len(token) > 5 and moby_tokens.count(token) > 150]
    return sorted(frequency)


def answer_five():
    answer = sorted(text1, key=len, reverse=True)[0]
    return (answer, len(answer))


def answer_six():
    only_words = list(filter(lambda x: x.isalpha(), set_moby_tokens))
    answer = [word for word in only_words if moby_tokens.count(word) > 2000]
    answer = sorted(map(lambda x: (moby_tokens.count(x), x), answer), key=lambda y: y[0], reverse=True)
    return answer


def answer_seven():
    moby_sentences = nltk.sent_tokenize(moby_raw)
    answer = np.array(list(map(lambda x: len(nltk.word_tokenize(x)), moby_sentences)))
    return answer.mean()


def answer_eight():
    counter = Counter()
    tagged_tokens = nltk.pos_tag(moby_tokens)
    for t in tagged_tokens:
        counter[t[1]] += 1
    print(counter)
    return counter.most_common(5)


# PART TWO: Spelling Recommender

correct_spellings = words.words()


def ngrams_maker(word, n):
    trigram_list = []
    for i in range(len(word) - (n - 1)):
        trigram_list.append(tuple(list(word[i:i + n])))
    return trigram_list


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    answer_list = []
    for entry in entries:
        correct_spellings_short = [word for word in correct_spellings if word[0] == entry[0]]
        answer_entry = sorted(
            map(lambda x: (x, nltk.jaccard_distance(set(ngrams_maker(x, 3)), set(ngrams_maker(entry, 3)))),
                correct_spellings_short), key=lambda y: y[1])[0]
        answer_list.append(answer_entry[0])

    return answer_list


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    answers = []
    for entry in entries:
        candidates = [w for w in correct_spellings if w[0] == entry[0]]
        answers.append(min(candidates, key=lambda candidate: nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)),
                                                                                   set(nltk.ngrams(candidate, n=4)))))
    return answers


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    answers = []
    for entry in entries:
        candidates = [w for w in correct_spellings if w[0] == entry[0]]
        answers.append(min(candidates, key=lambda candidate: nltk.edit_distance(entry, candidate)))
    
    return answers

