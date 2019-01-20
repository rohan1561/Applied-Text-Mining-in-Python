from nltk.corpus import wordnet as wn
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# PART 1: DOCUMENT SIMILARITY


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    # Tokenize the document
    tokens = nltk.word_tokenize(doc)  # returns list of tokens

    # POS tag each token
    tokens = nltk.pos_tag(tokens)

    # Convert token to wordnet.synsets format
    tokens = list(map(lambda x: (x[0], convert_tag(x[1])), tokens))

    # Get synsets of each (token, POS) pair
    synsets = [wn.synsets(word, tag) for (word, tag) in tokens]

    # Get the first synset
    synsets = [synset[0] for synset in synsets if len(synset) > 0]

    return synsets


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    synset_score = []
    for Synset1 in s1:
        scores = [x for x in [Synset1.path_similarity(Synset2) for Synset2 in s2] if x is not None]
        if scores:
            synset_score.append(max(scores))
        # else:
        #     print(scores, Synset1, s2)

    return sum(synset_score)/len(synset_score)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)


print(test_document_path_similarity())


paraphrases = pd.read_csv(r'/home/rohan/Downloads/ML/PythonProjectsSolutions/course4_downloads/paraphrases.csv')
print(paraphrases.head())
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


def most_similar_docs():
    paraphrases['Similarity'] = paraphrases.apply(lambda x: document_path_similarity(x['D1'], x['D2']), axis=1)
    return tuple(list(paraphrases.iloc[paraphrases['Similarity'].values.argmax()])[1:])


print('Most similar docs: ', most_similar_docs())


def label_accuracy():
    paraphrases['labels'] = np.where(paraphrases['Similarity'] > 0.75, 1, 0)
    accuracy = accuracy_score(paraphrases['Quality'], paraphrases['labels'])
    return accuracy


print('accuracy: ', label_accuracy())


# PART 2: TOPIC MODELLING
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# load the list of documents
with open(r'/home/rohan/Downloads/ML/PythonProjectsSolutions/course4_downloads/newsgroups', 'rb') as r:
    newsgroup_data = pickle.load(r)

# Use CountVectorizor to find three letter tokens, remove stop_words,
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')

# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate
# LDA model parameters on the corpus, and save to the variable `ldamodel`

ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=id_map, num_topics=10, passes=25, random_state=34)


def lda_topics():
    topic_words = ldamodel.print_topics(num_topics=10, num_words=10)
    return topic_words


print('LDA Topics from newsgroup_data: ', lda_topics())


new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]


def topic_distribution():
    new_doc_transformed = vect.transform(new_doc)
    corpus = gensim.matutils.Sparse2Corpus(new_doc_transformed, documents_columns=False)
    doc_topics = list(ldamodel.get_document_topics(corpus))
    return doc_topics[0]


print('Topic Distr for new_doc: ', topic_distribution())


def topic_names():
    return ['Education', 'Automobiles', 'Computers & IT', 'Religion', 'Automobiles', 'Sports', 'Science',
            'Society & Lifestyle', 'Computers & IT', 'Science']
