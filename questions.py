import string
import math
import os
import nltk
import sys
# import ssl
# from nltk import data
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# data.path += ['/tokenizers/punkt/PY3/english.pickle']

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    new_dir = {}
    for category in os.listdir(directory):
        filename = os.path.join(directory, category)
        with open(filename) as f:
            new_dir[category] = f.read()
    return new_dir
    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.tokenize.word_tokenize(document.lower())
    res = []
    for word in tokens:
        if word not in string.punctuation and word not in set(nltk.corpus.stopwords.words("english")):
            res.append(word)
    return res
    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    for filename in documents:
        words = documents[filename]
        for word in words:
            f = sum(word in documents[filename] for filename in documents)
            idf = math.log(len(documents) / f)
            idfs[word] = idf

    return idfs
    # raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = dict()
    for filename in files:
        value = 0
        for word in query:
            if word in files[filename]:
                tf = files[filename].count(word)
                value += (tf * idfs[word])
        tfidfs[filename] = value
    items = list(dict(sorted(tfidfs.items(), key=lambda x:x[1], reverse=True)[:n]).keys())
    return items
    # raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    term_density = dict()  
    tfidfs = dict()
    for key in sentences:
        term_density[key] = 0
        value = 0
        count = 0
        for word in query:
            if word in sentences[key]:
                value += idfs[word]
                count += 1
        tfidfs[key] = value
        term_density[key] = count/len(key) 

    items = list(dict(sorted(tfidfs.items(), key=lambda x:x[1], reverse=True)).keys())

    for i in range(n-1):
        temp = ''
        if tfidfs[items[i]] > tfidfs[items[i+1]]:
            continue
        elif tfidfs[items[i]] < tfidfs[items[i+1]]:
            temp = items[i]
            items[i] = items[i+1]
            items[i+1] = temp
        else:
            if term_density[items[i]] > term_density[items[i+1]]:
                continue
            elif term_density[items[i]] < term_density[items[i+1]]:
                temp = items[i]
                items[i] = items[i+1]
                items[i+1] = temp
        
    return items[:n]
    # raise NotImplementedError


if __name__ == "__main__":
    main()
