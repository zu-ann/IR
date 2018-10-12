import os
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from string import punctuation, digits
from math import log

punctuation = set(punctuation + '«»—–…“”\n\t' + digits)


def preprocess_words(mystem, text):
    table = str.maketrans({ch: ' ' for ch in punctuation})
    
    tokenized = word_tokenize(text.replace('\ufeff', '').lower().translate(table))
    return [mystem.lemmatize(word)[0] for word in tokenized], len(tokenized)


def preprocess_files(mystem, file, files_list):
    if file.endswith('.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = file
    
    words_list, text_length = preprocess_words(mystem, text)
    document_length[files_list.index(file)] = text_length
    
    return words_list

    
def get_inverted_index(mystem, files_list, save=True):
    """
    Create inverted index by input doc collection and count the length of each document 
    :return: inverted index
    """
    inverted_index = defaultdict(list)
    global document_length
    document_length = [None] * len(files_list)

    for file in files_list:
        for word in preprocess_files(mystem, file, files_list):
            inverted_index[word].append(files_list.index(file))
            
    if save:
        with open('inverted_index.json', 'w', encoding='utf-8') as fw:
            json.dump(inverted_index, fw, ensure_ascii=False)
           
        with open('document_length.json', 'w', encoding='utf-8') as fw:
            json.dump(document_length, fw, ensure_ascii=False)
    
    return inverted_index, document_length


def score_BM25(qf, dl, avgdl, k1, b, N, n):
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    score = log((N - n + 0.5) / (n + 0.5)) * (k1 + 1) * qf / (qf + k1 * (1 - b + b * dl / avgdl))
    return score


def compute_sim(lemma, inverted_index, document_length):
    """
    Compute similarity score between word in search query and all document from collection
    :return: score
    """
    if inverted_index.get(lemma):
        doc_list = inverted_index[lemma]
        relevance_score = {}
        avgdl = sum(document_length) / len(document_length)
        N = len(document_length)
    
        for doc in range(N):    
            qf = Counter(inverted_index[lemma])[doc]
            relevance_score[doc] = score_BM25(qf, document_length[doc], avgdl,
                                          2.0, 0.75, N, len(set(inverted_index[lemma])))
        return relevance_score
    return


def get_search_result(query, inverted_index, files_list, document_length, num_res):
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    relevance_dict = defaultdict(float)
    
    for lemma in query:
        score = compute_sim(lemma, inverted_index, document_length)
        if score:
            for elem in score:
                relevance_dict[elem] += score[elem]    
    result = sorted(relevance_dict, key=relevance_dict.get, reverse=True)[:num_res]
    
    return result
