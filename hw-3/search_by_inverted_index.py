import os
import json
from pymystem3.mystem import Mystem
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from string import punctuation, digits
from math import log

punctuation = set(punctuation + '«»—–…“”\n\t' + digits)
mystem = Mystem()


def preprocess_words(mystem, text):
    table = str.maketrans({ch: ' ' for ch in punctuation})
    
    tokenized = word_tokenize(text.replace('\ufeff', '').lower().translate(table))
    return [mystem.lemmatize(word)[0] for word in tokenized], len(tokenized)


def preprocess_files(mystem, file, files_list):
    with open(file, 'r', encoding='utf-8') as f:
        words_list, text_length = preprocess_words(mystem, f.read())
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


def preprocess_query(input_text, operations):
    stack = []
    output = []
    table = str.maketrans({'(': '( ', ')': ' )'})

    query = input_text.lower().translate(table).split()

    for elem in query:
        
        if elem == '(':
            stack.append(elem)
        
        elif elem in operations:
            if len(stack) > 0 and stack[-1] != '(':
                output.append(stack[-1])
                stack[-1] = elem
            else:
                stack.append(elem)
        
        elif elem == ')':
            if stack[-1] == '(':
                continue
            else:
                k = stack[-1]
                while k != '(':
                    output.append(stack.pop(-1))
                    k = stack[-1]
                stack.pop(-1)
    
        else:
            output.append(elem)

    if len(stack) == 1:
        output = output + stack
    elif len(stack) > 1:
        output = output + stack[::-1]

    return output


def boolean_search(input_text, inverted_index, collection_size):
    """
    Produces a Boolean search according with the inverted index
    :return: list of first 5 relevant documents
    """
    relevant_documents = []
    operations = ['&', 'или', 'не']
    query = preprocess_query(input_text, operations)
    
    for i, elem in enumerate(query):
        if elem not in operations:
            relevant_documents.append(elem)
    
        elif elem == 'не':
            a = relevant_documents.pop()
            if type(a) != set:
                a = set(inverted_index[a])
            relevant_documents.append(set(range(collection_size)) - a)
    
        else:
            a = relevant_documents.pop()
            if type(a) != set:
                a = set(inverted_index[a])
        
            b = relevant_documents.pop()
            if type(b) != set:
                b = set(inverted_index[b])

        if elem == '&':
                relevant_documents.append(a & b)
        elif elem == 'или':
            relevant_documents.append(a | b)
    
    return relevant_documents[:5]


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
    doc_list = inverted_index[lemma]
    relevance_score = {}
    avgdl = sum(document_length) / len(document_length)
    N = len(document_length)
    
    for doc in range(N):    
        qf = Counter(inverted_index[lemma])[doc]
        relevance_score[doc] = score_BM25(qf, document_length[doc], avgdl,
                                          2.0, 0.75, N, len(set(inverted_index[lemma])))
    return relevance_score


def get_search_result(query, inverted_index, mystem, files_list, document_length, num_res):
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    relevance_dict = defaultdict(float)
    lemmas, _ = preprocess_words(mystem, query)
    
    for lemma in lemmas:
        score = compute_sim(lemma, inverted_index, document_length)
        for elem in score:
            relevance_dict[elem] += score[elem]
            
    result = sorted(relevance_dict, key=relevance_dict.get, reverse=True)[:num_res]
    
    return [files_list[ind] for ind in result]
