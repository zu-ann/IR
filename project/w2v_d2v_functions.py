import pickle
import json
import numpy as np
from gensim import matutils
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm_notebook
from preprocessing import *
from judicial_splitter import splitter


def get_w2v_vectors(model, lemmas):
    """Получает вектор документа"""
    vec_list = []
    for word in lemmas:
        try:
            vec = model.wv[word]
            vec_list.append(vec)
        except:
            continue
    try:
        final_vec = sum(vec_list) / len(vec_list)
    except ZeroDivisionError:
        final_vec = [0] * 300
    return final_vec


def save_w2v_base(files_list, model, mystem, save=True, title='w2v_base'):
    """Индексирует всю базу для поиска через word2vec"""
    documents_info = []    
    
    for i, file in enumerate(tqdm_notebook(files_list)):
        if file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = file
            file = i

        lemmas = preprocessing(text)
        vec = get_w2v_vectors(model, lemmas)
            
        file_info = {'file': file, 'word2vec': vec}
        documents_info.append(file_info)
    
    if save:
        with open(title + '.pkl', 'wb') as fw:
            pickle.dump(documents_info, fw)
    
    return documents_info


def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def search_w2v(query, w2v_model, data_word2vec, n_results, return_sim=False):
    vec1 = get_w2v_vectors(w2v_model, query)
    similarity_dict = {}
    
    for elem in data_word2vec:
        sim = similarity(vec1, elem['word2vec'])
        similarity_dict[sim] = elem['file']
    
    if return_sim:
        relevant = [(similarity_dict[sim], sim) for sim in sorted(similarity_dict, reverse=True)[:n_results]]
    else:        
        relevant = [similarity_dict[sim] for sim in sorted(similarity_dict, reverse=True)[:n_results]]
    return relevant


def get_paragraphs(files_list, mystem, del_stopwords=False):
    file_text = {}
    data = []
    
    for i, file in enumerate(files_list):
        if file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                file_text[file] = text
        else:
            text = file
            file = i
        
        paragraphs = splitter(text, 1)
            
        for paragraph in paragraphs:
            paragraph_lemmatized = preprocessing(paragraph, del_stopwords)
            data.append({'file': file, 'paragraph': paragraph_lemmatized})

    if file_text:
        with open('file_text', 'w', encoding='utf-8') as fw:
            json.dump(file_text, fw)
        return data, file_text
    
    else:
        return data

    
def train_doc2vec(data, epochs, save=True, title='d2v_model'):
    tagged_data = [TaggedDocument(words=elem['paragraph'],
                                  tags=[str(i)]) for i, elem in enumerate(data)]
    model = Doc2Vec(vector_size=100, min_count=5, alpha=0.025, 
                min_alpha=0.025, epochs=epochs, workers=4, dm=1)
    
    model.build_vocab(tagged_data)
    print(len(model.wv.vocab))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    if save:
        with open(title + '.pkl', 'wb') as fw:
            pickle.dump(model, fw)
    
    return model


def get_d2v_vectors(model, lemmas):
    """Получает вектор документа"""
    vec = model.infer_vector(lemmas)
    return vec
    

def save_d2v_base(model, paragraphs, save=True, title='d2v_base'):
    """Индексирует всю базу для поиска через doc2vec"""
    documents_info = []    
    
    for paragraph in paragraphs:
        vec = get_d2v_vectors(model, paragraph['paragraph'])
            
        file_info = {'file': paragraph['file'], 'doc2vec': vec}
        documents_info.append(file_info)
    
    if save:
        with open(title + '.pkl', 'wb') as fw:
            pickle.dump(documents_info, fw)
    
    return documents_info 


def search_d2v(query, d2v_model, data_doc2vec, n_results, return_sim=False):
    vec1 = get_d2v_vectors(d2v_model, query)
    similarity_dict = {}
    
    for elem in data_doc2vec:
        sim = similarity(vec1, elem['doc2vec'])
        similarity_dict[sim] = elem['file']
     
    if return_sim:
        relevant = [(similarity_dict[sim], sim) for sim in sorted(similarity_dict, reverse=True)[:n_results]]
    else:    
        relevant = [similarity_dict[sim] for sim in sorted(similarity_dict, reverse=True)[:n_results]]
    return relevant