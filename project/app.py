from flask import Flask
from flask import request
from flask import render_template

import pickle
import os
import json
import pandas as pd
from preprocessing import *
from search_by_inverted_index import *
from w2v_d2v_functions import *

app = Flask(__name__)

data = pd.read_csv('avito_data.tsv', sep='\t', index_col=0)
print('______data ready______')

with open('inverted_index.json', 'r', encoding='utf-8') as f:
    inverted_index = json.load(f)
    print('______inverted index ready______')

with open('document_length.json', 'r', encoding='utf-8') as f:
    document_length = json.load(f)
    print('______document length ready______')
    
with open('w2v_base.pkl', 'rb') as f:
    w2v_base = pickle.load(f)
    print('______w2v_base ready______')
    
with open('d2v_model.pkl', 'rb') as f:
    d2v_model = pickle.load(f)
    print('______d2v_model ready______')
    
with open('d2v_base.pkl', 'rb') as f:
    d2v_base = pickle.load(f)
    print('______d2v_base ready______')
    
model_path = '../araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model'

w2v_model = FastText.load(model_path)
w2v_model.init_sims(replace=True)
print('______w2v_model ready______')

def search(query, search_method, inverted_index, data, 
           document_length, w2v_model, w2v_base,
           d2v_model, d2v_base, n_results=5):

    query = preprocessing(query, del_stopwords=False)
    
    if search_method == ['inverted_index', 'word2vec', 'doc2vec']:
        res_inv_ind = get_search_result(query, inverted_index, data['corpus'], document_length, n_results * 50,
                                        return_sim=True)
        res_w2v = search_w2v(query, w2v_model, w2v_base, n_results * 50, return_sim=True)
        res_d2v = search_d2v(query, d2v_model, d2v_base, n_results * 50, return_sim=True)
        combination = res_inv_ind + res_w2v + res_d2v
        
        search_result = [index for index, _ in sorted(combination, key=lambda x: x[1], reverse=True)[:n_results]]
    
    elif search_method == ['inverted_index', 'word2vec']:
        res_inv_ind = get_search_result(query, inverted_index, data['corpus'], document_length, n_results * 50,
                                        return_sim=True)
        res_w2v = search_w2v(query, w2v_model, w2v_base, n_results * 50, return_sim=True)
        combination = res_inv_ind + res_w2v
        
        search_result = [index for index, _ in sorted(combination, key=lambda x: x[1], reverse=True)[:n_results]]
        
    elif search_method == ['inverted_index', 'doc2vec']:
        res_inv_ind = get_search_result(query, inverted_index, data['corpus'], document_length, n_results * 50,
                                        return_sim=True)
        res_d2v = search_d2v(query, d2v_model, d2v_base, n_results * 50, return_sim=True)
        combination = res_inv_ind + res_d2v
        
        search_result = [index for index, _ in sorted(combination, key=lambda x: x[1], reverse=True)[:n_results]]
        
    elif search_method == ['word2vec', 'doc2vec']:
        res_w2v = search_w2v(query, w2v_model, w2v_base, n_results * 50, return_sim=True)
        res_d2v = search_d2v(query, d2v_model, d2v_base, n_results * 50, return_sim=True)
        combination = res_w2v + res_d2v
        
        search_result = [index for index, _ in sorted(combination, key=lambda x: x[1], reverse=True)[:n_results]]

    elif search_method == ['inverted_index']:
        search_result = get_search_result(query, inverted_index, data['corpus'], document_length, n_results)
    
    elif search_method == ['word2vec']:
        search_result = search_w2v(query, w2v_model, w2v_base, n_results)
    
    elif search_method == ['doc2vec']:
        search_result = search_d2v(query, d2v_model, d2v_base, n_results)
    
    else:
        raise TypeError('unsupported search method')
    
    results = [(data.loc[index, 'url'], data.loc[index, 'corpus']) for index in search_result]
    return results

@app.route('/',  methods=['GET'])
def index():
    if request.args:
        
        query = request.args['query']
        
        search_method = []
        if request.args.get('inverted_index'):
            search_method.append(request.args['inverted_index'])
        if request.args.get('word2vec'):
            search_method.append(request.args['word2vec'])
        if request.args.get('doc2vec'):
            search_method.append(request.args['doc2vec'])
        
        results = search(query, search_method, inverted_index, data, 
           document_length, w2v_model, w2v_base,
           d2v_model, d2v_base)
        
        return render_template('result.html', results=results, query=query)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost', port=5005, debug=False)
    
    

