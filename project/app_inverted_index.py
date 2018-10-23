from flask import Flask
from flask import request
from flask import render_template

import os
import json
import pandas as pd
from preprocessing import *
from search_by_inverted_index import *

app = Flask(__name__)

data = pd.read_csv('avito_data.tsv', sep='\t', index_col=0)
print('______data ready______')

with open('inverted_index.json', 'r', encoding='utf-8') as f:
    inverted_index = json.load(f)
    print('______inverted index ready______')

with open('document_length.json', 'r', encoding='utf-8') as f:
    document_length = json.load(f)
    print('______document length ready______')

def search(query, inverted_index, data, 
           document_length, n_results=5):

    query = preprocessing(query, del_stopwords=False)
    
    search_result = get_search_result(query, inverted_index, data['corpus'], document_length, n_results)
    
    results = [(data.loc[index, 'url'], data.loc[index, 'corpus']) for index in search_result]
    return results

@app.route('/',  methods=['GET'])
def index():
    if request.args:
        
        query = request.args['query']
        results = search(query, inverted_index, data, document_length)
        
        return render_template('result.html', results=results, query=query)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5005, debug=True)
    
    

