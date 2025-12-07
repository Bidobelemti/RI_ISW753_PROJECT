import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import porter
from nltk.corpus import stopwords
import unicodedata
from collections import Counter

nltk.download('stopwords')
stemmer = porter.PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocesamiento de texto

def clean_text(doc):
    """
    Limpia y normaliza texto: conversión a minúsculas y eliminación de caracteres no alfabéticos.
    ### Input:
             doc: string
    ### Output:
             doc: string

    """
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8').lower()
    
    doc = re.sub(r'[^a-z\s]', ' ', doc)
    
    doc = re.sub(r'\s+', ' ', doc).strip()
    
    return doc

def remove_stopwords(doc):
    """
    Elimina stopwords del documento. Toma la lista de stop_words en inles de NLTK
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    tokens = doc.split()
    return ' '.join(word for word in tokens if word not in stop_words)

def stemming(doc):
    """
    Aplica stemming a todas las palabras del documento usando PorterStemmer.
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    tokens = doc.split()
    return ' '.join(stemmer.stem(word) for word in tokens)

def filter_tokens(doc):
    """
    Filtra tokens por longitud, patrones válidos y estructura de palabras.
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    tokens = doc.split()
    
    valid_tokens = [
        tok for tok in tokens
        if 2 <= len(tok) <= 20  # Longitud válida
        and not re.search(r'(.)\1{2,}', tok)  # Sin 3+ caracteres repetidos
        and sum(c in 'aeiou' for c in tok) > 0  # Al menos una vocal
        and not re.search(r'[^aeiou]{5,}', tok)  # Máximo 4 consonantes consecutivas
    ]
    
    return ' '.join(valid_tokens)

def build_vocabulary(docs):
    """
    Construye vocabulario ordenado alfabéticamente a partir de los documentos.
    ### Input:
             docs: pd.DataFrame
    ### Output:
             vocab: conjunto ordenado del vocabulario de todos los docs  
    """
    return sorted(set(word for doc in docs for word in doc.split()))

# Modelo binario

def jaccard_similarity(vec1, vec2):
    intersection = np.sum(np.logical_and(vec1 == 1, vec2 == 1))
    union = np.sum(np.logical_or(vec1 == 1, vec2 == 1))
    
    if union == 0:
        return 0.0
    
    return intersection / union

# Modelo TFIDF

def get_tf(doc, vocab, vocab_len):

    """
    Calcula el vector TF normalizado de un documento.
    Ignora automáticamente las palabras que no están en el vocabulario.
    
    Parámetros:
        doc (str)        : documento crudo
        vocab (dict)     : diccionario {termino: indice}
        vocab_len (int)  : tamaño del vocabulario
    
    Retorna:
        np.ndarray de tamaño vocab_len con los valores TF.
    """
    tokens = doc.split()
    doc_len = len(tokens)

    # Documento vacío
    if doc_len == 0:
        return np.zeros(vocab_len, dtype=float)

    tf_vec = np.zeros(vocab_len, dtype=float)
    token_counts = Counter(tokens)

    for term, count in token_counts.items():
        idx = vocab.get(term)    # None si no existe
        if idx is not None:      # Solo se asignan términos del vocabulario
            tf_vec[idx] = count / doc_len

    return tf_vec

def get_df(tf):
    """
    Calcula Document Frequency (DF) a partir de matriz TF.
    """
    return np.sum(tf > 0, axis=0)

def get_idf(corpus, dft):
    """
    Calcula Inverse Document Frequency (IDF).
    """
    return np.log(len(corpus) / (dft + 1))

def get_tfidf(tf_mat, idf_vec):
    """
    Calcula matriz TF-IDF.
    """
    return tf_mat * idf_vec

def calculate_cos_similarity(corpus_tfidf, query_tfidf):
    return np.dot(corpus_tfidf, query_tfidf.flatten()) / (
        np.linalg.norm(corpus_tfidf, axis=1) * np.linalg.norm(query_tfidf) + 1e-9
    )

# Modelo okapi BM25

def avgdl(docs_tokenized):
    """
    Calcula la longitud promedio de los documentos en el corpus.
    """
    return np.mean([len(doc) for doc in docs_tokenized])

def basic_index(doc):
    """
    Construye un índice básico con TF y DF.
    """
    tf_ = {i: dict(Counter(word for word in doc_)) for i, doc_ in enumerate(doc)}
    resultado = Counter()
    for term in tf_.values():
        resultado.update(term.keys())
    return tf_, resultado

def idf_rsj(dft_, N):
    """
    Construte un indice idf con RSJ
    """
    idf__ = {
        term: np.log((N - df_t + 0.5) / (df_t + 0.5))
        for term, df_t in dft_.items()
    }
    return idf__

def calculate_scores(docs_tokenizados, avg_dl, tf_, idf_, k=1.5, b=0.75):
    """
    docs_tokenizados : pandas.Series(list())  Un df con N documentos, cada doc contiene una lista de palabras
    avg_dl : np.float()  El promedio de la longitud de los documentos en el corpus
    tf_ : dict{dict{}}  Diccionario que contiene otro diccionario con la frecuencia cruda de terminos en cada documento
    idf_ : dict{term : np.float{}}  IDF de cada termino
    k : 1.5   
    b : 0.75  
    """
    N = len(docs_tokenizados)
    terms = list(idf_.keys())
    V = len(terms)
    dl = np.array([len(doc) for doc in docs_tokenizados])
    K = k * (1 - b + b * (dl / avg_dl))
    scores = np.zeros((N, V), dtype=np.float32)
    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    k_plus_1 = k + 1
    for i in range(N):
        tf_doc = tf_[i]
        K_i = K[i]
        for term, tf in tf_doc.items():
            if term in term_to_idx:
                j = term_to_idx[term]
                scores[i, j] = idf_[term] * (tf * k_plus_1) / (tf + K_i)
    return scores, terms

def bm25_rank_query(query, terms, scores):
    """
    query: string
    terms: lista del vocabulario (en el mismo orden que 'scores')
    scores: matriz NxV con los scores BM25 ya calculados
    top_k: devuelve los K mejores documentos (None= todos)
    """
    query_terms = query.split()
    indices = [terms.index(t) for t in query_terms if t in terms]
    if not indices:
        return [], [] 
    doc_scores = scores[:, indices].sum(axis=1)
    ranking = np.argsort(doc_scores)[::-1]
    doc_scores = (doc_scores - np.min(doc_scores))/(np.max(doc_scores) - np.min(doc_scores))
    return ranking, doc_scores[ranking]

def make_index_inv(docs):
    '''
    Función construye un índice invertido a partir de un DataFrame
    ## docs : pd.DataFrame
       - DataFrame del cual se construirá el índice invertido
    ## return : dict
       - Diccionario que representa el índice invertido
    '''
    # Se crea una estructura de diccionario para el índice invertido que posee un array que almacenará las listas de documentos
    indice_invertido = dict(list=[])
    # docs.items() permite iterar sobre los pares clave-valor del DataFrame
    # Iteramos sobre cada documento en el DataFrame
    for idx, value in docs.items():
        # De cada documento, se extraen las palabras únicas, caracteristica de set para evitar duplicados
        words = set(str(value).split())
        # Iteramos sobre cada palabra única
        for word in words:
            # Si la palabra no está en el índice invertido, se inicializa con una lista vacía
            if word not in indice_invertido:
                indice_invertido[word] = []
            # Se agrega el índice del documento a la lista de la palabra en el índice invertido
            indice_invertido[word].append(idx)
    # Se retorna el índice invertido ordenado por las palabras
    return dict(sorted(indice_invertido.items()))

def get_binary_vector (docs, vocab_map, indice_inv):
    '''
    Función obtiene vector binario a partir de un tf 
    '''
    bin_vector = np.zeros((len(docs),len(vocab_map)), dtype=int) #mat se almacena el vector binario
    for key, docs in indice_inv.items():
        bin_vector[docs, [vocab_map[key]]] = 1
    return bin_vector