import pandas as pd
import numpy as np
from iswd723 import (
    clean_text, remove_stopwords, stemming, filter_tokens,
    build_vocabulary, jaccard_similarity, get_tf, get_df, get_idf, 
    get_tfidf, calculate_cos_similarity, avgdl, basic_index, idf_rsj,
    calculate_scores, bm25_rank_query, make_index_inv, get_binary_vector
)

def preprocess_docs(docs_raw):
    return docs_raw.apply(clean_text).apply(remove_stopwords).apply(stemming).apply(filter_tokens)

class IRProject:

    def __init__ (self, docs, vocab, indice_inv, model_name):
        self.query = None
        self.docs = docs
        self.vocab = vocab
        self.indice_inv = indice_inv
        self.model = model_name
    
    def rank(self):
        raise NotImplementedError("Este método debe implementarse en la subclase.")
    def setQuery(self, query):
        self.query = query

class JaccardRI(IRProject):
    
    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "jaccard")
        self.binary_matrix = get_binary_vector(self.docs, self.vocab, self.indice_inv)
    
    def rank(self):
        # Preprocesar query
        clean_q = stemming(remove_stopwords(clean_text(self.query)))

        # Vector binario de la query
        self.query_binary_vector = np.zeros(len(self.vocab), dtype=int)
        for word in clean_q.split():
            if word in self.vocab:
                self.query_binary_vector[self.vocab[word]] = 1

        # Calcular scores Jaccard contra cada documento
        self.scores = np.array([
            jaccard_similarity(doc_vector, self.query_binary_vector)
            for doc_vector in self.binary_matrix
        ])

        # Ordenar documentos por similitud
        self.ranked_indices = np.argsort(self.scores)[::-1]

    def getRankedDocs(self):
        # Ya están ordenados en self.ranked_indices
        return self.ranked_indices, self.scores[self.ranked_indices]
    
# =========================================================
if __name__ == "__main__":
    # ------------------ CARGAR CORPUS ------------------
    df = pd.read_csv("data/bbc_news.csv")
    docs_raw = df["description"]
    # ------------------ PREPROCESO ---------------------
    docs = preprocess_docs(docs_raw)
    # ------------------ VOCABULARIO ---------------------
    vocab= build_vocabulary(docs)
    vocab_map = {term: idx for idx, term in enumerate(vocab)}
    # ------------------ ÍNDICE INVERTIDO ------------------
    indice_inv = make_index_inv(docs)
    # ------------------ MODELO JACCARD --------------------
    jaccard_model = JaccardRI(docs, vocab_map, indice_inv)
    # Ejemplo de consulta
    query = "The war between Syria and Iraq leaves thousands wounded"
    jaccard_model.setQuery(query)
    jaccard_model.rank()
    ranked_indices, scores = jaccard_model.getRankedDocs()
    print("Top documentos por Jaccard:")
    for i in range(5):
        print(f"Doc {ranked_indices[i]} - Score {scores[i]:.4f}")