import pandas as pd
import numpy as np
import sys
import os

from scipy.sparse import csr_matrix

from src.iswd723 import (
    clean_text, remove_stopwords, stemming, filter_tokens,
    build_vocabulary, jaccard_similarity, get_tf, get_df, get_idf, 
    get_tfidf, calculate_cos_similarity, avgdl, basic_index, idf_rsj,
    calculate_scores, bm25_rank_query, make_index_inv, get_binary_vector
)

# ---------------------------------------------------------
# PREPROCESAMIENTO
# ---------------------------------------------------------
def preprocess_docs(docs_raw):
    return docs_raw.apply(clean_text).apply(remove_stopwords).apply(stemming).apply(filter_tokens)

# ---------------------------------------------------------
# CLASE BASE
# ---------------------------------------------------------
class IRProject:
    def __init__(self, docs, vocab, indice_inv, model_name):
        self.query = None
        self.docs = docs
        self.vocab = vocab
        self.indice_inv = indice_inv
        self.model = model_name
    
    def rank(self):
        raise NotImplementedError()

    def setQuery(self, query):
        self.query = query

# ---------------------------------------------------------
# MODELO TF-IDF
# ---------------------------------------------------------
class TFIDFRI(IRProject):

    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "tfidf")

        vocab_len = len(self.vocab)

        # 1) TF LIST (compacto)
        tf_list = [get_tf(doc, self.vocab, vocab_len) for doc in self.docs]

        # 2) Construcción sparse
        rows, cols, data = [], [], []
        for i, vec in enumerate(tf_list):
            nz = np.nonzero(vec)[0]
            rows.extend([i] * len(nz))
            cols.extend(nz)
            data.extend(vec[nz])

        self.tf_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(tf_list), vocab_len),
            dtype=np.float32
        )

        # 3) DF (correcto)
        self.dft = np.asarray(self.tf_matrix.astype(bool).sum(axis=0)).ravel()

        # 4) IDF (CORPUS ES LA SERIE DE DOCS)
        self.idf_vector = get_idf(self.docs, self.dft).astype(np.float32)

        # 5) TF-IDF
        self.tfidf_matrix = self.tf_matrix.multiply(self.idf_vector)

    def rank(self):
        clean_q = stemming(remove_stopwords(clean_text(self.query)))

        query_tf = get_tf(clean_q, self.vocab, len(self.vocab)).astype(np.float32)
        query_tfidf = query_tf * self.idf_vector

        self.scores = self.tfidf_matrix.dot(query_tfidf)
        self.scores = np.asarray(self.scores).ravel()
        self.ranked_indices = np.argsort(-self.scores)

    def getRankedDocs(self):
        return self.ranked_indices, self.scores[self.ranked_indices]

# ---------------------------------------------------------
# MODELO BM25
# ---------------------------------------------------------
class BM25RI(IRProject):
    def __init__(self, docs, vocab, indice_inv, k1=1.5, b=0.75):
        super().__init__(docs, vocab, indice_inv, "bm25")
        self.k1 = k1
        self.b = b

        # Tokenizar
        self.docs_tok = [doc.split() for doc in docs]
        N = len(self.docs_tok)
        # Construir índice básico
        self.tf_raw, self.dft = basic_index(self.docs_tok)
        # Calcular IDF RSJ
        self.idf_bm25 = idf_rsj(self.dft, N)
        # Longitud promedio
        self.avgdl = avgdl(self.docs_tok)

        # calculate_scores
        self.scores, self.terms = calculate_scores(
            self.docs_tok,
            self.avgdl,
            self.tf_raw,
            self.idf_bm25,
            self.k1,
            self.b
        )

    def rank(self):
        clean_q = stemming(remove_stopwords(clean_text(self.query)))
        # Ranking para esta consulta
        self.ranked_indices, self.similarities = bm25_rank_query(
            clean_q,
            self.terms,
            self.scores
        )

    def getRankedDocs(self):
        return self.ranked_indices, self.similarities

# ---------------------------------------------------------
# MODELO JACCARD
# ---------------------------------------------------------
class JaccardRI(IRProject):
    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "jaccard")
        # binary_matrix ya es eficiente
        self.binary_matrix = get_binary_vector(self.docs, self.vocab, self.indice_inv)

    def rank(self):
        clean_q = stemming(remove_stopwords(clean_text(self.query)))

        qv = np.zeros(len(self.vocab), dtype=np.uint8)
        for word in clean_q.split():
            if word in self.vocab:
                qv[self.vocab[word]] = 1
        scores = []
        for i in range(self.binary_matrix.shape[0]):
            row = self.binary_matrix[i]
            scores.append(jaccard_similarity(row, qv))

        self.scores = np.array(scores, dtype=np.float32)
        self.ranked_indices = np.argsort(self.scores)[::-1]

    def getRankedDocs(self):
        return self.ranked_indices, self.scores[self.ranked_indices]

# ---------------------------------------------------------
# PROGRAMA PRINCIPAL — CLI
# ---------------------------------------------------------
if __name__ == "__main__":

    data_path = "data/bbc_news.csv"

    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo '{data_path}'.")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        docs_raw = df["description"]

        print("Preprocesando corpus...")
        docs = preprocess_docs(docs_raw)

        vocab = build_vocabulary(docs)
        vocab_map = {term: idx for idx, term in enumerate(vocab)}
        indice_inv = make_index_inv(docs)

        print("Inicializando modelos con optimización de memoria...")
        jaccard_model = JaccardRI(docs, vocab_map, indice_inv)
        tfidf_model = TFIDFRI(docs, vocab_map, indice_inv)
        bm25_model = BM25RI(docs, vocab_map, indice_inv)
        print("Listo.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    while True:
        print("\n" + "=" * 50)
        print("Sistema IR")

        model_choice = input("Modelo [jaccard | tfidf | bm25] or eval: ").lower().strip()
        if model_choice not in ["jaccard", "tfidf", "bm25", "eval"]:
            print("Modelo inválido.")
            continue
        if model_choice == "eval":
            print("Modo de evaluación no implementado en este snippet.")
            continue
        else:
            query = input("Consulta: ").strip()
            if not query:
                print("Consulta vacía.")
                continue
            try:
                top_k = int(input("Top K: "))
            except:
                print("Top K inválido.")
                continue

            ir_model = {
                "jaccard": jaccard_model,
                "tfidf": tfidf_model,
                "bm25": bm25_model
            }[model_choice]

            ir_model.setQuery(query)
            ir_model.rank()

            indices, scores = ir_model.getRankedDocs()

            print(f"\n--- Top {top_k} resultados ---")
            results_df = pd.DataFrame({
                "Index": df.index[indices[:top_k]],
                "Score": scores[:top_k],
                "Documento": df.iloc[indices[:top_k]]["description"].values,
                "URL": df.iloc[indices[:top_k]]["link"].values,
            })

            print(results_df.to_string(index=False, max_colwidth=60))

            if input("¿Otra consulta? (S/N): ").lower().strip() != "s":
                break

    print("Saliendo...")
