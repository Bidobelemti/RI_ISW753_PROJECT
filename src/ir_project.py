import pandas as pd
import numpy as np
import sys
import os # Importar sys y os para la ejecución CLI y manejo de archivos

from iswd753 import (
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

class TFIDFRI(IRProject):
    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "tfidf")

        vocab_len = len(self.vocab)

        # 1. TF por documento
        self.tf = self.docs.apply(lambda doc: get_tf(doc, self.vocab, vocab_len))

        # 2. Matriz TF
        self.tf_matrix = np.asarray(self.tf.values.tolist())

        # 3. DF y IDF
        self.dft = get_df(self.tf_matrix)
        self.idf_vector = get_idf(self.docs, self.dft)

        # 4. Matriz TF-IDF final
        self.tfidf_matrix = get_tfidf(self.tf_matrix, self.idf_vector)

    def rank(self):
        # Procesamiento limpio idéntico al usado en el corpus
        clean_q = stemming(remove_stopwords(clean_text(self.query)))

        # TF de la query
        query_tf = get_tf(clean_q, self.vocab, len(self.vocab))

        # TF-IDF de la query
        query_tfidf = get_tfidf(query_tf, self.idf_vector)

        # Similaridad coseno
        self.scores = calculate_cos_similarity(self.tfidf_matrix, query_tfidf)

        # Ordenar descendentemente
        self.ranked_indices = np.argsort(-self.scores)

    def getRankedDocs(self):
        return self.ranked_indices, self.scores[self.ranked_indices]

class BM25RI(IRProject):

    def __init__(self, docs, vocab, indice_inv, k1=1.5, b=0.75):
        super().__init__(docs, vocab, indice_inv, "bm25")
        self.k1 = k1
        self.b = b
        # Tokenizar una sola vez
        self.docs_tok = [doc.split() for doc in docs]
        # Construir índice básico
        self.tf_raw, self.dft = basic_index(self.docs_tok)
        # Calcular IDF RSJ
        self.idf_bm25 = idf_rsj(self.dft, len(self.docs_tok))
        # Longitud promedio
        self.avgdl = avgdl(self.docs_tok)
        # Precomputar scores BM25 de todo el corpus
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
        # La función bm25_rank_query normaliza los scores antes de devolverlos
        return self.ranked_indices, self.similarities

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
    # --- 1. CARGA INICIAL Y PREPROCESAMIENTO DEL CORPUS (Solo se hace una vez) ---
    data_path = "data/bbc_news.csv"
    
    # Comprobación de existencia del archivo y Try-Catch inicial
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo de datos en '{data_path}'.")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        docs_raw = df["description"]
        # Preprocesamiento del Corpus
        docs = preprocess_docs(docs_raw) 
        
        # Vocabulario e Índice Invertido
        vocab = build_vocabulary(docs)
        vocab_map = {term: idx for idx, term in enumerate(vocab)}
        indice_inv = make_index_inv(docs)
        
        # Inicialización de Modelos (Se precalculan una sola vez)
        print("Cargando y precalculando modelos de recuperación (BM25, TF-IDF, Jaccard)...")
        jaccard_model = JaccardRI(docs, vocab_map, indice_inv)
        tfidf_model = TFIDFRI(docs, vocab_map, indice_inv)
        bm25_model = BM25RI(docs, vocab_map, indice_inv)
        print("Sistema listo para consultas.")
        
    except Exception as e:
        print(f"Error fatal durante la carga o preprocesamiento. Revise dependencias y archivos: {e}")
        sys.exit(1)


    # --- 2. BUCLE INTERACTIVO DEL CLI ---
    while True:
        try:
            print("\n" + "=" * 50)
            print("Sistema de Recuperación de Información (SRI) Interactivo")
            print("-" * 50)
            
            # 2.1 Obtener Modelo
            model_choice = input("Seleccione un modelo [jaccard, tfidf, bm25]: ").lower().strip()
            
            if model_choice not in ['jaccard', 'tfidf', 'bm25']:
                print("Error: Modelo no válido. Intente de nuevo.")
                continue

            # 2.2 Obtener Consulta
            query = input("Ingrese la consulta de búsqueda: ").strip()
            if not query:
                print("Error: La consulta no puede estar vacía.")
                continue

            # 2.3 Obtener Top K
            top_k_str = input("Ingrese el número de resultados (Top K): ").strip()
            
            try:
                top_k = int(top_k_str)
                if top_k <= 0:
                    raise ValueError
            except ValueError:
                print("Error: Top K debe ser un número entero positivo.")
                continue

            # 2.4 Asignar Modelo y Nombre
            if model_choice == 'jaccard':
                ir_model = jaccard_model
                model_name = "Jaccard (Binario)"
            elif model_choice == 'tfidf':
                ir_model = tfidf_model
                model_name = "Vectorial (TF-IDF)"
            elif model_choice == 'bm25':
                ir_model = bm25_model
                model_name = "Probabilístico (BM25)"
            
            # 2.5 Ejecución del Ranking
            ir_model.setQuery(query)
            ir_model.rank()
            ranked_indices, scores = ir_model.getRankedDocs()

            # 2.6 Visualización de Resultados
            print(f"\n--- Top {top_k} Resultados con {model_name} para: '{query}' ---")
            
            if len(ranked_indices) == 0 or scores[0] == 0:
                print("No se encontraron documentos relevantes para esta consulta.")
            else:
                results_df = pd.DataFrame({
                    'Rank': range(1, top_k + 1),
                    'Score': scores[:top_k],
                    'Documento': df.iloc[ranked_indices[:top_k]]['description'].values,
                    'URL': df.iloc[ranked_indices[:top_k]]['link'].values
                })
                print(results_df.to_string(index=False, max_colwidth=80))

            # --- NUEVO PASO: Preguntar si desea continuar ---
            print("-" * 50)
            continuar = input("¿Desea realizar otra consulta? (S/N): ").lower().strip()
            
            if continuar != 's':
                break # Sale del bucle while True si no es 's'

        except Exception as e:
            # Captura cualquier otro error inesperado (ej. durante el ranking)
            print(f"\nOcurrió un error inesperado. Intente de nuevo: {e}")
            continue

    print("\nSaliendo... ")