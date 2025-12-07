import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from iswd753 import clean_text, remove_stopwords, stemming, filter_tokens

# ===================================================================
# 1) CARGA Y PREPROCESAMIENTO — IGUAL QUE TU SISTEMA IR
# ===================================================================
df = pd.read_csv("data/bbc_news.csv")
docs_raw = df["description"]

def preprocess(x):
    return filter_tokens(stemming(remove_stopwords(clean_text(x))))

docs = docs_raw.apply(preprocess)

# ===================================================================
# 2) EXTRAER 10 QUERIES (docs reales)
# ===================================================================
NUM_QUERIES = 10
indexes = random.sample(range(len(docs)), NUM_QUERIES)

queries = {}
for i, idx in enumerate(indexes):
    qid = f"Q{i+1}"
    queries[qid] = docs[idx]

# ===================================================================
# 3) TF-IDF para seleccionar documentos relevantes AUTOMÁTICOS
# ===================================================================
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)

# k vecinos relevantes
k = 20
knn = NearestNeighbors(n_neighbors=k, metric="cosine")
knn.fit(tfidf)

# ===================================================================
# 4) Generar qrels.txt (docIDs correctos: 0..N-1)
# ===================================================================
qrels_lines = []

for qid, qtext in queries.items():
    q_vec = vectorizer.transform([qtext])
    dist, idxs = knn.kneighbors(q_vec)

    for docid in idxs[0]:
        qrels_lines.append(f"{qid} 0 {docid} 1")

with open("src/qrels.txt", "w", encoding="utf8") as f:
    f.write("\n".join(qrels_lines))

print("✔ qrels.txt generado correctamente")

# ===================================================================
# 5) Generar queries.txt
# ===================================================================
with open("src/queries.txt", "w", encoding="utf8") as f:
    for qid, text in queries.items():
        f.write(f"{qid} {text}\n")

print("✔ queries.txt generado correctamente")
