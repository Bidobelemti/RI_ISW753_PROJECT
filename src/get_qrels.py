import os
import numpy as np
import pandas as pd
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from math import log2

# -----------------------
# MÉTRICAS
# -----------------------

def precision_at_k(retr, relset, k):
    if k == 0: return 0
    return sum(1 for d in retr[:k] if d in relset) / k

def recall_at_k(retr, relset, k):
    if len(relset) == 0: return 0
    return sum(1 for d in retr[:k] if d in relset) / len(relset)

def average_precision(retr, relset):
    hits = 0
    s = 0.0
    for i, doc in enumerate(retr, start=1):
        if doc in relset:
            hits += 1
            s += hits / i
    return s / hits if hits > 0 else 0

def reciprocal_rank(retr, relset):
    for i, doc in enumerate(retr, start=1):
        if doc in relset:
            return 1 / i
    return 0

def dcg_at_k(retr, rel, k):
    dcg = 0.0
    for i, d in enumerate(retr[:k], start=1):
        r = rel.get(d, 0)
        dcg += r if i == 1 else r / log2(i)
    return dcg

def ndcg_at_k(retr, rel, k):
    dcg = dcg_at_k(retr, rel, k)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = 0.0
    for i, r in enumerate(ideal, start=1):
        idcg += r if i == 1 else r / log2(i)
    return dcg / idcg if idcg else 0

# -----------------------
# CARGAR CSV
# -----------------------

CSV_PATH = "data/bbc_news.csv"  # AJUSTA ESTO

df = pd.read_csv(CSV_PATH)

if "description" not in df.columns or "title" not in df.columns:
    raise ValueError("El CSV debe tener columnas 'title' y 'description'.")

# Usaremos title + description
df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)

if "id" not in df.columns:
    df = df.reset_index().rename(columns={"index": "id"})

df["id"] = df["id"].astype(str)

# -----------------------
# CLUSTERING PARA QRELS
# -----------------------

print("Vectorizando y generando clusters...")

tfidf = TfidfVectorizer(max_features=6000)
X = tfidf.fit_transform(df["text"])

K = 10  # número de clusters = número de pseudocategorías
km = KMeans(n_clusters=K, random_state=42)
df["cluster"] = km.fit_predict(X)

# qrels por cluster
qrels = {}
relevance_dicts = {}

for c in range(K):
    docs_c = df[df["cluster"] == c]["id"].tolist()
    relset = set(docs_c)
    relevancedict = {d: 1 for d in docs_c}
    cluster_id = f"C{c}"
    qrels[cluster_id] = relset
    relevance_dicts[cluster_id] = relevancedict

# -----------------------
# WHOOSH INDEX
# -----------------------

INDEX_DIR = "whoosh_index"

if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)

schema = Schema(
    id=ID(stored=True, unique=True),
    content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    link=STORED
)

if not index.exists_in(INDEX_DIR):
    ix = index.create_in(INDEX_DIR, schema)
    w = ix.writer()
    for _, row in df.iterrows():
        w.add_document(
            id=row["id"],
            content=row["text"],
            link=row.get("link", "")
        )
    w.commit()
else:
    ix = index.open_dir(INDEX_DIR)

# -----------------------
# GENERAR 10 QUERIES
# -----------------------

queries = {}
sample = df.sample(n=10, random_state=42)

for i, row in sample.iterrows():
    qid = f"Q{i+1}"
    queries[qid] = row["text"]

# -----------------------
# BÚSQUEDA Y EVALUACIÓN
# -----------------------

searcher = ix.searcher(weighting=BM25F())
parser = MultifieldParser(["content"], ix.schema)

retrieved_lists = {}

TOPK = 50
EVAL_K = 10

print("\nEjecutando consultas...")

for qid, qtext in queries.items():
    q = parser.parse(qtext)
    hits = searcher.search(q, limit=TOPK)
    retrieved = [hit["id"] for hit in hits]
    retrieved_lists[qid] = retrieved

print("\nCalculando métricas...\n")

p_list, r_list, ap_list, rr_list, nd_list = [], [], [], [], []

# asociar cada query a su cluster para usar qrels
for qid, qtext in queries.items():
    # obtener cluster del doc original
    original_id = sample[sample["text"] == qtext]["id"].item()
    c = df[df["id"] == original_id]["cluster"].item()
    cluster_id = f"C{c}"

    relset = qrels[cluster_id]
    reldict = relevance_dicts[cluster_id]
    retr = retrieved_lists[qid]

    P = precision_at_k(retr, relset, EVAL_K)
    R = recall_at_k(retr, relset, EVAL_K)
    AP = average_precision(retr, relset)
    RR = reciprocal_rank(retr, relset)
    N = ndcg_at_k(retr, reldict, EVAL_K)

    print(f"{qid} → P@{EVAL_K}={P:.3f}  R@{EVAL_K}={R:.3f}  AP={AP:.3f}  RR={RR:.3f}  nDCG@{EVAL_K}={N:.3f}")

    p_list.append(P)
    r_list.append(R)
    ap_list.append(AP)
    rr_list.append(RR)
    nd_list.append(N)

print("\n=== MÉTRICAS GLOBALES ===")
print(f"Mean P@{EVAL_K}: {np.mean(p_list):.4f}")
print(f"Mean R@{EVAL_K}: {np.mean(r_list):.4f}")
print(f"MAP: {np.mean(ap_list):.4f}")
print(f"MRR: {np.mean(rr_list):.4f}")
print(f"Mean nDCG@{EVAL_K}: {np.mean(nd_list):.4f}")

# guardar qrels
with open("qrels.txt", "w", encoding="utf8") as f:
    for cid, docs in qrels.items():
        for d in docs:
            f.write(f"{cid} 0 {d} 1\n")

print("\nqrels.txt generado.")
