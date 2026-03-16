# Databricks notebook source
# MAGIC %md
# MAGIC # ROBUST04 BM25+RM3 + Neural Reranking (Triplet Loss)
# MAGIC
# MAGIC **Final Project Part A - Run 3 (Neural Method)**
# MAGIC
# MAGIC **Architecture:**
# MAGIC 1. **BM25 + RM3** retrieves 1000 docs (from Run 2)
# MAGIC 2. **Custom Neural Reranker** trained with triplet loss
# MAGIC 3. **Ensemble fusion** of BM25 + neural scores
# MAGIC
# MAGIC **Key differences from friend's code:**
# MAGIC - Adapted for Databricks 
# MAGIC - Streamlined for faster execution
# MAGIC - Uses pre-computed embeddings or computes on-the-fly

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install Dependencies
# MAGIC
# MAGIC **Strategy:** Use Databricks system packages (numpy, torch already installed)

# COMMAND ----------

# Only install what's missing - use system numpy and torch
%pip install sentence-transformers -q
%pip install faiss-cpu -q
%pip install pyserini==0.21.0 -q
%pip install huggingface-hub #==0.20.0 -q
print("✅ Dependencies installed (using system numpy & torch)")

# Restart Python to load new packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
QUERIES_PATH = "queriesROBUST.txt"
QRELS_PATH = "qrels_50_Queries"
OUTPUT_DIR = "runs/"

# BM25 + RM3 parameters (from Run 2)
BM25_K1 = 0.6
BM25_B = 0.4
RM3_FB_TERMS = 20
RM3_FB_DOCS = 15
RM3_ORIGINAL_QUERY_WEIGHT = 0.5

# Neural model configuration
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
HIDDEN_DIM = 32

# Training configuration
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
MARGIN = 1.0

# Fusion weights (tune on train set)
BM25_WEIGHT = 0.35
MODEL_WEIGHT = 0.65

# Retrieval
RETRIEVAL_K = 1000
RUN_TAG = "bm25rm3_neural"

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Imports & Setup

# COMMAND ----------

import subprocess
import os

# Set Java environment
os.environ['JAVA_HOME'] = '/usr/lib/jvm/zulu17-ca-amd64'
os.environ['JVM_PATH'] = '/usr/lib/jvm/zulu17-ca-amd64/lib/server/libjvm.so'

# Imports
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Index & Data

# COMMAND ----------

# Load ROBUST04 index
print("Loading ROBUST04 index...")
searcher = LuceneSearcher.from_prebuilt_index('robust04')
index_reader = IndexReader.from_prebuilt_index('robust04')
print(f"✅ Index loaded: {index_reader.stats()}")

# Load queries
def load_queries(queries_path):
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                qid = parts[0].strip()
                query_text = parts[1].strip()
                queries[qid] = query_text
    return queries

# Load qrels
def load_qrels(qrels_path):
    qrels = defaultdict(dict)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                qrels[qid][doc_id] = relevance
    return qrels



# COMMAND ----------

all_queries = load_queries(QUERIES_PATH)
qrels = load_qrels(QRELS_PATH)
train_qids = list(qrels.keys())

train_queries = {qid: all_queries[qid] for qid in train_qids if qid in all_queries}
test_qids = [qid for qid in all_queries.keys() if qid not in train_qids]
test_queries = {qid: all_queries[qid] for qid in test_qids}

print(f"✅ Queries: {len(train_queries)} train, {len(test_queries)} test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load Sentence Transformer

# COMMAND ----------

print(f"Loading {SENTENCE_TRANSFORMER_MODEL}...")

# Try loading with trust_remote_code (needed for some models)
try:
    sentence_transformer = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL,
        device=device,
        trust_remote_code=True
    )
    sentence_transformer.eval()
    print("✅ Sentence transformer loaded")
except Exception as e:
    print(f"⚠️ Error loading {SENTENCE_TRANSFORMER_MODEL}: {e}")
    print("\nTrying alternative model: all-MiniLM-L6-v2 (smaller, faster)...")
    
    # Fallback to a smaller model
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Update embedding dimension
    
    sentence_transformer = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL,
        device=device
    )
    sentence_transformer.eval()
    print(f"✅ Loaded alternative model: {SENTENCE_TRANSFORMER_MODEL}")
    print(f"   Note: Embedding dim changed to {EMBEDDING_DIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Triplets from Qrels

# COMMAND ----------

print("Generating training triplets...")

# Create DataFrame from qrels
df = pd.DataFrame([
    {'query_id': qid, 'doc_id': doc_id, 'relevance': rel}
    for qid, docs in qrels.items()
    for doc_id, rel in docs.items()
])

# Generate triplets
triplets = []
grouped = df.groupby("query_id")

for query_id, group in tqdm(grouped, desc="Creating triplets"):
    positives = group[group["relevance"] == 1]["doc_id"].tolist()
    negatives = group[group["relevance"] == 0]["doc_id"].tolist()
    
    for pos in positives:
        for neg in negatives:
            triplets.append((query_id, pos, neg))

triplets_df = pd.DataFrame(triplets, columns=["query_id", "positive_doc_id", "negative_doc_id"])

print(f"✅ Generated {len(triplets_df)} triplets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Encode Queries (Train Set Only)

# COMMAND ----------

print("Encoding train queries...")

query_embeddings = {}

for qid, query_text in tqdm(train_queries.items(), desc="Encoding queries"):
    embedding = sentence_transformer.encode(query_text, convert_to_tensor=True, device=device)
    query_embeddings[qid] = embedding

print(f"✅ Encoded {len(query_embeddings)} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Encode Documents (On-Demand)
# MAGIC
# MAGIC **Strategy:** Encode documents on-the-fly during training to save memory.
# MAGIC For 528K docs, full pre-encoding would need ~400GB RAM!

# COMMAND ----------

# Document cache (will store embeddings as we need them)
doc_embedding_cache = {}

def get_doc_embedding(doc_id):
    """Get or compute document embedding."""
    if doc_id in doc_embedding_cache:
        return doc_embedding_cache[doc_id]
    
    try:
        # Get document text
        doc = searcher.doc(doc_id)
        doc_data = json.loads(doc.raw())
        doc_text = doc_data.get('contents', '')[:1000]  # First 1000 chars
        
        # Encode
        embedding = sentence_transformer.encode(doc_text, convert_to_tensor=True, device=device)
        
        # Cache
        doc_embedding_cache[doc_id] = embedding
        
        return embedding
    except Exception as e:
        # Return zero embedding if doc not found
        embedding_dim = sentence_transformer.get_sentence_embedding_dimension()
        return torch.zeros(embedding_dim, device=device)

print("✅ Document encoding function ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Neural Ranker Model

# COMMAND ----------

# Get actual embedding dimension from loaded model
ACTUAL_EMBEDDING_DIM = sentence_transformer.get_sentence_embedding_dimension()
print(f"Model embedding dimension: {ACTUAL_EMBEDDING_DIM}")

class RankSBert(nn.Module):
    """
    Neural ranking model with triplet loss.
    Maps embeddings to lower dimension for ranking.
    """
    def __init__(self, embedding_dim, hidden_dim=32, margin=1.0):
        super(RankSBert, self).__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Projection network
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
    
    def forward(self, query_embedding, positive_embedding, negative_embedding):
        """Compute triplet loss."""
        query_out = self.fc(query_embedding)
        positive_out = self.fc(positive_embedding)
        negative_out = self.fc(negative_embedding)
        
        loss = self.triplet_loss(query_out, positive_out, negative_out)
        return loss
    
    def encode(self, embedding):
        """Project embedding to ranking space."""
        return self.fc(embedding)

# Initialize model with actual embedding dimension
rank_sbert = RankSBert(
    embedding_dim=ACTUAL_EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    margin=MARGIN
).to(device)

optimizer = optim.Adam(rank_sbert.parameters(), lr=LEARNING_RATE)

print("✅ Neural ranker initialized")
print(f"   Input dim: {ACTUAL_EMBEDDING_DIM}, Output dim: {HIDDEN_DIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Triplet Dataset

# COMMAND ----------

class TripletDataset(Dataset):
    """Dataset that loads embeddings on-demand."""
    def __init__(self, triplets_df, query_embeddings):
        self.triplets_df = triplets_df
        self.query_embeddings = query_embeddings
    
    def __len__(self):
        return len(self.triplets_df)
    
    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]
        
        query_id = row["query_id"]
        positive_doc_id = row["positive_doc_id"]
        negative_doc_id = row["negative_doc_id"]
        
        # Get embeddings
        query_emb = self.query_embeddings[query_id]
        positive_emb = get_doc_embedding(positive_doc_id)
        negative_emb = get_doc_embedding(negative_doc_id)
        
        return query_emb, positive_emb, negative_emb

# Create dataset
dataset = TripletDataset(triplets_df, query_embeddings)

# Split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Dataset ready: {train_size} train, {val_size} val")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Training Loop

# COMMAND ----------

print(f"🚀 Training for {NUM_EPOCHS} epochs...")

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Training
    rank_sbert.train()
    total_train_loss = 0.0
    
    for query, positive, negative in tqdm(train_loader, desc="Training"):
        query = query.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        loss = rank_sbert(query, positive, negative)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    rank_sbert.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for query, positive, negative in tqdm(val_loader, desc="Validation"):
            query = query.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            loss = rank_sbert(query, positive, negative)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("\n✅ Training complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Encode All Queries for Inference

# COMMAND ----------

print("Encoding all queries for inference...")

all_query_embeddings = {}

for qid, query_text in tqdm(all_queries.items(), desc="Encoding"):
    embedding = sentence_transformer.encode(query_text, convert_to_tensor=True, device=device)
    all_query_embeddings[qid] = embedding

print(f"✅ Encoded {len(all_query_embeddings)} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Neural Reranking Function

# COMMAND ----------

def rank_with_neural_fusion(query_id, query_text, bm25_weight, model_weight, k=1000):
    """
    Two-stage ranking with neural fusion.
    
    1. BM25+RM3 retrieves candidates
    2. Neural model reranks
    3. Fuse BM25 + neural scores
    """
    # Stage 1: BM25 + RM3 retrieval
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)
    searcher.set_rm3(
        fb_terms=RM3_FB_TERMS,
        fb_docs=RM3_FB_DOCS,
        original_query_weight=RM3_ORIGINAL_QUERY_WEIGHT
    )
    
    hits = searcher.search(query_text, k=k)
    
    if not hits:
        return []
    
    # Get BM25 scores
    bm25_scores = [hit.score for hit in hits]
    
    # Stage 2: Neural reranking
    query_emb = all_query_embeddings[query_id]
    query_proj = rank_sbert.encode(query_emb)
    
    similarities = []
    for hit in hits:
        doc_emb = get_doc_embedding(hit.docid)
        doc_proj = rank_sbert.encode(doc_emb)
        similarity = F.cosine_similarity(query_proj, doc_proj, dim=0).item()
        similarities.append(similarity)
    
    # Normalize scores (z-score)
    bm25_mean = np.mean(bm25_scores)
    bm25_std = np.std(bm25_scores) if np.std(bm25_scores) > 0 else 1.0
    norm_bm25 = [(s - bm25_mean) / bm25_std for s in bm25_scores]
    
    sim_mean = np.mean(similarities)
    sim_std = np.std(similarities) if np.std(similarities) > 0 else 1.0
    norm_sim = [(s - sim_mean) / sim_std for s in similarities]
    
    # Fuse scores
    results = []
    for i, hit in enumerate(hits):
        fused_score = bm25_weight * norm_bm25[i] + model_weight * norm_sim[i]
        results.append((hit.docid, fused_score))
    
    # Sort by fused score
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

print("✅ Neural fusion function ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. MAP Evaluation

# COMMAND ----------

def calculate_ap(ranked_list, qrels_for_query):
    """Calculate Average Precision."""
    if not qrels_for_query:
        return 0.0
    
    num_relevant = sum(1 for rel in qrels_for_query.values() if rel > 0)
    if num_relevant == 0:
        return 0.0
    
    num_relevant_retrieved = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(ranked_list, start=1):
        if doc_id in qrels_for_query and qrels_for_query[doc_id] > 0:
            num_relevant_retrieved += 1
            precision_at_i = num_relevant_retrieved / i
            precision_sum += precision_at_i
    
    if num_relevant_retrieved == 0:
        return 0.0
    
    return precision_sum / num_relevant

def calculate_map(retrieval_results, qrels):
    """Calculate Mean Average Precision."""
    ap_scores = []
    
    for qid in retrieval_results:
        if qid not in qrels:
            continue
        
        ranked_list = [doc_id for doc_id, _ in retrieval_results[qid]]
        ap = calculate_ap(ranked_list, qrels[qid])
        ap_scores.append(ap)
    
    if not ap_scores:
        return 0.0
    
    return sum(ap_scores) / len(ap_scores)

print("✅ Evaluation ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Test on Train Set

# COMMAND ----------

print("🧪 Testing neural fusion on train set...")

rank_sbert.eval()

test_results = {}
for qid, query_text in tqdm(train_queries.items(), desc="Testing"):
    test_results[qid] = rank_with_neural_fusion(
        qid, query_text,
        bm25_weight=BM25_WEIGHT,
        model_weight=MODEL_WEIGHT,
        k=RETRIEVAL_K
    )

test_map = calculate_map(test_results, qrels)

print(f"\n{'='*60}")
print(f"📊 NEURAL FUSION TEST RESULTS")
print(f"{'='*60}")
print(f"Fusion weights: BM25={BM25_WEIGHT}, Neural={MODEL_WEIGHT}")
print(f"Train MAP: {test_map:.4f}")
print(f"\nComparison:")
print(f"  BM25 baseline: 0.2475")
print(f"  BM25+RM3: 0.2713")
print(f"  Neural Fusion: {test_map:.4f}")
print(f"  Gain over BM25: {((test_map - 0.2475) / 0.2475 * 100):+.2f}%")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Generate Run 3 - Full Test Set

# COMMAND ----------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Generating Run 3: BM25+RM3 + Neural Reranking")
print(f"   Processing all {len(all_queries)} queries...\n")

rank_sbert.eval()

all_results = {}
for qid, query_text in tqdm(all_queries.items(), desc="Run 3"):
    all_results[qid] = rank_with_neural_fusion(
        qid, query_text,
        bm25_weight=BM25_WEIGHT,
        model_weight=MODEL_WEIGHT,
        k=RETRIEVAL_K
    )

# Save TREC format
def save_trec_format(retrieval_results, output_path, run_tag):
    with open(output_path, 'w') as f:
        for qid in sorted(retrieval_results.keys(), key=int):
            ranked_docs = retrieval_results[qid]
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n"
                f.write(line)
    print(f"✅ Saved: {output_path}")

run3_path = os.path.join(OUTPUT_DIR, "run_3.res")
save_trec_format(all_results, run3_path, RUN_TAG)

print(f"\n📊 Run 3 Statistics:")
print(f"   Total queries: {len(all_results)}")
print(f"   Avg docs per query: {sum(len(docs) for docs in all_results.values()) / len(all_results):.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Summary

# COMMAND ----------

train_results_final = {qid: all_results[qid] for qid in train_queries.keys()}
train_map_final = calculate_map(train_results_final, qrels)

print(f"""
{'='*60}
🎉 NEURAL RERANKING COMPLETE!
{'='*60}

📊 Results Summary:

Architecture:
   • Stage 1: BM25 + RM3 retrieval
   • Stage 2: Custom neural reranker (triplet loss)
   • Stage 3: Score fusion (BM25={BM25_WEIGHT}, Neural={MODEL_WEIGHT})

Performance:
   • Train MAP: {train_map_final:.4f}
   • BM25 baseline: 0.2475
   • Improvement: {((train_map_final - 0.2475) / 0.2475 * 100):+.2f}%

Training:
   • Model: {SENTENCE_TRANSFORMER_MODEL}
   • Triplets: {len(triplets_df):,}
   • Epochs: {NUM_EPOCHS}
   • Hidden dim: {HIDDEN_DIM}

📁 Output:
   • {run3_path}

🔬 Method Innovation:
   This approach combines classical IR (BM25+RM3) with
   modern neural ranking (triplet loss), similar to
   production search systems!

{'='*60}
""")

