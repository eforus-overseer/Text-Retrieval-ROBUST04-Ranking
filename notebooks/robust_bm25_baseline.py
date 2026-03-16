# Databricks notebook source
# MAGIC %md
# MAGIC # ROBUST04 Ranking Competition - BM25 Baseline
# MAGIC
# MAGIC **Final Project Part A - Text Retrieval and Search Engines**
# MAGIC
# MAGIC This notebook implements a BM25 baseline for document ranking on ROBUST04.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Load ROBUST04 index via Pyserini
# MAGIC 2. Parse queries and qrels
# MAGIC 3. Tune BM25 parameters on 50 train queries
# MAGIC 4. Generate rankings for all 249 queries
# MAGIC 5. Output TREC format for submission

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths - UPDATE THESE FOR YOUR DBFS PATHS
QUERIES_PATH = "/dbfs/FileStore/robust04/queriesROBUST.txt"
QRELS_PATH = "/dbfs/FileStore/robust04/qrels_50_Queries"
OUTPUT_DIR = "/dbfs/FileStore/robust04/runs/"

# BM25 parameters to tune
BM25_K1_RANGE = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
BM25_B_RANGE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Default parameters (will be updated after tuning)
BM25_K1_DEFAULT = 0.9
BM25_B_DEFAULT = 0.4

# Retrieval configuration
TOP_K = 1000  # Number of documents to retrieve per query
RUN_TAG = "bm25_baseline"

# Parallel processing
MAX_WORKERS = 20

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install Dependencies

# COMMAND ----------

# Install Pyserini (version compatible with Java 17)
%pip install pyserini==0.21.0 -q

# Restart Python to pick up new packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Java Setup

# COMMAND ----------

import subprocess
import os

# Set Java environment variables
os.environ['JAVA_HOME'] = '/usr/lib/jvm/zulu17-ca-amd64'
os.environ['JVM_PATH'] = '/usr/lib/jvm/zulu17-ca-amd64/lib/server/libjvm.so'

# Verify Java installation
result = subprocess.run(['java', '-version'], capture_output=True, text=True)
print("Java version:")
print(result.stderr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load ROBUST04 Index

# COMMAND ----------

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader

# Load index
print("Loading ROBUST04 index...")
searcher = LuceneSearcher.from_prebuilt_index('robust04')
index_reader = IndexReader.from_prebuilt_index('robust04')

print(f"✅ Index loaded successfully!")
print(f"📊 Index stats: {index_reader.stats()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load Queries and Qrels

# COMMAND ----------

import re
from collections import defaultdict

def load_queries(queries_path):
    """Parse queriesROBUST.txt file."""
    queries = {}
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: "query_id<tab>query_text"
            parts = line.split('\t', 1)
            if len(parts) == 2:
                qid = parts[0].strip()
                query_text = parts[1].strip()
                queries[qid] = query_text
    
    return queries

def load_qrels(qrels_path):
    """
    Parse qrels file.
    Format: query_id 0 doc_id relevance
    """
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

# Load data
print("Loading queries...")
all_queries = load_queries(QUERIES_PATH)
print(f"✅ Loaded {len(all_queries)} queries")

print("\nLoading qrels (train)...")
qrels = load_qrels(QRELS_PATH)
train_qids = list(qrels.keys())
print(f"✅ Loaded qrels for {len(train_qids)} train queries")

# Split train/test
train_queries = {qid: all_queries[qid] for qid in train_qids if qid in all_queries}
test_qids = [qid for qid in all_queries.keys() if qid not in train_qids]
test_queries = {qid: all_queries[qid] for qid in test_qids}

print(f"\n📊 Dataset split:")
print(f"   Train: {len(train_queries)} queries")
print(f"   Test:  {len(test_queries)} queries")
print(f"   Total: {len(all_queries)} queries")

# Show sample
print(f"\n🔍 Sample queries:")
for qid in list(train_queries.keys())[:3]:
    print(f"   {qid}: {train_queries[qid]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. BM25 Retrieval Functions

# COMMAND ----------

def retrieve_bm25(query_text, k1=0.9, b=0.4, k=1000):
    """
    Retrieve documents using BM25.
    
    Args:
        query_text: Query string
        k1: BM25 k1 parameter
        b: BM25 b parameter  
        k: Number of documents to retrieve
        
    Returns:
        List of (doc_id, score) tuples
    """
    searcher.set_bm25(k1=k1, b=b)
    hits = searcher.search(query_text, k=k)
    
    results = [(hit.docid, hit.score) for hit in hits]
    return results

def batch_retrieve(queries_dict, k1=0.9, b=0.4, k=1000, desc="Retrieving"):
    """
    Retrieve documents for multiple queries.
    
    Returns:
        Dict mapping query_id -> [(doc_id, score), ...]
    """
    from tqdm import tqdm
    
    results = {}
    for qid, query_text in tqdm(queries_dict.items(), desc=desc):
        results[qid] = retrieve_bm25(query_text, k1=k1, b=b, k=k)
    
    return results

print("✅ BM25 retrieval functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MAP Evaluation

# COMMAND ----------

def calculate_ap(ranked_list, qrels_for_query):
    """
    Calculate Average Precision for a single query.
    
    Args:
        ranked_list: List of doc_ids in ranked order
        qrels_for_query: Dict mapping doc_id -> relevance (0 or 1)
        
    Returns:
        Average Precision score
    """
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
    """
    Calculate Mean Average Precision across all queries.
    
    Args:
        retrieval_results: Dict mapping qid -> [(doc_id, score), ...]
        qrels: Dict mapping qid -> {doc_id: relevance}
        
    Returns:
        MAP score
    """
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

print("✅ MAP evaluation functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Baseline Performance

# COMMAND ----------

print("🧪 Testing baseline BM25 (k1=0.9, b=0.4) on train set...\n")

# Run baseline
baseline_results = batch_retrieve(
    train_queries, 
    k1=BM25_K1_DEFAULT, 
    b=BM25_B_DEFAULT,
    k=TOP_K,
    desc="Baseline retrieval"
)

# Calculate MAP
baseline_map = calculate_map(baseline_results, qrels)

print(f"\n{'='*60}")
print(f"📊 BASELINE RESULTS")
print(f"{'='*60}")
print(f"BM25 Parameters: k1={BM25_K1_DEFAULT}, b={BM25_B_DEFAULT}")
print(f"MAP on train: {baseline_map:.4f}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Grid Search Parameter Tuning

# COMMAND ----------

from itertools import product
import time

def grid_search_bm25(queries_dict, qrels, k1_range, b_range, top_k=1000):
    """
    Grid search over BM25 parameters.
    
    Returns:
        Tuple of (best_k1, best_b, best_map, results_df)
    """
    import pandas as pd
    from tqdm import tqdm
    
    results = []
    best_map = 0
    best_k1 = None
    best_b = None
    
    param_combinations = list(product(k1_range, b_range))
    
    print(f"🔍 Grid searching {len(param_combinations)} parameter combinations...\n")
    
    for k1, b in tqdm(param_combinations, desc="Grid search"):
        # Retrieve documents
        retrieval_results = batch_retrieve(
            queries_dict, 
            k1=k1, 
            b=b, 
            k=top_k,
            desc=f"k1={k1:.1f}, b={b:.1f}"
        )
        
        # Calculate MAP
        map_score = calculate_map(retrieval_results, qrels)
        
        results.append({
            'k1': k1,
            'b': b,
            'MAP': map_score
        })
        
        if map_score > best_map:
            best_map = map_score
            best_k1 = k1
            best_b = b
    
    results_df = pd.DataFrame(results).sort_values('MAP', ascending=False)
    
    return best_k1, best_b, best_map, results_df

# COMMAND ----------

# Run grid search
print(f"Starting parameter tuning on {len(train_queries)} train queries...")
start_time = time.time()

best_k1, best_b, best_map, tuning_results = grid_search_bm25(
    train_queries,
    qrels,
    BM25_K1_RANGE,
    BM25_B_RANGE,
    top_k=TOP_K
)

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print(f"🎯 PARAMETER TUNING RESULTS")
print(f"{'='*60}")
print(f"Best parameters: k1={best_k1}, b={best_b}")
print(f"Best MAP: {best_map:.4f}")
print(f"Time elapsed: {elapsed:.1f}s")
print(f"{'='*60}\n")

print("Top 10 parameter combinations:")
display(tuning_results.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. TREC Output Format

# COMMAND ----------

def save_trec_format(retrieval_results, output_path, run_tag):
    """
    Save results in TREC format.
    
    Format: qid Q0 doc_id rank score run_tag
    """
    with open(output_path, 'w') as f:
        for qid in sorted(retrieval_results.keys(), key=int):
            ranked_docs = retrieval_results[qid]
            
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n"
                f.write(line)
    
    print(f"✅ Saved TREC format: {output_path}")

print("✅ TREC output function ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Generate Run 1 - Tuned BM25

# COMMAND ----------

import os

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Generating Run 1: Tuned BM25 (k1={best_k1}, b={best_b})")
print(f"   Processing all {len(all_queries)} queries...\n")

# Retrieve for all queries
all_results = batch_retrieve(
    all_queries,
    k1=best_k1,
    b=best_b,
    k=TOP_K,
    desc="Run 1 retrieval"
)

# Save TREC format
run1_path = os.path.join(OUTPUT_DIR, "run_1.res")
save_trec_format(all_results, run1_path, f"bm25_k1_{best_k1}_b_{best_b}")

# Verify output
print(f"\n📊 Run 1 Statistics:")
print(f"   Total queries: {len(all_results)}")
print(f"   Avg docs per query: {sum(len(docs) for docs in all_results.values()) / len(all_results):.1f}")

# Show sample output
print(f"\n🔍 Sample output (first 5 lines):")
with open(run1_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"   {line.strip()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validation Summary

# COMMAND ----------

# Calculate MAP on train set with tuned parameters
print("📊 Validating tuned parameters on train set...\n")

train_results_tuned = {qid: all_results[qid] for qid in train_queries.keys()}
train_map_tuned = calculate_map(train_results_tuned, qrels)

print(f"{'='*60}")
print(f"📈 FINAL VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Method: BM25")
print(f"Parameters: k1={best_k1}, b={best_b}")
print(f"MAP on train (50 queries): {train_map_tuned:.4f}")
print(f"Baseline MAP: {baseline_map:.4f}")
print(f"Improvement: {((train_map_tuned - baseline_map) / baseline_map * 100):+.2f}%")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Analysis: Per-Query Performance

# COMMAND ----------

import pandas as pd

# Calculate AP for each train query
per_query_results = []

for qid in train_queries.keys():
    ranked_list = [doc_id for doc_id, _ in train_results_tuned[qid]]
    ap = calculate_ap(ranked_list, qrels[qid])
    
    per_query_results.append({
        'query_id': qid,
        'query_text': train_queries[qid],
        'AP': ap,
        'num_relevant': sum(1 for rel in qrels[qid].values() if rel > 0),
        'num_retrieved': len(ranked_list)
    })

df_analysis = pd.DataFrame(per_query_results).sort_values('AP')

print("📊 Per-Query Analysis\n")
print("Worst 5 performing queries:")
display(df_analysis.head(5))

print("\nBest 5 performing queries:")
display(df_analysis.tail(5))

print(f"\nAP Statistics:")
print(df_analysis['AP'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Summary & Next Steps

# COMMAND ----------

print(f"""
{'='*60}
🎉 BM25 BASELINE COMPLETE
{'='*60}

📊 Results Summary:
   • Method: BM25 with parameter tuning
   • Best parameters: k1={best_k1}, b={best_b}
   • Train MAP: {train_map_tuned:.4f}
   • Baseline improvement: {((train_map_tuned - baseline_map) / baseline_map * 100):+.2f}%

📁 Output Files:
   • {run1_path}

📈 Next Steps:
   1. Review per-query performance to identify failure modes
   2. Implement Method 2: Query Likelihood Dirichlet (QLD)
   3. Implement Method 3: BM25 + Cross-Encoder Reranking
   4. Compare all methods and select best for submission

💡 Tips for Improvement:
   • Analyze queries with AP < 0.1 - what went wrong?
   • Consider query expansion for short queries
   • Try ensemble methods (combine BM25 + QLD)
   • Implement reranking with cross-encoders

{'='*60}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Download Submission File
# MAGIC
# MAGIC To download the submission file from DBFS:

# COMMAND ----------

# Display file location for download
print("📥 To download your submission file:")
print(f"   File location: {run1_path}")
print(f"\nOption 1: Use Databricks File Browser")
print(f"   Navigate to: FileStore/robust04/runs/")
print(f"\nOption 2: Use dbutils.fs.cp to move to a downloadable location")
print(f"   dbutils.fs.cp('file:{run1_path}', 'dbfs:/FileStore/downloads/run_1.res')")

print("\n✅ Notebook complete! Ready for Methods 2 & 3.")