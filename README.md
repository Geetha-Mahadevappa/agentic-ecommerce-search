# Agentic E-Commerce Search
**Hybrid Search • Semantic Understanding • Multi-Agent System**

## 🚀 Overview

This project is a **smart search system for e-commerce data** built to handle messy, real-world conditions.

Unlike traditional search engines that rely on clean product catalogs, this system works even when:
- Product names are generic
- Metadata is missing or inconsistent
- Reviews are noisy
- Product availability differs by country

The system uses **semantic embeddings, hybrid retrieval, and lightweight agents** to understand user intent and return relevant results.

---

## 🔍 What the System Does

Given a vague query like:

> “cheap laptop in France”  
> “premium blender”

The system:
1. Understands what the user is looking for
2. Infers missing details like price intent or country
3. Searches using both **semantic meaning** and **keywords**
4. Reranks results to surface the most relevant products
5. Learns from previous queries during the session

All of this runs locally in Python and is exposed through a **REST API**.

---

## 📦 Dataset

This project uses the Kaggle dataset:

**E-Commerce Purchases and Reviews**  
https://www.kaggle.com/datasets/pruthvirajgshitole/e-commerce-purchases-and-reviews

**Files used:**
- `customer_purchase_data.csv`
- `customer_reviews_data.csv`

The data is intentionally **noisy and incomplete**, which makes it ideal for testing real-world search behavior.

---

## ⚠️ Data Challenges (Why This Is Hard)

The dataset has several problems that shape the system design:

- ❌ No structured product catalog  
- 🔁 Product IDs are reused across countries  
- 🏷 Very generic product names (e.g., “Camera”)  
- 🗣 Short, repetitive, low-signal reviews  
- 🌍 Uneven product availability across countries  

Because of this, traditional keyword search performs poorly.

---

## 🧠 How the System Works (High Level)

The system is built from **three main parts**:

### 1️⃣ Embedding Pipeline

Raw product data is converted into a single **canonical text** per product, combining:
- Product name
- Category
- Country
- Price level (Low / Mid / High)
- Selected review snippets

This text is embedded using a **SentenceTransformer model** and indexed with **FAISS** for fast semantic search.

This step:
- Adds meaning to generic product names
- Reduces noise from bad reviews
- Separates products that share the same ID

---

### 2️⃣ Agentic Search Flow

A small set of Python agents work together to answer each query:

- **Query Understanding Agent**  
  Extracts product type, country, and price intent from the query.

- **Hybrid Retrieval Agent**  
  Combines semantic search (FAISS) with keyword search (BM25) to find candidates.

- **Reranker Agent**  
  Uses an LLM to score and reorder results for relevance.

- **Memory Agent**  
  Remembers things like country or price preference across queries.

This layered approach stabilizes search even when data is incomplete.

---

### 3️⃣ Memory-Aware Search

The system remembers user preferences during a session:
- Country
- Price sensitivity

This allows follow-up queries like:
> “show me cheaper ones”

without needing to restate all constraints.

---

## 📊 Evaluation Results

The system was evaluated using **150 realistic queries** generated from real purchase logs.

```json
{
  "conversion_per_search": 0.80,
  "p99_latency_ms": 1300,
  "ndcg_at_10": 0.26,
  "zero_result_rate": 0.093
}
````

### What This Means

* **80% conversion rate**
  The correct product was surfaced in most searches.

* **~1.3s worst-case latency**
  Acceptable for an MVP with LLM reranking.

* **Low zero-result rate (9.3%)**
  Hybrid retrieval avoids empty searches even with missing data.

---

## ⚙️ Production Notes (Simplified)

* FAISS is used for fast semantic retrieval
* Keyword search improves recall
* LLM reranking improves result quality
* The system can scale horizontally
* Faster cross-encoders can replace the LLM later to reduce latency

---

## 🗂 Project Structure (Simplified)

```bash
agentic-ecommerce-search/
├── agents/            # Query, retrieval, reranking, memory
├── embeddings_pipeline/
├── api.py             # FastAPI server
├── data/              # Raw data and embeddings
├── results/           # Evaluation outputs
├── Dockerfile
├── docker-compose.yml
└── Readme.md
```

---

## ▶️ How to Run

The system runs inside a **GPU-enabled Docker container**.

```bash
docker compose build
docker compose up
```

Once running, access the API at:

```
http://127.0.0.1:8000/docs
```

---

## ✅ Conclusion

This project shows how **semantic embeddings, hybrid retrieval, and agent-based orchestration**
can turn noisy, incomplete e-commerce data into a practical search system.

It is:

* Robust to missing metadata
* Designed for real-world messiness
* Ready for production extension and scaling
```
