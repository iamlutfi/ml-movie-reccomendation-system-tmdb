# 🎬 Movie Recommendation System

I built this project to explore how machine learning can be applied to a domain I interact with every day — deciding what to watch. The central question I wanted to answer: **can a system recommend films I'd actually enjoy, based purely on what a movie *is* rather than what other people think of it?**

This was my first hands-on experience with unsupervised learning and NLP-based similarity, and it reshaped how I think about the line between "training a model" and "building a system."

---

## The Problem It Addresses

Most recommendation systems rely on user behavior — ratings, watch history, interactions. But what if you only have the films themselves? No users, no ratings to learn from.

This project explores a content-only approach: given a movie title, find the most similar films by analyzing what they're actually made of — genres, keywords, cast, director, and plot overview — then surface results that are both relevant and well-regarded.

---

## Approach & Key Decisions

### Why Content-Based Filtering?

The TMDB 5000 dataset contains only film metadata — no user interaction data. Collaborative filtering (which learns from patterns across many users' behavior) wasn't an option here. Content-based filtering was the right choice given the data available, not just the simpler one.

### The Pipeline

```
Movie title input
      ↓
Fuzzy match → tolerant of typos, finds closest title in dataset
      ↓
TF-IDF vectorization → converts combined text features into numerical vectors
      ↓
Cosine similarity → finds top 20 most similar movies
      ↓
Re-rank with weighted rating (IMDB formula) → penalizes low-vote films
      ↓
Final score = (similarity × 0.7) + (weighted_rating × 0.3)
      ↓
Top 10 recommendations with poster
```

### Stemming Decision

I applied NLTK's Porter Stemmer selectively — only to the `overview` field, not to genres, cast, or keywords. Stemming proper nouns or genre labels would have corrupted those features. Applying it only to the plot overview improved similarity scores meaningfully, even if modestly. It was a small but deliberate tradeoff.

---

## What I Learned

The biggest conceptual shift in this project: **cosine similarity doesn't need to be trained**. Coming from a supervised learning background, I expected a fitting/training phase. Instead, the "model" here is just a precomputed matrix — similarity is a mathematical property of the vectors, not something learned from labels.

This changed how I think about the spectrum of ML approaches. Not every problem needs a loss function.

---

## Features

- **Content-based filtering** — similarity based on genres, keywords, cast, director, and overview
- **Fuzzy matching** — typo-tolerant title search via `difflib`
- **Selective NLTK stemming** — applied to overview only, preserving names and genre labels
- **Weighted rating** — IMDB formula to filter films with insufficient votes
- **Movie posters** — fetched live via TMDB API

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| uv | Package manager & environment |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | TF-IDF vectorization & cosine similarity |
| NLTK | Porter Stemmer for text normalization |
| Streamlit | Web application |
| TMDB API | Movie poster retrieval |

---

## Project Structure

```
├── data/
│   ├── raw/                            # Original TMDB 5000 dataset (not tracked)
│   └── preprocessed/                   # Cleaned & processed data (not tracked)
├── model/                              # Saved cosine similarity matrix (not tracked)
├── notebooks/
│   ├── 00_EDA.ipynb                    # Exploratory data analysis
│   ├── 01_Preprocessing.ipynb          # Cleaning, parsing, stemming, feature engineering
│   └── 02_ContentBasedModelling.ipynb  # TF-IDF, cosine similarity, recommendations
├── src/
│   └── app.py                          # Streamlit application
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock                             # Locked dependency versions
└── .gitignore
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv

# Clone the repository
git clone https://github.com/iamlutfi/ml-tmdb-recommended-movie.git
cd ml-tmdb-recommended-movie

# Create virtual environment & install all dependencies
uv sync
```

Create a `.env` file in the root directory:

```
TMDB_API_KEY=your_api_key_here
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place both CSV files in `data/raw/`. Then run the notebooks in order (`00` → `01` → `02`) to generate the preprocessed data and model files.

```bash
# Run the app
uv run streamlit run src/app.py
```

---

## Key Concepts

| Concept | Description |
|---|---|
| TF-IDF | Weighs words by importance across all documents |
| Cosine Similarity | Measures directional similarity between movie vectors |
| Weighted Rating | Balances rating quality with vote count (IMDB formula) |
| Porter Stemmer | Reduces words to base form — applied to overview only |
| Fuzzy Matching | Finds closest title match to handle typos |

---

## What's Next

This project intentionally keeps things content-only. The natural next step is a dataset that includes user interaction data — which opens the door to collaborative filtering and hybrid approaches. I'm exploring this in a follow-up project using the MyAnimeList dataset, where both film metadata and user ratings are available.
