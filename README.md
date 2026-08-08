# Movie Recommendation System (MovieLens 100k)

A content similarity movie recommender built on the MovieLens 100k dataset,
using **TruncatedSVD** to learn dense movie embeddings from the movie user
rating matrix, and **cosine similarity (KNN)** to find the top-K nearest
movies for a given title.

## Pipeline

- Movie-User Matrix
- SVD
- Movie Embeddings
- Cosine Similarity
- Top K nearest movies


## Project Structure

```
Movie-Recommendation-System/
│
├── data/
│   └── ml-100k/    # dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb    # EDA
│   └── 02_SVD_KNN_Recommender.ipynb    # Rec.System source code
│
├── models/    # Saved model
│   ├── movieEmbeddings.npy
│   ├── movieIds.npy
│   ├── movieIdToTitle.pkl
│   └── knnModel.pkl
│
├── README.md
└── requirements.txt
```

## Setup

1. Create an environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset from [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
   and extract it so the files live at:

   ```
   data/ml-100k/ratings.csv
   data/ml-100k/movies.csv
   data/ml-100k/tags.csv     (not used by the current notebooks, but present)
   data/ml-100k/links.csv    (not used by the current notebooks, but present)
   ```


## Usage

1. Run `notebooks/01_EDA.ipynb` to explore the dataset (user/movie/rating
   counts, sparsity, rating distribution, most-rated movies).
2. Run `notebooks/02_SVD_KNN_Recommender.ipynb` end to end. This builds the
   movie user matrix, fits SVD + KNN, saves the artifacts to `models/`, and
   finishes with an example inference call.

   Matching is fuzzy (via `difflib`), so partial or slightly misspelled
   titles still resolve to the closest movie in the dataset.


## Notes

- `TruncatedSVD` is used instead of full SVD since the movie-user matrix,
  while dense here, mirrors what you'd do on a sparse matrix at larger scale.
- The KNN model uses `metric="cosine"`, so a `NearestNeighbors` search over
  the SVD embeddings is equivalent to ranking by cosine similarity.
