# Individual Assignment Guidelines

## What you must implement

You are given a project structure with placeholder files. Your task is to implement the missing code and build a working movie or music recommender-system prototype.

## Required components

### 1. Data loading and preprocessing
Implement functions to load ratings/interactions and item metadata. Create a train/test split and compute dataset statistics.

### 2. Exploratory data analysis
Include basic analysis:

- number of users
- number of items
- number of ratings/interactions
- sparsity
- rating distribution
- most active users
- most popular items

### 3. Non-personalized recommenders
Implement at least two:

- most popular
- highest average rating with minimum number of ratings
- random recommender optional

### 4. Content-based recommender
Implement item feature vectors from metadata such as genres, tags, descriptions, or music attributes. Build user profiles from centered ratings and recommend items using cosine similarity.

### 5. Collaborative filtering
Implement item-item collaborative filtering or user-user collaborative filtering. Use cosine similarity or Pearson correlation and generate top-N recommendations.

### 6. Matrix factorization
Implement or use a matrix-factorization model. You may use a library such as scikit-surprise, but you must explain the model and how recommendations are generated.

### 7. Evaluation
Compute at least:

- Precision@K
- Recall@K
- NDCG@K or MRR
- Catalog coverage or another beyond-accuracy metric

Optional rating-prediction metrics:

- MAE
- RMSE

### 8. Recommendation examples
Show recommendations for at least three users and compare how the algorithms differ.

## Suggested extensions

To obtain a stronger grade, add one or more of:

- user-user and item-item CF comparison
- TF-IDF vs raw genre vectors
- parameter tuning for k or latent factors
- novelty metric
- diversity metric
- popularity-bias analysis
- simple Streamlit interface
- music dataset adaptation

## Final report structure

1. Introduction
2. Dataset description
3. Preprocessing and EDA
4. Algorithms implemented
5. Evaluation protocol
6. Results table
7. Recommendation examples
8. Discussion of limitations
9. Conclusion

## Important rule

The template is intentionally incomplete. You must implement the TODOs and explain your choices. Submitting the placeholder code without completing and analyzing it is not sufficient.
