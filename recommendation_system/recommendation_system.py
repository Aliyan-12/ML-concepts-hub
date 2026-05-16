"""
Collaborative filtering recommendation system using cosine similarity on a user-item matrix.
Covers: matrix construction and transposition, pairwise cosine similarity, top-N similar user
lookup, unrated item identification, weighted score recommendation, mean-centered normalization
to correct for user rating bias, scalability with 6 users x 6 items, and seaborn heatmap
visualization of the similarity matrix saved as similarity_heatmap.png.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ratings = {
    'User1': [5, 0, 3, 0, 4, 0],
    'User2': [4, 0, 0, 2, 0, 3],
    'User3': [0, 3, 4, 0, 5, 0],
    'User4': [0, 0, 0, 5, 0, 4],
    'User5': [3, 5, 0, 0, 4, 0],
    'User6': [0, 4, 3, 0, 0, 5],
}
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6']

df = pd.DataFrame(ratings, index=items)
print("=== User-Item Matrix ===")
print(df)
print()

df_T = df.T

similarity_matrix = cosine_similarity(df_T)
similarity_df = pd.DataFrame(similarity_matrix, index=df_T.index, columns=df_T.index)
print("=== User-User Cosine Similarity Matrix ===")
print(similarity_df.round(4))
print()


def top_similar_users(user, sim_df, n=3):
    return sim_df[user].drop(user).sort_values(ascending=False).head(n)


def unrated_items(user, matrix_df):
    return matrix_df.index[matrix_df[user] == 0].tolist()


def recommend(target, matrix_df, sim_df, n_similar=3, top_n=3):
    unrated = unrated_items(target, matrix_df)
    similar = top_similar_users(target, sim_df, n=n_similar)
    total_sim = similar.sum()
    scores = {
        item: (
            sum(sim * matrix_df.loc[item, u] for u, sim in similar.items()) / total_sim
            if total_sim > 0 else 0
        )
        for item in unrated
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


target = 'User1'
print(f"=== Top 3 Users Most Similar to {target} ===")
print(top_similar_users(target, similarity_df).round(4))
print()

print(f"=== Items Not Yet Rated by {target} ===")
print(unrated_items(target, df))
print()

print(f"=== Recommended Items for {target} ===")
for item, score in recommend(target, df, similarity_df):
    print(f"  {item}: predicted score = {score:.4f}")
print()

df_norm = df_T.astype(float).copy()
for user in df_norm.index:
    mask = df_norm.loc[user] > 0
    if mask.any():
        df_norm.loc[user, mask] -= df_norm.loc[user, mask].mean()

print("=== Normalized Ratings (Mean-Centered per User) ===")
print(df_norm.round(2))
print()

norm_sim_df = pd.DataFrame(
    cosine_similarity(df_norm), index=df_T.index, columns=df_T.index
)
print("=== Normalized User-User Similarity Matrix ===")
print(norm_sim_df.round(4))
print()

plt.figure(figsize=(8, 6))
sns.heatmap(
    similarity_df, annot=True, fmt=".2f", cmap="Blues",
    linewidths=0.5, square=True, cbar_kws={"label": "Cosine Similarity"}
)
plt.title("User-User Cosine Similarity Heatmap")
plt.tight_layout()
plt.savefig("similarity_heatmap.png", dpi=150)
plt.show()
print("Saved: similarity_heatmap.png")
