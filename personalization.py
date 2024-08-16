import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the Dataset
df = pd.read_csv('Cleaned_Amazon_Customer_Behavior_Survey.csv')

# Replace 'UserID', 'ProductID', and 'Shopping_Satisfaction' with actual column names from your dataset
user_column = 'UserID'  # Replace with the actual column name for users
product_column = 'ProductID'  # Replace with the actual column name for products
rating_column = 'Shopping_Satisfaction'  # Replace with the actual column name for ratings or satisfaction

# Step 2: Prepare the Data
df['Rating'] = df[rating_column]  # Assuming 'Shopping_Satisfaction' is used as a proxy for the rating

# Create a user-item matrix where rows are users and columns are products
user_item_matrix = df.pivot_table(index=user_column, columns=product_column, values='Rating').fillna(0)

# Step 3: Calculate Cosine Similarity Between Users
user_similarity = cosine_similarity(user_item_matrix)

# Convert the similarity matrix to a DataFrame for easy handling
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Step 4: Generate Recommendations
def get_top_n_recommendations(user_id, n=10):
    # Get the user's similarity scores with all other users
    user_sim_scores = user_similarity_df[user_id]

    # Multiply the similarity scores by the user-item matrix to get weighted ratings
    weighted_ratings = user_item_matrix.T.dot(user_sim_scores)

    # Normalize the weighted ratings by the sum of the similarities
    weighted_ratings = weighted_ratings / user_sim_scores.sum()

    # Get the products the user has already rated
    user_rated_products = user_item_matrix.loc[user_id, user_item_matrix.loc[user_id] > 0].index.tolist()

    # Filter out products the user has already rated
    recommendations = weighted_ratings.drop(user_rated_products).sort_values(ascending=False)

    # Return the top N recommended products
    return recommendations.head(n)

# Step 5: Display Recommendations for a Specific User
specific_user_id = user_item_matrix.index[0]  # Replace with an actual user ID
recommendations = get_top_n_recommendations(specific_user_id, n=10)
print(f"Top 10 recommendations for User {specific_user_id}:")
print(recommendations)
