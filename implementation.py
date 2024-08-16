import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from flask import Flask, request, jsonify

# Load and preprocess dataset
data = pd.read_csv('Amazon Customer Behavior Survey.csv')
data.fillna(method='ffill', inplace=True)
print(data.head())
print(data.info())
print(data.describe())

# Exploratory Data Analysis
user_interactions = data.groupby('user_id')['rating'].count()
plt.hist(user_interactions, bins=50)
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Users')
plt.title('User Interaction Distribution')
plt.show()

popular_products = data.groupby('product_id')['rating'].count().sort_values(ascending=False)
plt.bar(popular_products.index[:10], popular_products.values[:10])
plt.xlabel('Product ID')
plt.ylabel('Number of Interactions')
plt.title('Top 10 Popular Products')
plt.show()

# Build the recommendation system using SVD
reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
surprise_data = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(surprise_data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

# Cross-validation
cross_validate(svd, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Evaluate the model on test set
predictions = svd.test(testset)
accuracy.rmse(predictions)

# Calculate precision and recall
threshold = 3.5
precision, recall = [], []
for uid, _, true_r, est, _ in predictions:
    if est >= threshold:
        if true_r >= threshold:
            precision.append(1)
        else:
            precision.append(0)
    if true_r >= threshold:
        if est >= threshold:
            recall.append(1)
        else:
            recall.append(0)
precision = np.mean(precision)
recall = np.mean(recall)
print(f'Precision: {precision}, Recall: {recall}')

# Deployment with Flask
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    top_n = request.json.get('top_n', 10)
    user_interactions = data[data['user_id'] == user_id]
    recommendations = []
    for product_id in data['product_id'].unique():
        if product_id not in user_interactions['product_id'].values:
            est = svd.predict(user_id, product_id).est
            recommendations.append((product_id, est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommendations = recommendations[:top_n]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
