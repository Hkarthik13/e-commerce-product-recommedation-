import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import re

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="Amazon Product Recommender", layout="wide")

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Cleaned_Amazon_Customer_Behavior_Survey.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset file 'Cleaned_Amazon_Customer_Behavior_Survey.csv' not found. Please ensure it is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

df = load_data()

# Extract unique product categories
categories = ['Beauty and Personal Care', 'Clothing and Fashion', 'Groceries and Gourmet Food', 
              'Home and Kitchen', 'others']

# Preprocess text for user comments
def preprocess_text(text):
    if pd.isna(text):
        return []
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Create user-item matrix
def create_user_item_matrix(df, categories):
    user_item_matrix = pd.DataFrame(0, index=df.index, columns=categories)
    for idx, row in df.iterrows():
        purchased_categories = row['Purchase_Categories'].split(';')
        rating = row['Customer_Reviews_Importance']
        for cat in purchased_categories:
            if cat in categories:
                user_item_matrix.loc[idx, cat] = rating
    return user_item_matrix

user_item_matrix = create_user_item_matrix(df, categories)

# Function to get recommendations
def get_recommendations(user_ratings, user_item_matrix, categories, min_rating=0):
    try:
        user_vector = np.zeros(len(categories))
        for cat, rating in user_ratings.items():
            if cat in categories:
                user_vector[categories.index(cat)] = rating
        
        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        weighted_ratings = user_item_matrix.T.dot(similarities) / (similarities.sum() + 1e-9)
        
        unrated_categories = [cat for cat in categories if user_ratings.get(cat, 0) == 0]
        recommendations = pd.DataFrame({
            'category': unrated_categories,
            'predicted_rating': [weighted_ratings[categories.index(cat)] for cat in unrated_categories]
        })
        
        filtered_recommendations = recommendations[recommendations['predicted_rating'] >= min_rating]
        filtered_recommendations = filtered_recommendations.sort_values(by='predicted_rating', ascending=False)
        
        # Fallback: If no recommendations meet the threshold, suggest popular categories
        if filtered_recommendations.empty and unrated_categories:
            category_counts = df['Purchase_Categories'].str.split(';', expand=True).stack().value_counts()
            popular_categories = category_counts[category_counts.index.isin(unrated_categories)].head(3)
            fallback_recommendations = pd.DataFrame({
                'category': popular_categories.index,
                'predicted_rating': [0.0] * len(popular_categories)
            })
            return filtered_recommendations, fallback_recommendations
        return filtered_recommendations, None
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame(), None

# Function to analyze user comments
def analyze_comments(df, category):
    try:
        comments = df[df['Purchase_Categories'].str.contains(category, case=False, na=False)]['Improvement_Areas']
        all_tokens = []
        for comment in comments:
            tokens = preprocess_text(comment)
            all_tokens.extend(tokens)
        word_freq = pd.Series(all_tokens).value_counts().head(10)
        return word_freq
    except Exception as e:
        st.error(f"Error analyzing comments: {str(e)}")
        return pd.Series()

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stSlider > div > div > div > div { background-color: #ff9900; }
    .stButton > button { background-color: #ff9900; color: white; border-radius: 5px; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    h1 { color: #232f3e; }
    h2, h3 { color: #ff9900; }
    .st-expander { background-color: #ffffff; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.header("User Profile")
    age = st.slider("Your Age", 16, 70, 25, key="age_slider")
    gender = st.selectbox("Gender", ["Female", "Male", "Prefer not to say", "Others"], key="gender_select")
    purchase_freq = st.selectbox("Purchase Frequency", 
                                ["Less than once a month", "Once a month", "Few times a month", 
                                 "Once a week", "Multiple times a week"], key="freq_select")
    
    st.header("Rate Product Categories")
    user_ratings = {}
    for category in categories:
        rating = st.slider(f"{category} (0-5)", 0, 5, 0, key=f"rate_{category}")
        if rating > 0:
            user_ratings[category] = rating
    
    min_rating = st.slider("Minimum Predicted Rating", 0.0, 5.0, 2.0, 0.1, key="min_rating_slider")
    st.markdown("*Tip*: Lower the threshold if no recommendations are shown.")

# Main content
st.title("Amazon Product Category Recommendation System")
st.markdown("Discover personalized product category recommendations tailored to your preferences!")

# Display user profile
st.subheader("Your Profile")
col1, col2, col3 = st.columns(3)
col1.metric("Age", age)
col2.metric("Gender", gender)
col3.metric("Purchase Frequency", purchase_freq)

# Recommendations
if st.button("Get Recommendations", key="recommend_button"):
    if not user_ratings:
        st.warning("Please rate at least one product category to get recommendations.")
    else:
        with st.spinner("Generating recommendations..."):
            recommendations, fallback = get_recommendations(user_ratings, user_item_matrix, categories, min_rating)
            st.subheader("Recommended Product Categories")
            if not recommendations.empty:
                st.dataframe(recommendations.style.format({"predicted_rating": "{:.2f}"}))
                
                # Visualization of recommendations
                fig = px.bar(recommendations, x='category', y='predicted_rating', 
                            title="Predicted Ratings for Recommended Categories",
                            labels={'category': 'Product Category', 'predicted_rating': 'Predicted Rating'},
                            color='predicted_rating', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                # Comment analysis for top recommended category
                top_category = recommendations.iloc[0]['category']
                st.subheader(f"User Feedback for {top_category}")
                word_freq = analyze_comments(df, top_category)
                if not word_freq.empty:
                    fig_comments = px.bar(word_freq, x=word_freq.index, y=word_freq.values,
                                        title=f"Top Words in Feedback for {top_category}",
                                        labels={'index': 'Words', 'y': 'Frequency'},
                                        color=word_freq.values, color_continuous_scale='Oranges')
                    st.plotly_chart(fig_comments, use_container_width=True)
                else:
                    st.write("No feedback available for this category.")
            else:
                st.info("No recommendations meet the minimum rating threshold. Try lowering the threshold or rating more categories.")
                if fallback is not None and not fallback.empty:
                    st.subheader("Popular Categories (Fallback)")
                    st.write("Based on overall popularity in the dataset, you might like:")
                    st.dataframe(fallback.style.format({"predicted_rating": "{:.2f}"}))
                    # Visualization of fallback recommendations
                    fig_fallback = px.bar(fallback, x='category', y='predicted_rating', 
                                         title="Popular Categories (No Predicted Ratings)",
                                         labels={'category': 'Product Category', 'predicted_rating': 'Popularity Score'},
                                         color='category', color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig_fallback, use_container_width=True)

# Dataset insights
with st.expander("Dataset Insights"):
    st.subheader("Purchase Category Distribution")
    category_counts = df['Purchase_Categories'].str.split(';', expand=True).stack().value_counts()
    fig_dataset = px.pie(values=category_counts.values, names=category_counts.index,
                        title="Distribution of Purchase Categories")
    st.plotly_chart(fig_dataset, use_container_width=True)

# Instructions
with st.expander("How to Use"):
    st.markdown("""
        1. **Enter Your Profile**: Input your age, gender, and purchase frequency in the sidebar.
        2. **Rate Categories**: Rate product categories (0-5) using the sliders.
        3. **Set Rating Threshold**: Adjust the minimum predicted rating. Lower it if no recommendations appear.
        4. **Get Recommendations**: Click the button to see tailored categories, visualizations, and feedback analysis.
        5. **Fallback**: If no recommendations meet the threshold, popular categories are suggested.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data: Amazon Customer Behavior Survey")
