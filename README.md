swiggy-Restaurant-Recommendation-System
Restaurant Recommendation System
Overview
This project is a restaurant recommendation system that suggests restaurants based on user preferences. It uses clustering and similarity measures to find relevant restaurants based on features such as city, cuisine, rating, and cost. The system is integrated into a Streamlit application for easy user interaction.

Features
Data Preprocessing: Cleans and transforms restaurant data for analysis.
Encoding: Applies One-Hot Encoding to categorical features.
Clustering & Similarity Measures: Uses K-Means Clustering or Cosine Similarity to identify similar restaurants.
Result Mapping: Maps recommendation results back to the original dataset.
Streamlit Application: Provides an interactive web interface for users to get personalized restaurant recommendations.
Dataset
The dataset consists of restaurant details, including:

Name
City
Cuisine
Rating
Rating Count
Cost
Address
Menu Link
Project Workflow
1. Data Preprocessing
Removed duplicate entries.
Handled missing values.
Converted categorical columns to appropriate formats.
Transformed cost and rating fields into numeric values.
2. Encoding
Applied One-Hot Encoding to categorical features (e.g., city, cuisine).
Saved the encoder as a Pickle file (encoder.pkl).
Ensured all features are numerical after encoding.
Created a preprocessed dataset (encoded_data.csv).
Matched the indices of cleaned_data.csv and encoded_data.csv.
3. Recommendation Methodology
Implemented K-Means Clustering and Cosine Similarity to group and recommend restaurants.
Used encoded numerical data for computations.
4. Result Mapping
Mapped the recommendation results (indices) back to the non-encoded dataset (cleaned_data.csv).
5. Streamlit Application
Built an interactive web application that:

Accepts user input (e.g., preferred city, cuisine, rating, price range).
Processes the input and queries the encoded dataset for recommendations.
Displays recommended restaurants using cleaned_data.csv.
Installation
Prerequisites
Ensure you have Python 3.7+ and install the required dependencies:

pip install pandas numpy scikit-learn streamlit seaborn plotly
Running the Application
Run the Streamlit app using:

streamlit run app.py
Future Improvements
Improve the recommendation algorithm by incorporating user reviews.
Implement a hybrid model combining content-based and collaborative filtering.
Add a real-time API for live restaurant data integration.
Enhance the Streamlit UI with better filtering and visualization options.
