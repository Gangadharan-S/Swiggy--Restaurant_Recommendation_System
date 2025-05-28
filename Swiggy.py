# Swiggy’s Restaurant Recommendation System using Streamlit 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("D:\\App\\vscode1\\swiggy\\cleaned_df1.csv")

df = load_data()

# Load encoders, scalers, and KMeans model
with open("D:\\App\\vscode1\\swiggy\\encoders_scaler3.pkl", "rb") as f:
    encoders_scalers = pickle.load(f)

with open("D:\\App\\vscode1\\swiggy\\kmeans_model (1).pkl", "rb") as f:
    kmeans = pickle.load(f)

name_encoder = encoders_scalers["name_encoder"]
cuisine_encoder = encoders_scalers["cuisine_encoder"]
city_encoder = encoders_scalers["city_encoder"]
rating_scaler = encoders_scalers["rating_scaler"]
cost_scaler = encoders_scalers["cost_scaler"]

# Encode categorical features
df["name_encoded"] = name_encoder.transform(df["name"])
df["cuisine_encoded"] = cuisine_encoder.transform(df["cuisine"])

city_encoded = city_encoder.transform(df[["city"]])  # Keep as sparse matrix
city_encoded_df = pd.DataFrame(city_encoded.toarray(), columns=city_encoder.get_feature_names_out())

# Scale numerical features
df["rating_scaled"] = rating_scaler.transform(df[["rating"]]).astype(np.float32)
df["cost_scaled"] = cost_scaler.transform(df[["cost"]]).astype(np.float32)

# Merge encoded city features
df = pd.concat([df, city_encoded_df], axis=1)

# Ensure correct feature alignment for KMeans prediction
feature_columns = ["name_encoded", "cuisine_encoded", "rating_scaled", "cost_scaled"] + list(city_encoded_df.columns)
df_filtered = df[feature_columns]
df["cluster"] = kmeans.predict(df_filtered)

# Function to get restaurant recommendations
def get_recommendations(cluster, city, cuisines, min_rating, max_cost, top_n=10):
    if "All" in cuisines:
        cuisines = df["cuisine"].unique()
    
    filtered_df = df[(df["cluster"] == cluster) & 
                     (df["city"] == city) & 
                     (df["cuisine"].isin(cuisines)) & 
                     (df["rating"] >= min_rating) & 
                     (df["cost"] <= max_cost)]
    
    # Sort by rating (highest first) and cost (lowest first)
    filtered_df = filtered_df.sort_values(by=["rating", "cost"], ascending=[False, True])
    
    # Include address, link, and rating_count in recommendations
    return filtered_df[["name", "city", "cuisine", "cost", "rating", "rating_count", "address", "link"]].head(top_n)

# Streamlit UI

st.markdown("""
    <h1 style='font-size: 40px;'>Swiggy’s Restaurant Recommendation System</h1>
    """, unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("User Preferences")
city = st.sidebar.selectbox("Select City", df["city"].unique())
cuisine_options = ["All"] + list(df["cuisine"].unique())
cuisines = st.sidebar.multiselect("Select Cuisines", cuisine_options, default=["All"])
rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)
# cost = st.sidebar.slider("Maximum Cost", 0, 2000, 1000)
cost = st.sidebar.number_input('Maximum Cost', min_value=0.0, step=0.1)

# Handle unseen categories safely
if city in city_encoder.categories_[0]:
    city_transformed = city_encoder.transform([[city]])
else:
    city_transformed = csr_matrix((1, city_encoded.shape[1]))  # Sparse zero vector for unseen cities

cuisine_encoded = [cuisine_encoder.transform([c])[0] if c in cuisine_encoder.classes_ else -1 for c in cuisines if c != "All"]

# Encode name feature with a default value (e.g., -1) for unseen names
name_encoded = -1  # Placeholder, as we don't have user input for restaurant names

# Prepare user input array
user_input = np.hstack([[name_encoded, cuisine_encoded[0] if cuisine_encoded else -1, 
                         rating_scaler.transform([[rating]])[0][0], 
                         cost_scaler.transform([[cost]])[0][0]], 
                         city_transformed.toarray().flatten()])

# Ensure correct feature alignment
user_input_df = pd.DataFrame([user_input], columns=feature_columns)

# Predict user cluster
user_cluster = kmeans.predict(user_input_df)[0]

# Get recommendations
recommendations = get_recommendations(user_cluster, city, cuisines, rating, cost)

# Display results
st.subheader("Recommended Restaurants:")
if not recommendations.empty:
    for _, row in recommendations.iterrows():
        # st.write(f"**{row['name']}**")
        st.markdown(f"""
        <h3 style='font-size: 24px;'><b>{row['name']}</b></h3>
        """, unsafe_allow_html=True)
        st.write(f" Cuisine: {row['cuisine']} | Cost: {row['cost']} INR")
        st.write(f" Rating: {row['rating']}  | Reviews: {row['rating_count']}| City: {row['city']}" )
        st.write(f"Address: {row['address']}")
        st.write(f"Link: ({row['link']})")
        st.markdown("---")
else:
    st.warning("No restaurants match your criteria. Try adjusting filters.")
