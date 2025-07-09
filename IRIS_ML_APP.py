import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="🌸 Iris Flower Predictor", layout="centered")

st.title("🌸 Iris Flower Prediction App")
st.write("Adjust the sliders on the sidebar to input flower measurements. The app will predict the Iris flower type using a machine learning model.")

# Sidebar input
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.subheader("📋 User Input")
st.dataframe(df)

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, Y)

# Prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Class label table
class_labels_df = pd.DataFrame({
    'Index': [0, 1, 2],
    'Class Label': iris.target_names
})
st.subheader("🧾 Iris Class Labels")
st.dataframe(class_labels_df)

# Output prediction
st.subheader("🔍 Prediction")
predicted_class = iris.target_names[prediction][0].capitalize()
st.success(f"The predicted Iris flower is: **{predicted_class}**")

# Probability table

st.subheader("📈 Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'))

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ using Streamlit\n\nAuthor: Sakshi Gupta")
