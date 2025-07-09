import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
target_names = iris.target_names

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, Y)

# Prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Class label table
class_labels_df = pd.DataFrame({
    'Index': [0, 1, 2],
    'Class Label': target_names
})
st.subheader("🧾 Iris Class Labels")
st.dataframe(class_labels_df)

# Output prediction
st.subheader("🔍 Prediction")
predicted_class = target_names[prediction][0].capitalize()
st.success(f"The predicted Iris flower is: **{predicted_class}**")

# Show flower image
st.subheader("🌸 Flower Image")
images = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
}
st.image(images[target_names[prediction][0]], caption=f"{predicted_class} Iris", use_container_width=True)


# Show probability table
st.subheader("📈 Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'))

# Show bar chart
st.subheader("📊 Probability Chart")
fig, ax = plt.subplots()
ax.bar(target_names, prediction_proba[0], color=["#90ee90", "#add8e6", "#ffb6c1"])
ax.set_ylabel("Probability")
ax.set_ylim([0, 1])
ax.set_title("Prediction Probability")
st.pyplot(fig)

# Download result
st.subheader("📥 Download Prediction")
result_df = df.copy()
result_df["Predicted Class"] = predicted_class
for i, name in enumerate(target_names):
    result_df[f"Probability ({name})"] = prediction_proba[0][i]

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Result as CSV",
    data=csv,
    file_name='iris_prediction.csv',
    mime='text/csv'
)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ using Streamlit\n\nAuthor: Sakshi Gupta")
