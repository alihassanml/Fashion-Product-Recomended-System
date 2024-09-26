# Importing necessary libraries
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

# Load data
df = pd.read_csv('styles.csv', on_bad_lines='skip')
feature_list = np.array(pickle.load(open('feature_data.pkl', 'rb')))
filenames = pickle.load(open('file_names.pkl', 'rb'))

# Reshape feature list
feature_list = feature_list.reshape(feature_list.shape[0], -1)

# Load pre-trained model
model = ResNet50(include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

# Function to extract features from an image
def extract_feature(file, model):
    img = load_img(file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    process_img = preprocess_input(img)
    result = model.predict(process_img)
    result = result / norm(result)
    return result

# Fit Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Streamlit app
st.title("Image Similarity Search")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Extract features from uploaded image
    image_feature = extract_feature(uploaded_file, model)
    image_feature = np.array(image_feature)
    
    # Find the nearest neighbors
    distances, indices = neighbors.kneighbors(image_feature)

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True,width=50)
    
    # Display similar images
    st.write("## Similar Images:")
    path = './images/'
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.style.use('ggplot')
    for i, ax in enumerate(axes.flatten()):
        if i < len(indices[0]):
            name = filenames[indices[0][i]]
            img = imd.imread(path + name)
            ax.imshow(img)
            val = int(name.split('.')[0])
            gender = df[df['id'] == val]['gender'].iloc[0]
            article_type = df[df['id'] == val]['articleType'].iloc[0]
            season = df[df['id'] == val]['season'].iloc[0]
            ax.set_title(f"{gender}\n{article_type}")
            ax.set_xlabel(f"Season: {season}")
        ax.axis('off')
    
    # Show the plot in Streamlit
    st.pyplot(fig)
