import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model you saved from your cnn.ipynb
# This uses the modern .keras format as recommended by TensorFlow
@st.cache_resource
def load_original_model():
    try:
        # Ensure the filename matches what you saved in your notebook
        return tf.keras.models.load_model('cifar10_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_original_model()

# 2. CIFAR-10 Class Names
# These are the standard 10 labels the model was trained to recognize
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 3. Streamlit UI Layout
st.set_page_config(page_title="CNN Image Classifier", layout="centered")

st.title("🖼️ Image Classification using CNN")
st.write("Upload an image and the model trained in your notebook will classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 4. Preprocessing
    # Step A: Convert to RGB to ensure 3 channels (removes Alpha/Transparency)
    image_rgb = image.convert('RGB')
    
    # Step B: Resize to 32x32 pixels (Must match model input)
    img_resized = image_rgb.resize((32, 32))
    
    # Step C: Convert to array and Normalize pixel values
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    # Step D: Add Batch Dimension (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prediction
    # Added a unique 'key' to prevent the DuplicateElementId error
    if st.button('Run Classification', key='predict_action_button'):
        if model is not None:
            with st.spinner('Analyzing Image...'):
                predictions = model.predict(img_array)
                
                # Get the index of the highest probability
                result_index = np.argmax(predictions)
                result_label = class_names[result_index]
                confidence = np.max(predictions) * 100

                # Display the results
                st.divider()
                st.subheader(f"Prediction: **{result_label.upper()}**")
                st.write(f"Confidence Level: **{confidence:.2f}%**")
                
                # Progress bar to visualize confidence
                st.progress(int(confidence))
        else:
            st.error("Model not found. Please check your .keras file.")