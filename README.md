CIFAR-10 Image Classification with CNN & Streamlit
This project features a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into ten distinct categories (e.g., airplanes, cats, dogs, trucks). It includes a Jupyter Notebook for training and a Streamlit web application for real-time image inference.

🚀 Features
CNN Architecture: Built using TensorFlow/Keras with layers designed for image feature extraction.

Streamlit Web App: A user-friendly interface to upload images and get instant classification results.

Data Augmentation: Techniques used during training to improve model generalization and accuracy.

Modern Serialization: Saves and loads models using the latest .keras format.

📁 Project Structure
cnn.ipynb: The main notebook used to load data, preprocess labels, and train the CNN model.
app.py: The Streamlit application script for the web interface.
cifar10_model.keras: The trained model weights (generated after running the notebook).
1. Train the ModelOpen cnn.ipynb in your preferred editor (Jupyter Lab or VS Code). Run all cells to:Load and normalize the CIFAR-10 dataset.Fix label dimensions using to_categorical.Train the model for 20 epochs.Save the final model as cifar10_model.keras.2. Run the Streamlit AppOnce the cifar10_model.keras file is generated, launch the web application from your terminal:Bashstreamlit run app.py
3. Upload & ClassifyThe app will open in your browser (usually at http://localhost:8501).Upload any image (JPG, PNG, or JPEG).The app automatically resizes the image to $32 \times 32$ pixels and predicts the class.📊 Model ClassesThe model can identify the following 10 classes:AirplaneAutomobileBirdCatDeerDogFrogHorseShipTruck⚠️ NotesLarge Images: If you plan to test with very large high-resolution images (e.g., 200MB+), run the app with the following command to bypass default upload limits:Bashstreamlit run app.py --server.maxUploadSize 500
Performance: The model performs best on images where the subject is centered and clearly matches one of the ten categories.
