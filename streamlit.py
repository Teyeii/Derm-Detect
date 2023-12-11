# streamlit_app.py
import streamlit as st
import opencv-python as cv2
import numpy as np
import pickle

# Load your trained model from GitHub
model_url = './teyeis_svm_model.pkl'
svm_model = pickle.load(urllib.request.urlopen(model_url))

# Function to preprocess and predict
def predict_skin_cancer(image):
    # Preprocess the image (you may need to adjust this based on your training data preprocessing)
    # Assuming your model expects input shape (28, 28, 3)
    image = cv2.resize(image, (28, 28))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize to [0, 1]

    # Make prediction
    prediction = svm_model.predict(image.flatten().reshape(1, -1))
    confidence = np.max(svm_model.decision_function(image.flatten().reshape(1, -1)))

    return prediction[0], confidence

# Streamlit app
def main():
    st.title("Skin Cancer Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = cv2.imread(uploaded_file.name)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict
        class_index, confidence = predict_skin_cancer(image)

        classes = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7"]
        st.write(f"Prediction: {classes[class_index]}, Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()
