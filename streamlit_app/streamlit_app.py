import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time
import os
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os
# Ensure utils can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui import inject_css

# Apply Global Enterprise Styling
inject_css()

# Cache functions for performance
@st.cache_data
def load_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn), (y_train_cat, y_test_cat)

def create_mlp_model():
    """Create MLP (Multi-Layer Perceptron) model"""
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model():
    """Create CNN (Convolutional Neural Network) model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, epochs=5, batch_size=128):
    """Train a model and return history"""
    with st.spinner('Training model...'):
        start_time = time.time()
        
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
    return history, training_time

def measure_latency(model, x_test, batch_size=100):
    """Measure prediction latency"""
    # Warm up
    model.predict(x_test[:10], verbose=0)
    
    # Measure latency
    start_time = time.time()
    predictions = model.predict(x_test[:batch_size], verbose=0)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) * 1000) / batch_size
    return avg_latency_ms

def evaluate_model(model, x_test, y_test):
    """Evaluate model and return metrics"""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Get predictions for F1 score
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    f1 = f1_score(y_test, predicted_classes, average='weighted')
    
    return test_loss, test_acc, f1

def save_model(model, filename):
    """Save model to file"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, filename)
    model.save(model_path)
    return model_path

@st.cache_resource
def load_saved_model(filename):
    """Load saved model"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, filename)
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None

def preprocess_uploaded_image(uploaded_file):
    """Preprocess uploaded image for prediction"""
    try:
        # Open image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert colors if needed (MNIST digits are white on black)
        if img_array.mean() > 127:
            img_array = 255 - img_array
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("🔢 MNIST Handwritten Digit Predictor")
    st.markdown("Compare MLP vs CNN performance on handwritten digit recognition")
    
    # Load data
    (x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn), (y_train_cat, y_test_cat) = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Select Tab", ["EDA", "Train & Compare", "Predict"])
    
    if tab == "EDA":
        st.header("📊 Exploratory Data Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", f"{len(x_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(x_test):,}")
        with col3:
            st.metric("Classes", "10 digits (0-9)")
        
        st.subheader("Dataset Distribution")
        
        # Class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            unique, counts = np.unique(y_train, return_counts=True)
            plt.bar(unique, counts, color='skyblue', alpha=0.7)
            plt.xlabel('Digit')
            plt.ylabel('Count')
            plt.title('Training Set - Class Distribution')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            unique, counts = np.unique(y_test, return_counts=True)
            plt.bar(unique, counts, color='lightcoral', alpha=0.7)
            plt.xlabel('Digit')
            plt.ylabel('Count')
            plt.title('Test Set - Class Distribution')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.subheader("Sample Images")
        
        # Display sample images
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        fig.suptitle('Sample Images from Each Class', fontsize=16)
        
        for digit in range(10):
            # Find first occurrence of each digit
            idx = np.where(y_train == digit)[0][0]
            
            # Training sample
            axes[0, digit].imshow(x_train[idx], cmap='gray')
            axes[0, digit].set_title(f'Train: {digit}')
            axes[0, digit].axis('off')
            
            # Test sample
            idx_test = np.where(y_test == digit)[0][0]
            axes[1, digit].imshow(x_test[idx_test], cmap='gray')
            axes[1, digit].set_title(f'Test: {digit}')
            axes[1, digit].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Image statistics
        st.subheader("Image Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set:**")
            st.write(f"- Shape: {x_train.shape}")
            st.write(f"- Min pixel value: {x_train.min():.3f}")
            st.write(f"- Max pixel value: {x_train.max():.3f}")
            st.write(f"- Mean pixel value: {x_train.mean():.3f}")
            
        with col2:
            st.write("**Test Set:**")
            st.write(f"- Shape: {x_test.shape}")
            st.write(f"- Min pixel value: {x_test.min():.3f}")
            st.write(f"- Max pixel value: {x_test.max():.3f}")
            st.write(f"- Mean pixel value: {x_test.mean():.3f}")
    
    elif tab == "Train & Compare":
        st.header("🏋️ Train & Compare Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Training Epochs", 1, 20, 5)
        with col2:
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
        
        if st.button("Train Both Models", type="primary"):
            
            # Create models
            mlp_model = create_mlp_model()
            cnn_model = create_cnn_model()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧠 MLP Model")
                st.code("""
Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])""")
                
                # Train MLP
                mlp_history, mlp_training_time = train_model(
                    mlp_model, x_train, y_train_cat, epochs, batch_size
                )
                
                # Evaluate MLP
                mlp_loss, mlp_acc, mlp_f1 = evaluate_model(mlp_model, x_test, y_test_cat)
                mlp_latency = measure_latency(mlp_model, x_test)
                
                # Save MLP model
                mlp_path = save_model(mlp_model, 'mlp_baseline.h5')
                
                st.success("✅ MLP Model Trained!")
            
            with col2:
                st.subheader("🖼️ CNN Model")
                st.code("""
Sequential([
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])""")
                
                # Train CNN
                cnn_history, cnn_training_time = train_model(
                    cnn_model, x_train_cnn, y_train_cat, epochs, batch_size
                )
                
                # Evaluate CNN
                cnn_loss, cnn_acc, cnn_f1 = evaluate_model(cnn_model, x_test_cnn, y_test_cat)
                cnn_latency = measure_latency(cnn_model, x_test_cnn)
                
                # Save CNN model
                cnn_path = save_model(cnn_model, 'mnist_cnn.h5')
                
                st.success("✅ CNN Model Trained!")
            
            # Comparison table
            st.subheader("📈 Model Comparison")
            
            comparison_data = {
                'Model': ['MLP', 'CNN'],
                'Test Accuracy': [f"{mlp_acc:.4f}", f"{cnn_acc:.4f}"],
                'F1 Score': [f"{mlp_f1:.4f}", f"{cnn_f1:.4f}"],
                'Avg Latency (ms)': [f"{mlp_latency:.2f}", f"{cnn_latency:.2f}"],
                'Training Time (s)': [f"{mlp_training_time:.2f}", f"{cnn_training_time:.2f}"]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Store models in session state
            st.session_state['mlp_model'] = mlp_model
            st.session_state['cnn_model'] = cnn_model
    
    elif tab == "Predict":
        st.header("🔮 Make Predictions")
        
        # Try to load saved models
        mlp_model = load_saved_model('mlp_baseline.h5')
        cnn_model = load_saved_model('mnist_cnn.h5')
        
        # Check session state for models
        if mlp_model is None and 'mlp_model' in st.session_state:
            mlp_model = st.session_state['mlp_model']
        if cnn_model is None and 'cnn_model' in st.session_state:
            cnn_model = st.session_state['cnn_model']
        
        if mlp_model is None or cnn_model is None:
            st.warning("⚠️ Models not found. Please train the models first in the 'Train & Compare' tab.")
            return
        
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file (preferably a handwritten digit)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a handwritten digit (0-9)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
            
            # Preprocess image
            processed_img = preprocess_uploaded_image(uploaded_file)
            
            if processed_img is not None:
                # Display processed image
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(processed_img, cmap='gray')
                    ax.set_title('Processed Image (28x28)')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Make predictions
                st.subheader("🎯 Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**MLP Model Prediction:**")
                    
                    # MLP prediction
                    start_time = time.time()
                    mlp_pred = mlp_model.predict(processed_img.reshape(1, 28, 28), verbose=0)
                    mlp_pred_time = (time.time() - start_time) * 1000
                    
                    mlp_predicted_digit = np.argmax(mlp_pred)
                    mlp_confidence = np.max(mlp_pred) * 100
                    
                    st.metric("Predicted Digit", mlp_predicted_digit)
                    st.metric("Confidence", f"{mlp_confidence:.2f}%")
                    st.metric("Prediction Time", f"{mlp_pred_time:.2f} ms")
                
                with col2:
                    st.write("**CNN Model Prediction:**")
                    
                    # CNN prediction
                    start_time = time.time()
                    cnn_pred = cnn_model.predict(processed_img.reshape(1, 28, 28, 1), verbose=0)
                    cnn_pred_time = (time.time() - start_time) * 1000
                    
                    cnn_predicted_digit = np.argmax(cnn_pred)
                    cnn_confidence = np.max(cnn_pred) * 100
                    
                    st.metric("Predicted Digit", cnn_predicted_digit)
                    st.metric("Confidence", f"{cnn_confidence:.2f}%")
                    st.metric("Prediction Time", f"{cnn_pred_time:.2f} ms")
                
                # Prediction summary
                st.subheader("📋 Prediction Summary")
                
                summary_data = {
                    'Model': ['MLP', 'CNN'],
                    'Predicted Digit': [mlp_predicted_digit, cnn_predicted_digit],
                    'Confidence (%)': [f"{mlp_confidence:.2f}", f"{cnn_confidence:.2f}"],
                    'Prediction Time (ms)': [f"{mlp_pred_time:.2f}", f"{cnn_pred_time:.2f}"]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        # Test with random samples
        st.subheader("🎲 Test with Random Sample")
        
        if st.button("Get Random Test Sample"):
            # Get random test sample
            idx = np.random.randint(0, len(x_test))
            test_image = x_test[idx]
            true_label = y_test[idx]
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(test_image, cmap='gray')
                ax.set_title(f'True Label: {true_label}')
                ax.axis('off')
                st.pyplot(fig)
            
            # Make predictions
            col1, col2 = st.columns(2)
            
            with col1:
                # MLP prediction
                start_time = time.time()
                mlp_pred = mlp_model.predict(test_image.reshape(1, 28, 28), verbose=0)
                mlp_pred_time = (time.time() - start_time) * 1000
                
                mlp_predicted = np.argmax(mlp_pred)
                mlp_conf = np.max(mlp_pred) * 100
                
                st.write("**MLP Prediction:**")
                st.write(f"Predicted: {mlp_predicted}")
                st.write(f"Confidence: {mlp_conf:.2f}%")
                st.write(f"Time: {mlp_pred_time:.2f} ms")
                st.write(f"Correct: {'✅' if mlp_predicted == true_label else '❌'}")
            
            with col2:
                # CNN prediction
                start_time = time.time()
                cnn_pred = cnn_model.predict(test_image.reshape(1, 28, 28, 1), verbose=0)
                cnn_pred_time = (time.time() - start_time) * 1000
                
                cnn_predicted = np.argmax(cnn_pred)
                cnn_conf = np.max(cnn_pred) * 100
                
                st.write("**CNN Prediction:**")
                st.write(f"Predicted: {cnn_predicted}")
                st.write(f"Confidence: {cnn_conf:.2f}%")
                st.write(f"Time: {cnn_pred_time:.2f} ms")
                st.write(f"Correct: {'✅' if cnn_predicted == true_label else '❌'}")

if __name__ == "__main__":
    main()