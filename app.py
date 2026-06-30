import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

# 1. Dynamically resolve paths to find the 'src' directory and model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
model_path = os.path.join(current_dir, "models", "best_mobilenet.pt")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import your modular architecture definition
from model import get_pretrained_model

# 2. Configure the Streamlit Page Aesthetics
st.set_page_config(page_title="AI Vision Classifier", page_icon="🔮", layout="centered")
st.title("🔮 Deep Learning Vision Classifier")
st.write("Upload any image, and our GPU-backed MobileNetV2 network will classify it into one of the 10 CIFAR-10 categories.")

# 3. Cache the model loading step so it stays in memory and runs instantly
@st.cache_resource
def load_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_pretrained_model(num_classes=10).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_classifier()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please ensure train.py ran completely!")

# Class mapping order matching your dataset training
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 4. Standard ImageNet pre-processing pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Build UI Elements
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the user's uploaded image side-by-side
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        
    with col2:
        st.subheader("Model Predictions")
        with st.spinner("Analyzing image patterns..."):
            # Prepare image tensor for model input
            img_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            
            # Execute Inference forward pass
            with torch.no_grad():
                outputs = model(img_tensor)
                # Convert raw output logits to percentage probabilities [0, 1]
                probabilities = F.softmax(outputs, dim=1)[0]
            
            # Extract top predicted class
            top_prob, top_cat_idx = torch.max(probabilities, 0)
            top_class = classes[top_cat_idx.item()]
            
            st.metric(label="Primary Classification Target", value=top_class.upper(), delta=f"{top_prob.item()*100:.1f}% Confidence")
            st.write("---")
            
            # Create a breakdown graph of all class probabilities
            prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
            # Sort highest probability first
            sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
            
            # Render visual progress bars for the top 3 possibilities
            st.write("**Top 3 Match Possibilities:**")
            for label, score in sorted_probs[:3]:
                st.write(f"**{label.title()}** ({score*100:.1f}%)")
                st.progress(score)