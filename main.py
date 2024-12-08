import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import BertModel, BertTokenizer

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load the answer space
def load_answer_space(file_path):
    try:
        with open(file_path, "r") as file:
            answers = [line.strip() for line in file if line.strip()]  # Read and clean lines
        return answers
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading the file: {e}")

# Define the VQAModel class
class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel, self).__init__()
        # Image feature extractor
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer

        # Question feature extractor
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Fusion and final classification
        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = checkpoint(self.cnn, images)  # Use checkpointing for ResNet

        # Extract question features
        outputs = checkpoint(self.bert, input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, 768)

        # Concatenate features
        combined_features = torch.cat((image_features, question_features), dim=1)

        # Classification
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model (1).pth"
    model = VQAModel(num_answers=len(answer_space))  # Use dynamic num_answers
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device

# Preprocessing functions
def preprocess_image(image):
    return transform(image).unsqueeze(0)

def preprocess_question(question, tokenizer, max_length=50):
    encoding = tokenizer(question, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    return encoding["input_ids"], encoding["attention_mask"]

# Load the answer space
answer_file_path = "answer_space.txt"  # Adjust this path if needed
answer_space = load_answer_space(answer_file_path)

# Load model and tokenizer
model, device = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Streamlit app setup
st.title("Visual Question Answering System")

# Session state for storing inputs
if 'image_tensor' not in st.session_state:
    st.session_state.image_tensor = None

if 'question' not in st.session_state:
    st.session_state.question = None

# Step 1: Image upload
st.header("Step 1: Upload an Image")
uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        img = Image.open(uploaded_image).convert("RGB")
        st.session_state.image_tensor = preprocess_image(img).to(device)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Step 2: Question input
st.header("Step 2: Enter a Question")
question_input = st.text_input("Ask a question about the uploaded image:")

if question_input:
    try:
        st.session_state.question = question_input
        question_ids, attention_mask = preprocess_question(question_input, tokenizer)
        question_ids = question_ids.to(device)
        attention_mask = attention_mask.to(device)
    except Exception as e:
        st.error(f"Error processing question: {e}")

# Display inputs and run inference
if st.session_state.image_tensor is not None and st.session_state.question is not None:
    st.header("Inputs Received")
    st.write("**Question:**", st.session_state.question)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Model inference
    with torch.no_grad():
        output = model(st.session_state.image_tensor, question_ids, attention_mask)
        topk_values, topk_indices = torch.topk(output, k=5, dim=1)

    # Display predictions
    st.header("Predictions")
    for rank, (score, idx) in enumerate(zip(topk_values[0], topk_indices[0])):
        st.write(f"Rank {rank + 1}: {answer_space[idx.item()]} (Score: {score.item():.4f})")
