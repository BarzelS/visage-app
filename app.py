import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn

# Define the path to the model weights
model_path = "/Users/shir.barzel/BSO/visage/2024-09-15_10-39-01_model_epoch_373_interrupted.pth"

# Load the pre-trained ResNet50 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()  # Output a single probability value
)
model = model.to(device)

# Load the model weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title("Image Classification with ResNet50")

# Capture image from camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer).convert('RGB')
    
    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Transform the image
    img = transform(img).unsqueeze(0).to(device)
    
    # Run the model on the image
    with torch.no_grad():
        output = model(img)
        probability = output[0][0].item()
    
    # Display the output
    st.write(f"Predicted Probability: {probability:.2f}")