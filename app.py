import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from io import BytesIO
import base64

# Function to convert the image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Define the path to the model weights
model_path = "/mount/src/visage-app/2024-09-15_10-39-01_model_epoch_373_interrupted.pth"

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

# # Streamlit app
# st.title("VisageMed: Diabetes Detection from Facial Images")
# st.write("""
# ### Welcome to VisageMed
# This app analyzes facial images to detect the likelihood of diabetes using deep learning techniques.
# Upload a facial image or capture one using your camera, and our model will predict the probability of diabetes.
# """)

# st.sidebar.title("About VisageMed")
# st.sidebar.info("""
# VisageMed leverages state-of-the-art deep learning models to assess health risks from facial images. This tool is designed to provide a probability score for diabetes based on facial image analysis.
# For more information on diabetes detection, visit [American Diabetes Association](https://www.diabetes.org/).
# """)

logo_image = Image.open("/mount/src/visage-app/visageLogo.jpg")

# Combined markdown for HTML and CSS
st.markdown(
    f"""
    <style>
    .centered {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}
    .centered img {{
        margin-bottom: 20px;
    }}
    </style>
    <div class="centered">
        <img src="data:image/png;base64,{image_to_base64(logo_image)}" width="400">
        <h1>VisageMed: Diabetes Detection from Facial Images</h1>
        <h3>Welcome to VisageMed</h3>
        <p>This app analyzes facial images to detect the likelihood of diabetes using deep learning techniques.</p>
        <p>Upload a facial image or capture one using your camera, and our model will predict the probability of diabetes.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("About VisageMed")
st.sidebar.info("""
VisageMed leverages state-of-the-art deep learning models to assess health risks from facial images. This tool is designed to provide a probability score for diabetes based on facial image analysis.
""")

# Capture image from camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer).convert('RGB')
    
    # Display the image
    st.image(img, caption='Facial Image for Diabetes Detection', use_column_width=True)
    
    # Transform the image
    img = transform(img).unsqueeze(0).to(device)
    
    # Run the model on the image
    with torch.no_grad():
        output = model(img)
        probability = output[0][0].item()
    
    # Display the output
    if probability > 0.99:
        st.write(f"**Diabetes Detected:** High probability ({probability:.2f})")
    else:
        st.write(f"**No Diabetes Detected:** Low probability ({probability:.2f})")
