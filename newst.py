import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet().to(device)

# Predefined style models
STYLE_MODELS = {
    "Cuphead": "cuphead_10000.pth",
    "Starry Night": "starry_night_10000.pth",
    "Mosaic": "mosaic_10000.pth",
}

def load_model(style_name):
    """Load the selected style model."""
    model_path = STYLE_MODELS[style_name]
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()

# Streamlit UI
def main():
    st.title("Fast Neural Style Transfer")

    st.sidebar.header("Upload and Settings")

    # Style selection
    style_name = st.sidebar.selectbox("Select a Style", list(STYLE_MODELS.keys()))

    # Upload content image
    content_image_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Display the original image
    if content_image_file:
        content_image = Image.open(content_image_file)
        st.image(content_image, caption="Original Image", use_column_width=True)

    # Apply style transfer to image
    if st.sidebar.button("Apply Style to Image"):
        if content_image_file:
            with st.spinner("Processing Image..."):
                load_model(style_name)

                # Load and process content image
                transform = style_transform()
                content_tensor = transform(content_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    stylized_tensor = transformer(content_tensor)
                stylized_image = deprocess(stylized_tensor)

                # Save and display result
                output_path = "stylized_output.jpg"
                Image.fromarray(stylized_image).save(output_path)

                st.image(output_path, caption="Stylized Image", use_column_width=True)

if __name__ == "__main__":
    main()
