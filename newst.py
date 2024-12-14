import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess, extract_frames, save_video
import os
import cv2
import tqdm
import tempfile

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

    # Upload content video
    content_video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    # Buttons to apply style transfer
    if st.sidebar.button("Apply Style to Image"):
        if content_image_file:
            with st.spinner("Processing Image..."):
                load_model(style_name)

                # Load and process content image
                content_image = Image.open(content_image_file)
                transform = style_transform()
                content_tensor = transform(content_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    stylized_tensor = transformer(content_tensor)
                stylized_image = deprocess(stylized_tensor)

                # Save and display result
                output_path = "stylized_output.jpg"
                Image.fromarray(stylized_image).save(output_path)
                st.image(output_path, caption="Stylized Image", use_column_width=True)

    if st.sidebar.button("Apply Style to Video"):
        if content_video_file:
            with st.spinner("Processing Video..."):
                load_model(style_name)

                # Save uploaded video
                input_video_path = f"uploaded_{content_video_file.name}"
                with open(input_video_path, "wb") as f:
                    f.write(content_video_file.read())

                # Process video frames
                stylized_frames = []
                fps = 24  # Default FPS; can be extracted from the video
                frame_height, frame_width = None, None

                for frame in tqdm.tqdm(extract_frames(input_video_path), desc="Processing Frames"):
                    if frame_height is None or frame_width is None:
                        frame_height, frame_width = frame.size[1], frame.size[0]

                    transform = style_transform((frame_height, frame_width))
                    frame_tensor = transform(frame).unsqueeze(0).to(device)

                    with torch.no_grad():
                        stylized_frame = transformer(frame_tensor)
                    stylized_frames.append(deprocess(stylized_frame))

                # Save the stylized video to a temporary directory
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    output_video_path = tmp_file.name
                    save_video(stylized_frames, output_video_path, fps, (frame_width, frame_height))

                    # Display the video in Streamlit
                    st.video(output_video_path)
                    st.success(f"Stylized video saved at {output_video_path}!")

if __name__ == "__main__":
    main()
