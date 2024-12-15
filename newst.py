import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess

# 加载模型并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet().to(device)

# 预定义的风格模型
STYLE_MODELS = {
    "Cuphead": "cuphead_10000.pth",
    "Starry Night": "starry_night_10000.pth",
    "Mosaic": "mosaic_10000.pth",
}

def load_model(style_name):
    """加载所选的风格模型。"""
    model_path = STYLE_MODELS[style_name]
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()

# Streamlit 页面 UI
def main():
    st.title("快速图像风格迁移 APP")
    st.title("快速神经风格迁移")

    st.sidebar.header("上传和设置")


    # 点击按钮进行风格迁移
    if st.sidebar.button("应用风格到图片"):
        if content_image_file:
            with st.spinner("正在处理图像..."):
            with st.spinner("正在飞速运转..."):
                load_model(style_name)

                # 加载并处理内容图像

                Image.fromarray(stylized_image).save(output_path)

                st.image(output_path, caption="风格迁移后的图像", use_container_width=True)
                st.info("长按或者右键可以保存")

if __name__ == "__main__":
    main()
