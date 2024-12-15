import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import numpy as np

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

    # 侧边栏
    st.sidebar.header("上传和设置")
    style_name = st.sidebar.selectbox("选择风格模型", list(STYLE_MODELS.keys()))
    content_image_file = st.sidebar.file_uploader("上传内容图片", type=["jpg", "png"])

    # 点击按钮进行风格迁移
    if st.sidebar.button("应用风格到图片"):
        if content_image_file:
            with st.spinner("正在处理图像..."):
                load_model(style_name)

                # 加载并处理内容图像
                content_image = Image.open(content_image_file).convert("RGB")
                content_tensor = style_transform(content_image).unsqueeze(0).to(device)

                # 风格迁移
                with torch.no_grad():
                    output_tensor = transformer(content_tensor)
                output_image = denormalize(output_tensor.squeeze())
                stylized_image = deprocess(output_image)

                # 保存和显示结果
                output_path = "stylized_image.jpg"
                Image.fromarray(stylized_image).save(output_path)

                st.image(output_path, caption="风格迁移后的图像", use_column_width=True)
                st.info("长按或者右键可以保存图片")
        else:
            st.warning("请先上传内容图片！")

if __name__ == "__main__":
    main()
