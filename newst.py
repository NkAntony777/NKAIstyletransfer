import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import time

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

# 使用缓存来避免重复计算，设置 TTL 1小时
@st.cache_data(ttl=3600)  # 缓存1小时
def process_stylized_image(content_image, style_name):
    """处理风格迁移，返回处理后的图片"""
    # 加载并处理内容图像
    transform = style_transform()
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        stylized_tensor = transformer(content_tensor)
    stylized_image = deprocess(stylized_tensor)

    return stylized_image

# Streamlit 页面 UI
def main():
    st.title("快速图像风格迁移 APP")

    # 侧边栏内容
    st.sidebar.header("上传和设置")

    # 风格选择
    style_name = st.sidebar.selectbox("选择风格", list(STYLE_MODELS.keys()))

    # 上传内容图像，并检查文件大小
    content_image_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if content_image_file:
        # 获取上传文件的大小，单位为字节
        file_size = content_image_file.size

        # 如果文件大小超过1.5MB，提示用户并要求上传小于1MB的图片
        if file_size > 3 * 1024 * 1024:  # 1MB = 1 * 1024 * 1024 bytes
            st.sidebar.warning("文件过大，请上传小于3MB的图片，或者选取下载到本地的微信对话中的非原图。")
        else:
            # 如果文件大小合适，则继续处理图片
            content_image = Image.open(content_image_file)
            st.image(content_image, caption="原始图像", use_container_width=True)

    # 点击按钮进行风格迁移
    if st.sidebar.button("应用风格到图片"):
        if content_image_file and file_size <= 3 * 1024 * 1024:  # 确保图片文件大小小于1.5MB
            with st.spinner("正在处理图像..."):
                load_model(style_name)
                
                # 使用缓存处理图像
                stylized_image = process_stylized_image(content_image, style_name)

                # 保存并显示结果
                output_path = "stylized_output.jpg"
                Image.fromarray(stylized_image).save(output_path)

                st.image(output_path, caption="风格迁移后的图像", use_container_width=True)

    # 在侧边栏显示小组成员信息
    with st.sidebar.container():
        st.markdown(
            """<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px;'>
            <b>小组成员：</b><br>
            薛世彤<br>
            陈鹏<br>
            卢厚任<br>
            宋达铠<br>
            张宁
            </div>""",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
