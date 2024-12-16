import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import os

# 设置设备
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

@st.cache_data
def compress_image(image, max_size_kb):
    """将图像压缩到指定大小以下。"""
    buffer = image.copy()
    width, height = buffer.size
    while True:
        buffer.thumbnail((width, height), Image.ANTIALIAS)
        with st.spinner("正在压缩图片..."):
            buffer_bytes = buffer.tobytes()
        if len(buffer_bytes) <= max_size_kb * 1024:
            return buffer
        width = int(width * 0.9)
        height = int(height * 0.9)

@st.cache_data
def process_image(style_name, content_image):
    """对内容图像进行风格迁移。"""
    load_model(style_name)
    transform = style_transform()
    content_tensor = transform(content_image).unsqueeze(0).to(device)
    with torch.no_grad():
        stylized_tensor = transformer(content_tensor)
    return deprocess(stylized_tensor)

# Streamlit 页面 UI
def main():
    st.title("快速图像风格迁移 APP")

    # 侧边栏内容
    st.sidebar.header("上传和设置")

    # 风格选择
    style_name = st.sidebar.selectbox("选择风格", list(STYLE_MODELS.keys()))

    # 上传内容图像
    content_images = st.sidebar.file_uploader("上传图片 (可多选)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # 检查上传的图片
    if content_images:
        for content_image_file in content_images:
            file_size_kb = content_image_file.size / 1024  # 转换为 KB
            content_image = Image.open(content_image_file)

            if file_size_kb > 1024:  # 超过 1MB
                st.sidebar.warning(f"{content_image_file.name} 文件过大，正在压缩...")
                content_image = compress_image(content_image, max_size_kb=1024)

            st.image(content_image, caption=f"{content_image_file.name} (已处理)", use_container_width=True)

        # 点击按钮进行风格迁移
        if st.sidebar.button("应用风格到所有图片"):
            results = []
            with st.spinner("正在处理所有图像..."):
                for content_image_file in content_images:
                    content_image = Image.open(content_image_file)
                    stylized_image = process_image(style_name, content_image)

                    output_path = f"stylized_{os.path.splitext(content_image_file.name)[0]}.jpg"
                    Image.fromarray(stylized_image).save(output_path)
                    results.append((output_path, stylized_image))

            # 显示处理结果
            for path, img in results:
                st.image(img, caption=f"风格迁移结果 - {path}", use_container_width=True)

    st.sidebar.markdown(
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
