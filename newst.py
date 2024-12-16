import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import io

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

def compress_image(image, max_size_kb):
    """将图像压缩到指定大小以下。"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=85)
        buffer_size_kb = len(buffer.getvalue()) / 1024

        while buffer_size_kb > max_size_kb:
            image = image.resize((int(image.width * 0.9), int(image.height * 0.9)))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            buffer_size_kb = len(buffer.getvalue()) / 1024

        buffer.seek(0)
        return Image.open(buffer)

def process_image(style_name, content_image):
    """对内容图像进行风格迁移。"""
    load_model(style_name)
    transform = style_transform()
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    # 分步处理逻辑，降低计算资源使用
    with torch.no_grad():
        stylized_tensor = transformer(content_tensor)
        stylized_tensor = stylized_tensor.cpu()  # 将结果转回 CPU

    return deprocess(stylized_tensor)

# Streamlit 页面 UI
def main():
    st.title("快速图像风格迁移 APP")

    # 侧边栏内容
    st.sidebar.header("上传和设置")

    # 风格选择
    style_name = st.sidebar.selectbox("选择风格", list(STYLE_MODELS.keys()))

    # 上传单张内容图像
    content_image_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if content_image_file:
        file_size_kb = content_image_file.size / 1024  # 转换为 KB
        content_image = Image.open(content_image_file)

        if file_size_kb > 1024:  # 超过 1MB
            st.sidebar.warning("文件过大，正在压缩...")
            content_image = compress_image(content_image, max_size_kb=1024)

        st.image(content_image, caption="上传的图像", use_container_width=True)

        # 点击按钮进行风格迁移
        if st.sidebar.button("应用风格"):
            with st.spinner("正在处理图像..."):
                stylized_image = process_image(style_name, content_image)

                # 显示处理结果
                st.image(stylized_image, caption="风格迁移结果", use_container_width=True)

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
