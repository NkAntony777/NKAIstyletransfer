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

    # 侧边栏内容
    st.sidebar.header("上传和设置")

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

    # 风格选择
    style_name = st.sidebar.selectbox("选择风格", list(STYLE_MODELS.keys()))

    # 上传内容图像
    content_image_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    # 显示原始图像
    if content_image_file:
        content_image = Image.open(content_image_file)
        st.image(content_image, caption="原始图像", use_container_width=True)

    # 点击按钮进行风格迁移
    if st.sidebar.button("应用风格到图片"):
        if content_image_file:
            with st.spinner("正在处理图像..."):
                load_model(style_name)

                # 加载并处理内容图像
                transform = style_transform()
                content_tensor = transform(content_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    stylized_tensor = transformer(content_tensor)
                stylized_image = deprocess(stylized_tensor)

                # 保存并显示结果
                output_path = "stylized_output.jpg"
                Image.fromarray(stylized_image).save(output_path)

                st.image(output_path, caption="风格迁移后的图像", use_container_width=True)

if __name__ == "__main__":
    main()
