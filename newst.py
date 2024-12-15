import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import os
import schedule
import time
import threading

# 加载模型并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet().to(device)

# 预定义的风格模型，确保文件路径正确
STYLE_MODELS = {
    "Cuphead": "cuphead_10000.pth",
    "Starry Night": "starry_night_10000.pth",
    "Mosaic": "mosaic_10000.pth",
}

# 清理缓存的功能
def clear_cache():
    """清理临时缓存文件"""
    # 删除风格迁移后的图像
    output_path = "stylized_output.jpg"
    if os.path.exists(output_path):
        os.remove(output_path)
        print("缓存已清理：", output_path)

    # 你可以在这里添加其他的缓存文件夹或文件路径进行清理

def schedule_cache_cleanup():
    """定时清理缓存，每小时执行一次"""
    schedule.every(1).hours.do(clear_cache)

    while True:
        schedule.run_pending()
        time.sleep(1)

# 启动定时清理线程
def start_cleanup_thread():
    cleanup_thread = threading.Thread(target=schedule_cache_cleanup)
    cleanup_thread.daemon = True  # 设置为守护线程
    cleanup_thread.start()

# 加载模型并设置设备
def load_model(style_name):
    """加载所选的风格模型。"""
    if style_name not in STYLE_MODELS:
        st.error(f"风格模型 {style_name} 未找到！")
        return
    model_path = STYLE_MODELS[style_name]
    
    # 确保模型路径存在
    if not os.path.exists(model_path):
        st.error(f"模型文件 {model_path} 未找到，请检查路径。")
        return
    
    # 加载模型
    transformer.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    transformer.eval()

# Streamlit 页面 UI
def main():
    # 启动定时清理缓存的线程
    start_cleanup_thread()

    st.title("快速图像风格迁移 APP")

    st.sidebar.header("上传和设置")

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
            with st.spinner("正在飞速运转..."):
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
                st.info("长按或者右键可以保存")

if __name__ == "__main__":
    main()
