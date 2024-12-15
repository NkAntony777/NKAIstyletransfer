import streamlit as st
from PIL import Image
import torch
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import os
import threading
import gc
import time


# 加载模型并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 缓存模型加载以提高性能
@st.cache_resource
def load_model_once(style_name):
    """加载所选的风格模型，仅加载一次。"""
    model_path = STYLE_MODELS[style_name]
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 预定义的风格模型，确保文件路径正确
STYLE_MODELS = {
    "Cuphead": "cuphead_10000.pth",
    "Starry Night": "starry_night_10000.pth",
    "Mosaic": "mosaic_10000.pth",
}

# 清理缓存的功能
def clear_cache():
    """清理临时缓存文件"""
    output_path = "stylized_output.jpg"
    if os.path.exists(output_path):
        os.remove(output_path)
        print("缓存已清理：", output_path)
    # 清理 PyTorch 缓存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 上传文件大小限制
MAX_FILE_SIZE_MB = 10  # 限制文件大小为 10MB

def validate_file_size(file):
    """验证文件大小是否符合要求"""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"文件过大，请上传小于 {MAX_FILE_SIZE_MB} MB 的文件！")
        return False
    return True

# 启动定时清理线程
def start_cleanup_thread():
    """定时清理缓存，每小时执行一次"""
    def cleanup_scheduler():
        while True:
            clear_cache()
            time.sleep(3600)  # 每小时执行一次

    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()

# 主程序
def main():
    # 启动清理线程
    start_cleanup_thread()

    st.title("快速图像风格迁移 APP")

    st.sidebar.header("上传和设置")

    # 风格选择
    style_name = st.sidebar.selectbox("选择风格", list(STYLE_MODELS.keys()))

    # 上传内容图像
    content_image_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    # 验证上传文件大小
    if content_image_file and not validate_file_size(content_image_file):
        return

    # 显示原始图像
    if content_image_file:
        content_image = Image.open(content_image_file)
        st.image(content_image, caption="原始图像", use_container_width=True)

    # 点击按钮进行风格迁移
    if st.sidebar.button("应用风格到图片"):
        if content_image_file:
            try:
                with st.spinner("正在飞速运转..."):
                    # 加载模型
                    transformer = load_model_once(style_name)

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

                    # 清理内存
                    clear_cache()

            except Exception as e:
                st.error(f"处理时发生错误：{e}")
                clear_cache()

if __name__ == "__main__":
    main()
