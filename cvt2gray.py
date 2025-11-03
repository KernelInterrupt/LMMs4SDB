from PIL import Image
import rasterio
from pathlib import Path

# ==========================================================
# =================== 在这里配置您的文件路径 ===================
# ==========================================================

# 1. 输入：您用图像编辑模型生成的、尺寸不匹配的图
GENERATED_IMAGE_PATH = Path('gpt-image.png')

# 2. 模板：我们的“标准答案”，它的尺寸是我们的目标
TEMPLATE_GEOTIFF_PATH = Path('bathymetry_aligned.tif')

# 3. 输出：一个全新的、尺寸对齐的、科学灰度图
OUTPUT_ALIGNED_GRAYSCALE_PATH = Path('gpt-output.png')

# ==========================================================

def align_and_convert(generative_path, template_path, output_path):
    """
    将生成的图像转换为灰度，并将其尺寸强制对齐到模板文件的尺寸。
    """
    print(f"--- 正在预处理 '{generative_path.name}' ---")
    
    if not generative_path.exists() or not template_path.exists():
        raise FileNotFoundError("错误：找不到输入或模板文件！")

    try:
        # 1. 从模板文件获取目标尺寸
        with rasterio.open(template_path) as template_src:
            target_shape = template_src.shape  # 格式: (高度, 宽度)
            print(f"目标尺寸 (高, 宽): {target_shape}")

        # 2. 打开生成的图像并转换为灰度
        with Image.open(generative_path) as img:
            print(f"原始生成图像尺寸 (宽, 高): {img.size}")
            
            # 转换为科学灰度图 ('L' mode)
            grayscale_img = img.convert('L')
            
            # 3. 这是最关键的一步：强制缩放图像
            # Pillow的resize方法需要 (宽度, 高度) 格式，与rasterio的shape相反
            target_size_pil = (target_shape[1], target_shape[0])
            
            # 使用LANCZOS插值算法，这是高质量缩放的最佳选择之一
            aligned_img = grayscale_img.resize(target_size_pil, Image.Resampling.LANCZOS)
            
            print(f"图像已成功缩放到: {aligned_img.size}")
            
            # 4. 保存最终的、对齐的灰度图
            aligned_img.save(output_path)
            
            print("-" * 50)
            print("成功！")
            print(f"对齐后的灰度图已保存到: {output_path}")
            print("-" * 50)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == '__main__':
    align_and_convert(GENERATED_IMAGE_PATH, TEMPLATE_GEOTIFF_PATH, OUTPUT_ALIGNED_GRAYSCALE_PATH)