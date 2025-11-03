import rasterio
import numpy as np
from PIL import Image

# --- 1. 定义文件路径 ---
# 请修改为您的GeoTIFF文件的路径
input_geotiff_path = './tampa_bay_s2_final_subset.tif' 

# 您希望保存的PNG文件的路径和名称
output_png_path = 'output_rgb_image_for_model.png'

print(f"准备处理文件: {input_geotiff_path}")

# --- 2. 使用 rasterio 读取 GeoTIFF 文件 ---
try:
    with rasterio.open(input_geotiff_path) as dataset:
        print("GeoTIFF 文件成功打开。")
        
        # 读取所有波段 (B4, B3, B2) 到一个NumPy数组中
        # 顺序是 红, 绿, 蓝
        # 数组的形状将是 (通道数, 高度, 宽度)，例如 (3, 891, 800)
        raw_data = dataset.read()
        print(f"读取的数据维度: {raw_data.shape}")
        print(f"原始数据类型: {raw_data.dtype}")

        # --- 3. 这是最关键的一步：将科学数值“拉伸”到0-255的视觉范围 ---
        
        # 根据我们之前在GEE中反复试验得到的最佳可视化参数
        # 这些值决定了图像的“亮度和对比度”
        min_val = 0
        max_val = 1500 # 这是我们最终确定的、效果最好的'max'值
        
        print(f"正在将数值从 [{min_val}, {max_val}] 范围拉伸到 [0, 255]...")
        
        # 首先，使用np.clip将所有值限制在min_val和max_val之间
        # 这可以防止极端值（如云的亮点）影响整体图像的亮度
        clipped_data = np.clip(raw_data, min_val, max_val)
        
        # 其次，进行线性拉伸
        # (像素值 - 最小值) / (最大值 - 最小值) -> 得到0到1之间的浮点数
        # 然后乘以255 -> 得到0到255之间的浮点数
        scaled_data = ((clipped_data - min_val) / (max_val - min_val)) * 255
        
        # 最后，将数据类型转换为8位无符号整数 (uint8)
        # 这是图像文件所要求的标准格式
        rgb_data_uint8 = scaled_data.astype(np.uint8)
        print(f"数据已成功转换为 uint8 类型。")

        # --- 4. 调整数组维度以适应图像库的要求 ---
        
        # rasterio读取的顺序是 (通道, 高, 宽)
        # Pillow库需要的顺序是 (高, 宽, 通道)
        # 我们需要用np.transpose来交换轴的顺序
        rgb_data_hwc = np.transpose(rgb_data_uint8, (1, 2, 0))
        print(f"数组维度已调整为: {rgb_data_hwc.shape}")

        # --- 5. 使用Pillow创建并保存最终的RGB图像 ---
        
        # 从NumPy数组创建Pillow图像对象
        image = Image.fromarray(rgb_data_hwc)
        
        # 保存为PNG文件
        image.save(output_png_path)
        
        print("-" * 50)
        print("成功！")
        print(f"模型可用的RGB图像已保存到: {output_png_path}")
        print("-" * 50)

except FileNotFoundError:
    print(f"错误：找不到GeoTIFF文件。请检查路径是否正确: {input_geotiff_path}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")