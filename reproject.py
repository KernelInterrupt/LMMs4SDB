import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path

# --- 请再次确认文件路径 ---
SATELLITE_IMAGE_PATH = Path('./tampa_bay_s2_final_subset.tif')
BATHYMETRY_DATA_PATH = Path('exportImage.tiff')
ALIGNED_BATHY_PATH = Path('bathymetry_aligned.tif') # 我们将要创建的、正确的文件

# 为我们的输出文件定义一个明确的、不会与真实数据冲突的无效值
# -9999 是一个非常安全和常用的选择
OUTPUT_NODATA_VALUE = -9999.0

print("开始执行修正后的重采样流程...")

# 以卫星图像为“基准”或“模板”
with rasterio.open(SATELLITE_IMAGE_PATH) as master:
    master_profile = master.profile
    master_crs = master.crs
    master_transform = master.transform
    master_width = master.width
    master_height = master.height

    # 以高分辨率水深数据为“源”
    with rasterio.open(BATHYMETRY_DATA_PATH) as slave:
        # 准备一个空的NumPy数组，但这次我们用我们定义的无效值来填充它
        aligned_array = np.full((master_height, master_width), OUTPUT_NODATA_VALUE, dtype=np.float32)

        # 执行重投影/重采样
        reproject(
            source=rasterio.band(slave, 1),
            destination=aligned_array,
            src_transform=slave.transform,
            src_crs=slave.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            # 指定当目标像素找不到源数据时，用什么值来填充
            dst_nodata=OUTPUT_NODATA_VALUE,
            resampling=Resampling.bilinear
        )
        
        # ******** 这是最关键的修正 ********
        # 在保存文件的profile中，明确地注册我们的nodata值
        master_profile.update(
            dtype=np.float32, 
            count=1,
            nodata=OUTPUT_NODATA_VALUE
        )
        # ***********************************

        # 将这个完美对齐、且元数据正确的数组，写入一个新的GeoTIFF文件
        with rasterio.open(ALIGNED_BATHY_PATH, 'w', **master_profile) as dst:
            dst.write(aligned_array, 1)

print("-" * 50)
print("成功！")
print(f"已创建一个元数据正确的、对齐的水深文件: {ALIGNED_BATHY_PATH}")
print("现在，您可以重新运行您的 'final_analysis.py' 脚本了。")