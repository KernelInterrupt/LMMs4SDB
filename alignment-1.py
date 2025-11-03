import rasterio
import numpy as np

# --- 请修改为您的两个文件的真实路径 ---
satellite_image_path = './tampa_bay_s2_final_subset.tif'
bathymetry_data_path = 'exportImage.tiff'
# ------------------------------------

print("正在验证数据对齐状态...")

try:
    with rasterio.open(satellite_image_path) as sat_src:
        with rasterio.open(bathymetry_data_path) as bathy_src:
            
            # 1. 验证坐标参考系统 (CRS)
            crs_match = sat_src.crs == bathy_src.crs
            print(f"坐标参考系统 (CRS) 是否匹配: {crs_match}")
            if not crs_match:
                print(f"  - 卫星图 CRS: {sat_src.crs}")
                print(f"  - 水深图 CRS: {bathy_src.crs}")

            # 2. 验证地理变换参数 (Transform)，它包含了分辨率和原点信息
            # 我们使用 aclose 来比较浮点数，因为可能会有微小的精度差异
            transform_match = np.allclose(sat_src.transform, bathy_src.transform)
            print(f"地理变换参数 (Transform) 是否匹配: {transform_match}")
            if not transform_match:
                 print(f"  - 卫星图 Transform: {sat_src.transform}")
                 print(f"  - 水深图 Transform: {bathy_src.transform}")

            # 3. 验证维度 (Shape)
            shape_match = sat_src.shape == bathy_src.shape
            print(f"数据维度 (Shape) 是否匹配: {shape_match}")
            if not shape_match:
                print(f"  - 卫星图 Shape: {sat_src.shape}")
                print(f"  - 水深图 Shape: {bathy_src.shape}")
            
            print("-" * 50)
            if crs_match and transform_match and shape_match:
                print("恭喜！您的两个数据集已完美对齐，可以直接使用！")
            else:
                print("数据未完全对齐。请继续执行第二步进行重采样。")

except Exception as e:
    print(f"发生错误: {e}")