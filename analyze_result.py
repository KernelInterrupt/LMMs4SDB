import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path

# ===================================================================
# ======================== CONFIGURATION ==========================
# ===================================================================

PREDICTED_DEPTH_PATH = Path('output_rgb_image_for_model_depth_gray.png')
GROUND_TRUTH_PATH = Path('bathymetry_aligned.tif')
OUTPUT_DIR = Path('./analysis_results')

# ===================================================================

def load_and_prepare_data(predicted_path, truth_path):
    """Loads, validates, and prepares the data for analysis."""
    print("--- 1. Loading and preparing data ---")
    if not predicted_path.exists() or not truth_path.exists():
        raise FileNotFoundError(f"Input file not found! Please check paths:\n  - {predicted_path}\n  - {truth_path}")

    with rasterio.open(predicted_path) as pred_src, rasterio.open(truth_path) as truth_src:
        if pred_src.shape != truth_src.shape:
            raise ValueError("Error: Shape mismatch between predicted image and ground truth!")
        
        predicted_array = pred_src.read(1)
        truth_array = truth_src.read(1)
        nodata_value = truth_src.nodata
    
    if nodata_value is not None:
        mask = truth_array != nodata_value
    else:
        mask = np.ones_like(truth_array, dtype=bool)
    
    predicted_flat = predicted_array[mask]
    truth_flat = truth_array[mask]
    
    print(f"Data loaded successfully. Found {len(truth_flat)} valid pixels for intersection analysis.")
    return predicted_array, truth_array, predicted_flat, truth_flat, mask

def perform_quantitative_analysis(predicted_flat, truth_flat):
    """Performs all quantitative calculations."""
    print("\n--- 2. Performing quantitative analysis ---")
    
    corr, p_value = pearsonr(predicted_flat, truth_flat)
    X = predicted_flat.reshape(-1, 1)
    reg = LinearRegression().fit(X, truth_flat)
    m, c = reg.coef_[0], reg.intercept_
    calibrated_predictions = reg.predict(X)
    rmse = np.sqrt(mean_squared_error(truth_flat, calibrated_predictions))
    mae = mean_absolute_error(truth_flat, calibrated_predictions)
    
    results = {
        "Pearson R": corr, "P-value": p_value, "RMSE (meters)": rmse,
        "MAE (meters)": mae, "Regression Slope (m)": m, "Regression Intercept (c)": c,
        "Calibrated Predictions": calibrated_predictions, "Predicted Flat": predicted_flat, "Truth Flat": truth_flat
    }
    
    print("Quantitative analysis complete.")
    return results

def save_report(results, output_dir):
    """Saves the analysis results to a text file."""
    print("\n--- 3. Saving analysis report ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*30 + "\n")
        f.write("  Analysis Report for Depth Prediction Model\n")
        f.write("="*30 + "\n\n")
        f.write(f"Pearson Correlation Coefficient (R): {results['Pearson R']:.4f}\n")
        f.write(f"P-value: {results['P-value']:.4g} (p < 0.001 indicates high statistical significance)\n\n")
        f.write("Linear Regression Calibration Equation:\n")
        f.write(f"  True Depth (meters) = {results['Regression Slope (m)']:.4f} * Predicted Value (0-255) + ({results['Regression Intercept (c)']:.4f})\n\n")
        f.write(f"Root Mean Square Error (RMSE): {results['RMSE (meters)']:.4f} meters\n")
        f.write(f"Mean Absolute Error (MAE): {results['MAE (meters)']:.4f} meters\n")
    
    print(f"Analysis report saved to: {report_path}")

def create_visualizations(predicted_array, truth_array, predicted_flat, truth_flat, mask, results, output_dir):
    """Creates and saves all publication-quality plots."""
    print("\n--- 4. Creating visualizations ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Side-by-side Comparison
    truth_masked = np.ma.masked_where(~mask, truth_array)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im1 = axes[0].imshow(truth_masked, cmap='viridis')
    axes[0].set_title('Ground Truth (NOAA Bathymetry)')
    fig.colorbar(im1, ax=axes[0], label="Depth (meters)")
    im2 = axes[1].imshow(predicted_array, cmap='gray')
    axes[1].set_title('Model Prediction ')
    fig.colorbar(im2, ax=axes[1], label="Relative Depth (0-255)")
    fig.suptitle('Ground Truth vs. Model Prediction', fontsize=18)
    plt.savefig(output_dir / 'comparison_side_by_side.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  - Saved: Side-by-side Comparison Plot")

    # Plot 2: 2D Density Scatter Plot (Hexbin)
    fig, ax = plt.subplots(figsize=(10, 10))
    hb = ax.hexbin(predicted_flat, truth_flat, gridsize=50, cmap='viridis', mincnt=1)
    ax.plot(predicted_flat, results['Calibrated Predictions'], color='red', linewidth=2, label='Linear Regression Fit')
    ax.set_title('2D Density Scatter Plot: Predicted vs. True Depth')
    ax.set_xlabel('Predicted Relative Depth (0-255)')
    ax.set_ylabel('True Depth (meters)')
    ax.legend()
    fig.colorbar(hb, ax=ax, label='Pixel Density')
    plt.savefig(output_dir / 'density_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  - Saved: 2D Density Scatter Plot")
    
    # Plot 3: Error Map
    error_map = np.full_like(truth_array, np.nan, dtype=float)
    error_map[mask] = truth_flat - results['Calibrated Predictions']
    error_masked = np.ma.masked_invalid(error_map)
    vmax = np.nanpercentile(np.abs(error_map), 98)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(error_masked, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title('Error Map (True - Predicted)')
    fig.colorbar(im, ax=ax, label='Error (meters) | Blue=Predicted Shallower, Red=Predicted Deeper')
    plt.savefig(output_dir / 'error_map.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  - Saved: Error Map")
    
    # Plot 4: Error Histogram
    errors = truth_flat - results['Calibrated Predictions']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, density=True, alpha=0.75)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f'Error Distribution Histogram (Mean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f})')
    ax.set_xlabel('Prediction Error (meters)')
    ax.set_ylabel('Probability Density')
    plt.savefig(output_dir / 'error_histogram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  - Saved: Error Histogram")

def main():
    """Main execution function"""
    try:
        predicted_array, truth_array, predicted_flat, truth_flat, mask = load_and_prepare_data(PREDICTED_DEPTH_PATH, GROUND_TRUTH_PATH)
        results = perform_quantitative_analysis(predicted_flat, truth_flat)
        save_report(results, OUTPUT_DIR)
        create_visualizations(predicted_array, truth_array, predicted_flat, truth_flat, mask, results, OUTPUT_DIR)
        print("\nAnalysis pipeline completed successfully! Please check the 'analysis_results' folder.")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main()