import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def parse_dat_file(file_path, use_cols, col_names):
    """
    Robustly parses .dat files by finding the data block and selecting columns by position.
    """
    try:
        # Find the line number where the data starts
        start_line = 0
        # Use 'latin-1' encoding to handle special characters from measurement software
        with open(file_path, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if line.strip() == ":START":
                    start_line = i + 1
                    break
        
        if start_line == 0:
            raise ValueError(":START marker not found in file.")

        df = pd.read_csv(
            file_path,
            sep=r'\s+',              # Use raw string to avoid SyntaxWarning
            skiprows=start_line,
            header=None,
            encoding='latin-1',      # Also specify encoding here for pandas
            usecols=use_cols
        )
        df.columns = col_names
        return df

    except Exception as e:
        print(f"Error parsing file {Path(file_path).name}: {e}")
        return None

def correct_flatness(raw_flatness_data, pitch_data_arcsec, roll_data_arcsec, offset_x, offset_y):
    """Removes pitch and roll errors from a raw flatness measurement."""
    # Convert arcseconds to degrees (1 degree = 3600 arcseconds)
    pitch_deg = np.asarray(pitch_data_arcsec) / 3600.0
    roll_deg = np.asarray(roll_data_arcsec) / 3600.0

    # Convert degrees to radians for use in trig functions
    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)

    # Calculate the linear error
    error_from_pitch = offset_x * np.tan(pitch_rad)
    error_from_roll = offset_y * np.tan(roll_rad)

    return np.asarray(raw_flatness_data) - error_from_pitch - error_from_roll

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. DEFINE YOUR SETUP PARAMETERS (MUST BE MEASURED) ---
    PITCH_LEVER_ARM_X = 60.0  # mm, longitudinal distance from pitch axis
    ROLL_LEVER_ARM_Y = 120.0 # mm, lateral distance from roll axis

    # --- 2. GET FILE PATHS USING A FILE DIALOG ---
    root = tk.Tk()
    root.withdraw()

    print("Opening file dialog to select flatness data...")
    flatness_file = filedialog.askopenfilename(title="Select FLATNESS .dat File")
    if not flatness_file:
        print("No flatness file selected. Exiting.")
        exit()

    print(f"Selected: {Path(flatness_file).name}")
    print("\nOpening file dialog to select angular data...")
    angular_file = filedialog.askopenfilename(title="Select ANGULAR .dat File")
    if not angular_file:
        print("No angular file selected. Exiting.")
        exit()
        
    print(f"Selected: {Path(angular_file).name}")

    # --- 3. PARSE AND PROCESS THE DATA ---
    try:
        flatness_df = parse_dat_file(flatness_file, use_cols=[0, 1], col_names=['Position', 'Data'])
        angular_df = parse_dat_file(angular_file, use_cols=[0, 2, 3], col_names=['Position', 'Pitch', 'Roll'])
        
        if flatness_df is None or angular_df is None:
            raise ValueError("Failed to parse one or more files. Exiting.")

        angular_avg_df = angular_df.groupby('Position').mean().reset_index()

        flatness_pos = flatness_df['Position'].values
        raw_flatness = flatness_df['Data'].values
        angular_pos = angular_avg_df['Position'].values
        pitch_data = angular_avg_df['Pitch'].values
        roll_data = angular_avg_df['Roll'].values

        interp_pitch = np.interp(flatness_pos, angular_pos, pitch_data)
        interp_roll = np.interp(flatness_pos, angular_pos, roll_data)

        corrected_for_angles = correct_flatness(
            raw_flatness_data=raw_flatness,
            pitch_data_arcsec=interp_pitch,
            roll_data_arcsec=interp_roll,
            offset_x=PITCH_LEVER_ARM_X,
            offset_y=ROLL_LEVER_ARM_Y
        )

        # --- 4. SLOPE REMOVAL ---
        slope, intercept = np.polyfit(flatness_pos, corrected_for_angles, 1)
        best_fit_line = slope * flatness_pos + intercept
        final_flatness = corrected_for_angles - best_fit_line

        # --- 5. VISUALIZE THE RESULTS ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(flatness_pos, raw_flatness, 'o-', label='Raw Measured Data', color='red', alpha=0.7)
        ax.plot(flatness_pos, corrected_for_angles, 'o-', label='Corrected for Pitch/Roll', color='orange', alpha=0.7)
        ax.plot(flatness_pos, final_flatness, 'o-', label='Final Flatness (Slope Removed)', color='blue', linewidth=2)

        ax.set_title('Flatness Measurement Correction', fontsize=16)
        ax.set_xlabel('Stage Position (mm)', fontsize=12)
        ax.set_ylabel('Deviation (microns)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Convert units to microns for reporting
        raw_ptp_microns = np.ptp(raw_flatness)
        corrected_ptp_microns = np.ptp(corrected_for_angles)
        final_ptp_microns = np.ptp(final_flatness)
        
        # --- NEW: Add results text box to the plot ---
        results_text = (
            f"Peak-to-Valley Results:\n"
            f"---------------------------------\n"
            f"Raw Data: {raw_ptp_microns:.4f} µm\n"
            f"After Correction: {corrected_ptp_microns:.4f} µm\n"
            f"Final Flatness: {final_ptp_microns:.4f} µm"
        )
        
        ax.text(0.05, 0.95, results_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # --- End of new section ---

        print("\n--- Results (in microns) ---")
        print(f"Data points processed: {len(flatness_pos)}")
        print(f"Initial raw peak-to-valley: {raw_ptp_microns:.4f} µm")
        print(f"After pitch/roll correction peak-to-valley: {corrected_ptp_microns:.4f} µm")
        print(f"Final slope-removed peak-to-valley: {final_ptp_microns:.4f} µm")

        plt.show()

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")