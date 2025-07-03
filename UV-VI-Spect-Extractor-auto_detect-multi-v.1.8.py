# -----------------------------------------------------------------------------
# UV-Vis Digitizer: Fixed 800x800 Display, ROI, Free Point Click, Spacebar
#
# Author: Mehdi Nikzad Semeskandi
# Brand: AiMatrixPedia
# Version: 1.8.0 # Updated version
# Date: May 18, 2025
#
# Description:
# Image is standardized to 800x800 for display and ROI selection.
# User selects ROI. Clicks key data points freely within ROI.
# Presses SPACEBAR to finish. Axes calibrated for ROI.
# Spectrum reconstructed, analyzed, exported.
#
# Contact: Mehdi.nikzad2@gmail.com
# -----------------------------------------------------------------------------

"""
UV-Vis Spectrum Digitizer Tool (Fixed 800x800 Display & ROI)

Developed by Mehdi Nikzad Semeskandi - AiMatrixPedia.
Image is displayed at a fixed 800x800 (padded). Define ROI on this view.
Freely click key data points in the ROI. Press SPACEBAR to confirm points.
For support or inquiries, please contact Mehdi.nikzad2@gmail.com.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import datetime

# --- Config ---
TARGET_DISPLAY_SIZE = 800 # For the square 800x800 display
INTERPOLATION_RESOLUTION_NM = 0.5
RECON_PEAK_PROMINENCE = 0.005
RECON_PEAK_MIN_HEIGHT = 0.01

_persistent_tk_root = None 

def get_screen_size():
    global _persistent_tk_root
    if _persistent_tk_root is None:
        _persistent_tk_root = tk.Tk(); _persistent_tk_root.withdraw()
    return _persistent_tk_root.winfo_screenwidth(), _persistent_tk_root.winfo_screenheight()

def resize_and_pad_image(img, target_size=(TARGET_DISPLAY_SIZE, TARGET_DISPLAY_SIZE), pad_color=(255, 255, 255)):
    """
    Resizes an image to fit within target_size maintaining aspect ratio, then pads to make it square.
    """
    h, w = img.shape[:2]
    th, tw = target_size

    # Calculate scaling factor
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale) # New width and height

    # Resize
    resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Create a new image with padding color and place the resized image in the center
    if len(resized_img.shape) == 2: # Grayscale
        padded_img = np.full(target_size, int(np.mean(pad_color)), dtype=np.uint8) # Use mean for grayscale pad
        is_gray = True
    else: # Color
        padded_img = np.full((th, tw, 3), pad_color, dtype=np.uint8)
        is_gray = False
    
    # Calculate top-left placement
    x_offset = (tw - nw) // 2
    y_offset = (th - nh) // 2

    if is_gray:
        padded_img[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized_img
    else:
        padded_img[y_offset:y_offset + nh, x_offset:x_offset + nw, :] = resized_img
        
    return padded_img


class SpectrumDigitizer:
    def __init__(self, image_roi_data, original_image_filename="source_image"):
        # image_roi_data is ALREADY the ROI cropped from the standardized 800x800 image
        self.img_raw_roi = image_roi_data 
        self.original_image_filename = original_image_filename
        self.img_display_roi, self.img_gray_roi = None, None
        self.roi_height, self.roi_width = 0, 0
        self._prepare_roi_images() # Prepares display/gray versions of the passed ROI

        self.defined_data_points_px_in_roi = []
        self.x_axis_pixel_map_points_roi, self.y_axis_pixel_map_points_roi = [], []
        self.x_axis_numerical_values_roi, self.y_axis_numerical_values_roi = [], []
        self.transform_x_roi, self.transform_y_roi = None, None
        self.m_x, self.c_x, self.m_y, self.c_y = 0,0,0,0
        self.calibrated_defined_points_wv_abs = []
        self.reconstructed_wavelength, self.reconstructed_absorbance = None, None
        self.final_peaks, self.final_shoulders, self.final_peak_ratios = [], [], {}

    def _prepare_roi_images(self): # This now just converts the passed ROI
        if self.img_raw_roi is None: raise ValueError("Image ROI data is None.")
        if len(self.img_raw_roi.shape) == 2: # Already grayscale
            self.img_gray_roi = self.img_raw_roi
            self.img_display_roi = cv2.cvtColor(self.img_raw_roi, cv2.COLOR_GRAY2BGR)
        elif len(self.img_raw_roi.shape) == 3: # Color
            # Assuming BGR if it came from cv2.imread -> resize_and_pad -> crop
            self.img_display_roi = cv2.cvtColor(self.img_raw_roi, cv2.COLOR_BGR2RGB)
            self.img_gray_roi = cv2.cvtColor(self.img_raw_roi, cv2.COLOR_BGR2GRAY)
        else: raise ValueError("Invalid image ROI data format.")
        self.roi_height, self.roi_width = self.img_gray_roi.shape[:2]
        print(f"ROI data assigned to digitizer ({self.roi_width}x{self.roi_height}).")

    def _get_pixel_input_for_calibration(self, num_points, title_message):
        if self.img_display_roi is None: raise ValueError("ROI Image not prepared.")
        print(f"\n{title_message} (on ROI). Click {num_points} point(s).")
        
        # Matplotlib window sizing can be simpler as ROI is from a known max size
        # Or keep dynamic sizing based on ROI's actual dimensions
        fig_w_inches = max(8, self.roi_width / 100) 
        fig_h_inches = max(6, self.roi_height / 100)
        
        fig, ax = plt.subplots(figsize=(fig_w_inches, fig_h_inches))
        # Centering attempt for Matplotlib window
        screen_w, screen_h = get_screen_size()
        try: 
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'geometry'):
                win_width_px = int(fig_w_inches * fig.dpi)
                win_height_px = int(fig_h_inches * fig.dpi)
                x_pos = (screen_w - win_width_px) // 2; y_pos = (screen_h - win_height_px) // 2
                fig_manager.window.geometry(f'{win_width_px}x{win_height_px}+{max(0,x_pos)}+{max(0,y_pos)}')
        except Exception: pass

        ax.imshow(self.img_display_roi); ax.set_title(f"{title_message} - ROI")
        plt.draw(); points = plt.ginput(num_points, timeout=0, show_clicks=True); plt.close(fig)
        if len(points) != num_points: raise ValueError(f"Expected {num_points}, got {len(points)}.")
        return [(float(p[0]), float(p[1])) for p in points]

    def _get_numerical_value(self, p_msg, v_type=float):
        while True:
            try: return v_type(input(p_msg))
            except ValueError: print(f"Invalid. Enter {v_type.__name__}.")

    def collect_defined_points_in_roi(self):
        print("\n--- Define Key Data Points in ROI ---")
        if self.img_display_roi is None: raise ValueError("ROI Image not prepared.")
        self.defined_data_points_px_in_roi = []
        
        fig_w_inches = max(10, self.roi_width / 80)
        fig_h_inches = max(7, self.roi_height / 80)
        fig, ax = plt.subplots(figsize=(fig_w_inches, fig_h_inches))
        screen_w, screen_h = get_screen_size()
        try: 
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'geometry'):
                win_width_px = int(fig_w_inches * fig.dpi); win_height_px = int(fig_h_inches * fig.dpi)
                x_pos = (screen_w - win_width_px) // 2; y_pos = (screen_h - win_height_px) // 2
                fig_manager.window.geometry(f'{win_width_px}x{win_height_px}+{max(0,x_pos)}+{max(0,y_pos)}')
        except Exception: pass

        ax.imshow(self.img_display_roi)
        line, = ax.plot([], [], 'ro-', markersize=5, alpha=0.8)
        title_obj = ax.set_title("Click Points: (L-Add, R-Undo Last). Press SPACEBAR to Finish.")

        def update_plot_and_title():
            if self.defined_data_points_px_in_roi:
                sorted_pts = sorted(self.defined_data_points_px_in_roi, key=lambda p: p[0])
                xs, ys = zip(*sorted_pts); line.set_data(xs, ys)
            else: line.set_data([],[])
            title_obj.set_text(f"Pts: {len(self.defined_data_points_px_in_roi)}. (L-Add, R-Undo Last). SPACE to Finish.")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                print("Clicked outside valid plot area. Try again."); return
            x, y = event.xdata, event.ydata
            if event.button == 1: self.defined_data_points_px_in_roi.append((x,y)); print(f"Add {len(self.defined_data_points_px_in_roi)}:({x:.1f},{y:.1f})")
            elif event.button == 3:
                if self.defined_data_points_px_in_roi: removed=self.defined_data_points_px_in_roi.pop(); print(f"Undo:({removed[0]:.1f},{removed[1]:.1f})")
                else: print("No points to undo.")
            update_plot_and_title()

        def on_key_press(event):
            if event.key == ' ': finish_collection()
        
        def finish_collection():
            fig.canvas.mpl_disconnect(cid_click); fig.canvas.mpl_disconnect(cid_key)
            plt.close(fig); print(f"Finished. {len(self.defined_data_points_px_in_roi)} pts collected.")
        
        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key_press)
        update_plot_and_title(); plt.show(block=True)
        return len(self.defined_data_points_px_in_roi) >= 2

    def calibrate_axes_for_roi(self):
        print("\n--- Interactive Axis Calibration (for ROI) ---")
        self.x_axis_pixel_map_points_roi = self._get_pixel_input_for_calibration(2, "X-AXIS (ROI): Click MIN X, then MAX X")
        self.x_axis_numerical_values_roi = [self._get_numerical_value("Enter numerical MIN X (wavelength): "),
                                            self._get_numerical_value("Enter numerical MAX X (wavelength): ")]
        self.y_axis_pixel_map_points_roi = self._get_pixel_input_for_calibration(2, "Y-AXIS (ROI): Click MIN Y, then MAX Y")
        self.y_axis_numerical_values_roi = [self._get_numerical_value("Enter numerical MIN Y (absorbance): "),
                                            self._get_numerical_value("Enter numerical MAX Y (absorbance): ")]
        self._create_roi_transform_functions(); return True

    def _create_roi_transform_functions(self):
        px1x, _=self.x_axis_pixel_map_points_roi[0]; px2x, _=self.x_axis_pixel_map_points_roi[1]
        v1x, v2x=self.x_axis_numerical_values_roi[0], self.x_axis_numerical_values_roi[1]
        if px1x==px2x: raise ValueError("ROI X-axis calib. pixels same.")
        self.m_x=(v2x-v1x)/(px2x-px1x); self.c_x=v1x-self.m_x*px1x
        self.transform_x_roi=lambda px:self.m_x*px+self.c_x
        _, py1y=self.y_axis_pixel_map_points_roi[0]; _, py2y=self.y_axis_pixel_map_points_roi[1]
        v1y, v2y=self.y_axis_numerical_values_roi[0], self.y_axis_numerical_values_roi[1]
        if py1y==py2y: raise ValueError("ROI Y-axis calib. pixels same.")
        self.m_y=(v2y-v1y)/(py2y-py1y); self.c_y=v1y-self.m_y*py1y
        self.transform_y_roi=lambda py:self.m_y*py+self.c_y
        print(f"ROI X-transform: wv={self.m_x:.4f}*px+{self.c_x:.4f}")
        print(f"ROI Y-transform: abs={self.m_y:.4f}*py+{self.c_y:.4f}")

    def _convert_defined_roi_pixels_to_wv_abs(self):
        if not self.defined_data_points_px_in_roi: print("No defined points in ROI to convert."); return False
        if not self.transform_x_roi or not self.transform_y_roi: print("ROI transforms not set."); return False
        self.calibrated_defined_points_wv_abs = []
        for px,py in self.defined_data_points_px_in_roi:
            try: self.calibrated_defined_points_wv_abs.append((self.transform_x_roi(px), self.transform_y_roi(py)))
            except Exception as e: print(f"Warn: Error converting ROI pixel ({px},{py}): {e}.")
        if not self.calibrated_defined_points_wv_abs: print("No ROI points converted."); return False
        self.calibrated_defined_points_wv_abs.sort(key=lambda p:p[0])
        print(f"Converted {len(self.calibrated_defined_points_wv_abs)} defined ROI points to Wv/Abs."); return True

    def reconstruct_curve_from_defined_points(self, resolution_nm=INTERPOLATION_RESOLUTION_NM):
        print("\n--- Reconstructing Curve from Defined Points (ROI) ---")
        if not self.calibrated_defined_points_wv_abs or len(self.calibrated_defined_points_wv_abs) < 2:
            print("Not enough defined points (<2) for spline."); return False
        unique_data = {};
        for wv, ab in self.calibrated_defined_points_wv_abs:
            if wv not in unique_data: unique_data[wv] = []
            unique_data[wv].append(ab)
        proc_pts = [(wv, np.mean(unique_data[wv])) for wv in sorted(unique_data.keys())]
        if len(proc_pts) < 2: print("Not enough unique defined pts for spline."); return False
        wv, ab = zip(*proc_pts); wv_arr, ab_arr = np.array(wv), np.array(ab)
        try:
            cs = CubicSpline(wv_arr, ab_arr)
            min_d, max_d = np.min(wv_arr), np.max(wv_arr)
            min_p = max(min_d, min(self.x_axis_numerical_values_roi))
            max_p = min(max_d, max(self.x_axis_numerical_values_roi))
            if min_p >= max_p: min_p, max_p = min_d, max_d
            if min_p >= max_p: print("Invalid interpolation range."); return False
            self.reconstructed_wavelength = np.arange(min_p, max_p + resolution_nm, resolution_nm)
            if len(self.reconstructed_wavelength)==0 or self.reconstructed_wavelength[-1] < max_p:
                self.reconstructed_wavelength = np.append(self.reconstructed_wavelength, max_p)
            self.reconstructed_wavelength = np.unique(self.reconstructed_wavelength)
            self.reconstructed_absorbance = cs(self.reconstructed_wavelength)
            print(f"Curve for ROI (defined points) reconstructed: {len(self.reconstructed_wavelength)} pts."); return True
        except ValueError as e: print(f"Spline Error (ROI defined): {e}\nWv: {wv_arr}"); return False

    def find_features_on_reconstructed_curve(self, peak_prominence=RECON_PEAK_PROMINENCE, peak_min_height=RECON_PEAK_MIN_HEIGHT):
        print("\n--- Finding Peaks/Shoulders on Reconstructed Curve (ROI) ---")
        if self.reconstructed_absorbance is None or len(self.reconstructed_absorbance) < 2:
            print("No reconstructed ROI data for final feature finding."); return
        res = (self.reconstructed_wavelength[1] - self.reconstructed_wavelength[0]) if len(self.reconstructed_wavelength) > 1 else 1
        min_peak_dist = int(max(1, 10 / res if res > 1e-9 else 1))
        peak_idx, _ = find_peaks(self.reconstructed_absorbance, prominence=peak_prominence, height=peak_min_height, distance=min_peak_dist)
        self.final_peaks = [(self.reconstructed_wavelength[i], self.reconstructed_absorbance[i]) for i in peak_idx]
        print(f"Identified {len(self.final_peaks)} final peaks (ROI):")
        for i, (wv,ab) in enumerate(self.final_peaks): print(f"  Peak {i+1}: Wv={wv:.2f} nm, Abs={ab:.4f} AU")
        if len(self.final_peaks) >= 2:
            s_peaks = sorted(self.final_peaks, key=lambda p:p[1], reverse=True)
            if s_peaks[1][1]>1e-6: self.final_peak_ratios[f"P({s_peaks[0][0]:.1f})/P({s_peaks[1][0]:.1f})"] = s_peaks[0][1]/s_peaks[1][1]
        if len(self.reconstructed_absorbance) > 3:
            grad1 = np.gradient(self.reconstructed_absorbance, self.reconstructed_wavelength)
            grad2 = np.gradient(grad1, self.reconstructed_wavelength)
            inflec_idx = np.where(np.diff(np.sign(grad2)))[0]
            print("\nPotential final shoulders (ROI):")
            for idx in inflec_idx:
                if idx + 1 < len(grad2):
                    wv_i,abs_i,slp_i = self.reconstructed_wavelength[idx],self.reconstructed_absorbance[idx],grad1[idx]
                    near_peak = any(abs(wv_i - p_wv) < 15 for p_wv, _ in self.final_peaks)
                    max_g1_abs = np.max(np.abs(grad1)) if len(grad1) > 0 else 0
                    if not near_peak and abs(slp_i) > max_g1_abs * 0.03 and (grad2[idx]*grad2[idx+1]<0):
                        self.final_shoulders.append((wv_i, abs_i)); print(f"  Shoulder: Wv={wv_i:.2f}, Abs={abs_i:.4f}")
            if not self.final_shoulders: print("No distinct final shoulders (ROI).")

    def export_to_excel(self, plot_index=0):
        if self.reconstructed_wavelength is None or len(self.reconstructed_wavelength) == 0:
            messagebox.showwarning("Export Warning", "No reconstructed data available to export.")
            print("No reconstructed ROI data to export.")
            return

        try:
            import openpyxl
        except ImportError:
            error_msg = "The 'openpyxl' library is required to export to .xlsx files.\n\nPlease install it by running:\npip install openpyxl"
            print(error_msg)
            messagebox.showerror("Dependency Error", error_msg)
            return

        base_name = os.path.splitext(os.path.basename(self.original_image_filename))[0]
        suffix = f"_plot{plot_index}" if plot_index > 0 else ""
        default_excel_name = base_name + suffix + "_extracted_defined_pts.xlsx"
        
        output_filename = filedialog.asksaveasfilename(
            initialfile=default_excel_name,
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title=f"Save Data for Plot {plot_index} As..."
        )

        if not output_filename:
            print("Excel export for ROI cancelled.")
            return

        try:
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                pd.DataFrame({
                    'Wavelength (nm)': np.round(self.reconstructed_wavelength, 1),
                    'Absorbance (AU)': np.round(self.reconstructed_absorbance, 4)
                }).to_excel(writer, sheet_name='Reconstructed Data', index=False)

                feat_data = []
                for wv, ab in self.final_peaks:
                    feat_data.append({'Feature': 'Peak', 'Wavelength (nm)': f"{wv:.2f}", 'Absorbance (AU)': f"{ab:.4f}", 'Details': 'From Spline'})
                for wv, ab in self.final_shoulders:
                    feat_data.append({'Feature': 'Shoulder', 'Wavelength (nm)': f"{wv:.2f}", 'Absorbance (AU)': f"{ab:.4f}", 'Details': 'From Spline'})
                for r, v in self.final_peak_ratios.items():
                    feat_data.append({'Feature': 'Peak Ratio', 'Wavelength (nm)': r, 'Absorbance (AU)': f"{v:.3f}", 'Details': ''})
                if not feat_data:
                    feat_data.append({'Feature': 'N/A', 'Wavelength (nm)': 'N/A', 'Absorbance (AU)': 'N/A', 'Details': ''})
                pd.DataFrame(feat_data).to_excel(writer, sheet_name='Identified Features', index=False)

                notes_data = {
                    'Note Type': ['Software Used', 'Developed By', 'Brand', 'Contact', 'Extraction Date', 'Source Image', 'Plot Index', 'Method'],
                    'Detail': [
                        'UV-Vis Spectrum Digitizer', 'Mehdi Nikzad Semeskandi', 'AiMatrixPedia',
                        'Mehdi.nikzad2@gmail.com', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.original_image_filename, str(plot_index), 'ROI (800px display), Free Point Click, Spacebar Finish'
                    ]
                }
                pd.DataFrame(notes_data).to_excel(writer, sheet_name='Processing Notes', index=False)

            success_msg = f"Data for ROI (Plot {plot_index}) exported successfully to:\n{os.path.abspath(output_filename)}"
            print(success_msg)
            messagebox.showinfo("Export Successful", success_msg)

        except Exception as e:
            error_details = f"An error occurred while trying to save the Excel file to:\n'{output_filename}'\n\nError: {e}\n\nPlease check file permissions and ensure the data is valid."
            print(error_details)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Excel Export Error", error_details)

    def display_results(self, plot_index=0):
        if self.reconstructed_wavelength is None or len(self.reconstructed_wavelength)==0: print("No ROI data to display."); return
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(self.reconstructed_wavelength, self.reconstructed_absorbance, 'b-', lw=1.5, label='Reconstructed Curve')
        if self.calibrated_defined_points_wv_abs:
            wv, ab = zip(*self.calibrated_defined_points_wv_abs); ax.plot(wv, ab, 'ro', ms=6, label='User-Defined Points')
        if self.final_peaks: wv, ab = zip(*self.final_peaks); ax.plot(wv, ab, 'g^', ms=9, label='Peaks (on Spline)')
        if self.final_shoulders: wv, ab = zip(*self.final_shoulders); ax.plot(wv, ab, 'ms', ms=7, label='Shoulders (on Spline)')
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance (AU)")
        ax.set_title(f"Spectrum (Plot {plot_index} from {os.path.basename(self.original_image_filename)}) - Defined Points")
        ax.legend(); ax.grid(True, alpha=0.6); plt.tight_layout(); plt.show()

    def run_roi_defined_points_workflow(self, plot_index=0):
        print(f"\n--- Starting Defined Points Workflow for ROI Plot {plot_index} ---")
        try:
            if not self.collect_defined_points_in_roi():
                 print("Point definition incomplete (<2 points). Cannot proceed for this ROI."); return False
            if not self.calibrate_axes_for_roi(): return False
            if not self._convert_defined_roi_pixels_to_wv_abs(): return False
            if not self.reconstruct_curve_from_defined_points(): return False
            self.find_features_on_reconstructed_curve()
            self.display_results(plot_index)
            self.export_to_excel(plot_index)
            print(f"\n--- Defined Points Workflow for ROI Plot {plot_index} COMPLETED ---"); return True
        except ValueError as ve: print(f"Error: {ve}"); return False
        except Exception as e: print(f"Critical Error: {e}"); import traceback; traceback.print_exc(); return False

# --- Main Application Start ---
if __name__ == '__main__':
    SCRIPT_AUTHOR="Mehdi Nikzad Semeskandi"; SCRIPT_BRAND="AiMatrixPedia"
    SCRIPT_VERSION="1.8.0"; SCRIPT_CONTACT="Mehdi.nikzad2@gmail.com" # Version for 800px display
    print("="*70 + f"\nUV-Vis Digitizer (800px Display, ROI, Free Click, Space Finish)".center(70) +
          f"\nDeveloped by: {SCRIPT_AUTHOR} ({SCRIPT_BRAND})".center(70) +
          f"\nVersion: {SCRIPT_VERSION} | Contact: {SCRIPT_CONTACT}".center(70) +
          f"\nDate: {datetime.date.today().strftime('%B %d, %Y')}".center(70) + "\n" + "="*70)

    _persistent_tk_root = tk.Tk(); _persistent_tk_root.withdraw() 

    main_app_running = True
    while main_app_running:
        full_image_path = filedialog.askopenfilename(master=_persistent_tk_root, 
            title="Select Main UV-Vis Spectrum Image",
            filetypes=[("Image Files","*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),("All Files","*.*")])
        if not full_image_path:
            print("No image selected.")
            if input("Exit program completely? (y/n): ").lower() == 'y': main_app_running = False
            continue 

        original_image_bgr = cv2.imread(full_image_path)
        if original_image_bgr is None: print(f"Error loading: {full_image_path}"); continue

        # Standardize the image for ROI selection display
        standardized_display_image_bgr = resize_and_pad_image(original_image_bgr)
        # All ROI selections will now happen on this standardized_display_image_bgr

        plot_idx_in_img = 0; process_rois_for_current_image = True
        while process_rois_for_current_image:
            plot_idx_in_img += 1
            print(f"\n--- Defining Graph Area (ROI) for Plot {plot_idx_in_img} from '{os.path.basename(full_image_path)}' ---")
            
            # ROI choice is now on the standardized 800x800 image
            roi_choice = input(f"Select ROI on the {TARGET_DISPLAY_SIZE}x{TARGET_DISPLAY_SIZE} display: (M)anually draw, or (A)ll of this {TARGET_DISPLAY_SIZE}x{TARGET_DISPLAY_SIZE} view? [M/A]: ").upper()
            roi_coords_on_standardized = (0,0,0,0) 

            if roi_choice == 'A':
                h_std, w_std = standardized_display_image_bgr.shape[:2]
                roi_coords_on_standardized = (0, 0, w_std, h_std) 
                print(f"Using entire {w_std}x{h_std} standardized view as ROI.")
            elif roi_choice == 'M':
                screen_w_main, screen_h_main = get_screen_size()
                # The standardized_display_image_bgr is already 800x800 (or TARGET_DISPLAY_SIZE)
                disp_w, disp_h = standardized_display_image_bgr.shape[1], standardized_display_image_bgr.shape[0]
                
                win_name = f"Select Graph Area (ROI Plot {plot_idx_in_img}) on {disp_w}x{disp_h} view. Drag, then ENTER/SPACE. ESC to cancel."
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE) 
                win_x = (screen_w_main - disp_w) // 2; win_y = (screen_h_main - disp_h) // 2
                cv2.moveWindow(win_name, max(0,win_x), max(0,win_y))
                print("Drag a rectangle. Press ENTER/SPACE to confirm, ESC to cancel.")
                # cv2.setWindowProperty(win_name,cv2.WND_PROP_TOPMOST,1) # Can be unstable
                selected_roi_display = cv2.selectROI(win_name, standardized_display_image_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(win_name)

                if selected_roi_display == (0,0,0,0): 
                    print("ROI selection cancelled for this plot."); plot_idx_in_img-=1
                    if input("Stop processing this image and select a new image file? (y/n): ").lower() == 'y':
                        process_rois_for_current_image = False 
                    continue 
                roi_coords_on_standardized = selected_roi_display # These are already for the 800x800 image
            else:
                print("Invalid choice. Skipping ROI selection for this plot."); plot_idx_in_img-=1; continue
            
            x,y,w,h = roi_coords_on_standardized
            if w==0 or h==0: print("Empty ROI obtained. Skipping."); plot_idx_in_img-=1; continue
            
            # Crop from the standardized_display_image_bgr
            roi_bgr_for_processing = standardized_display_image_bgr[y:y+h, x:x+w]

            print(f"Processing selected ROI for Plot {plot_idx_in_img}...")
            digitizer = SpectrumDigitizer(image_roi_data=roi_bgr_for_processing, original_image_filename=os.path.basename(full_image_path))
            if digitizer.run_roi_defined_points_workflow(plot_index=plot_idx_in_img): print(f"Plot {plot_idx_in_img} processed.")
            else: print(f"Issues with Plot {plot_idx_in_img}.")
            
            if input(f"\nProcess another area from '{os.path.basename(full_image_path)}' (on the {TARGET_DISPLAY_SIZE}x{TARGET_DISPLAY_SIZE} view)? (y/n): ").lower()!='y':
                process_rois_for_current_image = False 
        
        if not process_rois_for_current_image and plot_idx_in_img == 0: pass
        elif input("\nProcess a different image file? (y/n): ").lower()!='y':
            main_app_running = False 
            
    if _persistent_tk_root: _persistent_tk_root.destroy()
    print("Exiting. Thank you!");