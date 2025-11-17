import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt, label
import seaborn as sns
from tqdm import tqdm

"""
このClassでは、ImデータとLabelデータでSobelフィルターをかけてプロットするところと、DataFrameとして保存するところ
まで行うことができる。大元のLogitdfとMergeする関数も最後につけている：
"""

class Sobelfilter:

    def __init__(self,pred_ar,label_ar):

        self.pred_ar = pred_ar
        self.label_ar = label_ar


    """
    SOBEL Filter Edge検出関数
    """  

    def edge_mask(self,image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        image = image.astype(np.float32)
        gradx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grady = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.abs(gradx) + np.abs(grady)
        return grad_mag > threshold
    

    def plot(self):

        pred_edges = self.edge_mask(self.pred_ar)
        label_img_display = self.label_ar.astype(float)
        label_img_display[label_img_display == 255] = np.nan
        label_edges = self.edge_mask(self.label_ar)
        error_mask = self.label_ar != self.pred_ar 
        boundary_errors = error_mask & label_edges
        interior_errors = error_mask & (~label_edges)
        ignore_mask = np.isnan(label_img_display)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].set_title('Label boundaries')
        axes[0].imshow(label_edges, cmap='gray')
        axes[0].axis('off')
        axes[1].set_title('Prediction boundaries')
        axes[1].imshow(pred_edges, cmap='gray')
        axes[1].axis('off')
        axes[2].set_title('Boundary errors (red) vs interior (blue)')
        vis = np.zeros((self.label_ar.shape[0], self.label_ar.shape[1], 3), dtype=np.float32) #RGBのカラーをこの後指定することができる
        vis[boundary_errors] = [1.0, 0.0, 0.0]
        vis[interior_errors] = [0.0, 0.0, 1.0]
        vis[ignore_mask] = [0.1,0.1,0.1]
        axes[2].imshow(vis)
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

    def edged_array_return(self):

        pred_edges = self.edge_mask(self.pred_ar)
        label_img_display = self.label_ar.astype(float)
        label_img_display[label_img_display == 255] = np.nan
        label_edges = self.edge_mask(self.label_ar)
        error_mask = self.label_ar != self.pred_ar 
        boundary_errors = error_mask & label_edges
        interior_errors = error_mask & (~label_edges)
        ignore_mask = np.isnan(label_img_display)

    
        vis = np.zeros((self.label_ar.shape[0], self.label_ar.shape[1], 3), dtype=np.float32) #RGBのカラーをこの後指定することができる
        # vis[boundary_errors] = [1.0, 0.0, 0.0]
        # vis[interior_errors] = [0.0, 0.0, 1.0]
        # vis[ignore_mask] = [0.1,0.1,0.1]

        H, W = self.label_ar.shape
        y_coords, x_coords = np.indices((H, W))

        df_data = {
            "y": y_coords.ravel(),
            "x": x_coords.ravel(),
            "label": self.label_ar.ravel(), # 既に (H, W) NumPy配列
            "pred": self.pred_ar.ravel(),# 既に (H, W) NumPy配列
            "boundary": boundary_errors.ravel(),
            "interior": interior_errors.ravel()
            

        }

        df_total = pd.DataFrame(df_data)

        return df_total
        