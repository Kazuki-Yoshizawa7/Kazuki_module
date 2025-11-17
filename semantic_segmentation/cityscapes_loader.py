import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from tqdm import tqdm

from pathlib import Path  # Path オブジェクトを扱うための標準ライブラリ
import sys # Module 探索

# Moduleの探索を行いたい：

class Cityscapesloading:

    def __init__(self, image_path, label_path, pretrained_dict, model):
        self.image_path = image_path
        self.label_path = label_path
        self.pretrained_dict = pretrained_dict
        self.model = model

        # ルート探索
        self.REPO_ROOT = Path.cwd().resolve()
        self.LOCATE_ROOT = self._locate_root(self.REPO_ROOT)
        self.DSNET_ROOT = self.LOCATE_ROOT / "models"

        if not self.DSNET_ROOT.exists():
            raise FileNotFoundError(self.DSNET_ROOT)

        if str(self.DSNET_ROOT) not in sys.path:
            sys.path.insert(0, str(self.DSNET_ROOT))

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available : {torch.cuda.is_available()}")

    def _locate_root(self, start: Path) -> Path:
        candidates = [start] + list(start.parents)
        for base in candidates:
            candidate = base / "Semantic_model" / "External"
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError("Could not locate /Semantic_model/External directory")

    # ============================ 重み適用 =============================
    def applying_weight(self):
        if 'state_dict' in self.pretrained_dict:
            pretrained_dict_to_load = self.pretrained_dict['state_dict']
        else:
            pretrained_dict_to_load = self.pretrained_dict

        model_dict = self.model.state_dict()

        cleaned_dict = {
            k[6:]: v for k, v in pretrained_dict_to_load.items()
            if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        }

        model_dict.update(cleaned_dict)
        self.model.load_state_dict(model_dict, strict=False)
        print("モデルに重みを適用しました。")

    # ============================ 画像ロード・推論 =============================
    def load(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_path, cv2.IMREAD_GRAYSCALE)

        ignore_label = 255
        label_mapping = {
            -1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3,
            13: 4, 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label,
            31: 16, 32: 17, 33: 18
        }

        # ---- 内部関数 ----
        def convert_label(label, mapping):
            temp = label.copy()
            for k, v in mapping.items():
                label[temp == k] = v
            return label

        def input_transform(image):
            image = image.astype(np.float32)[:, :, ::-1]
            image = image / 255.0
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            image -= mean
            image /= std
            return image

        # ---- 変換 ----
        label = convert_label(label, label_mapping)
        image = input_transform(image)
        image = image.transpose((2, 0, 1))

        inp = torch.from_numpy(image).float().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            out = self.model(inp)

        # out[1] が logits
        pred = F.interpolate(out[1], size=image.shape[-2:], mode='bilinear', align_corners=True)
        pred_exp = pred.exp()
        pred_l = np.asarray(np.argmax(pred_exp.cpu(), axis=1), dtype=np.uint8)[0]

        # ---- DataFrame にする ----
        H, W = label.shape
        y_coords, x_coords = np.indices((H, W))
        logits = pred[0].cpu().numpy().transpose(1, 2, 0).reshape(-1, pred.shape[1])

        df = pd.DataFrame({
            "y": y_coords.ravel(),
            "x": x_coords.ravel(),
            "label": label.ravel(),
            "pred": pred_l.ravel(),
        })

        for i in range(pred.shape[1]):
            df[f"class_{i}"] = logits[:, i]

        return df, pred_l,label # logitを全て返す
    
    
    def _calculate_errors(self, df):
        df['pred_flag'] = (df['label'] != df['pred'])
        df['ignore_error_flag'] = (df['label'] == 255) & df['pred_flag']
        df['error'] = (df['label'] != df['pred']) & (~df['ignore_error_flag'])
        return df

    def find_error(self, df):
        df = self._calculate_errors(df)

        grid_pred  = df.pivot_table(index='y', columns='x', values='pred')
        grid_label = df.pivot_table(index='y', columns='x', values='label')
        grid_error = df.pivot_table(index='y', columns='x', values='error')

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        ax[0, 0].imshow(grid_pred, cmap='viridis')
        ax[0, 0].set_title('Predicted class')

        ax[0, 1].imshow(grid_label, cmap='viridis')
        ax[0, 1].set_title('True label')

        ignores = df[df['label'] == 255]
        if not ignores.empty:
            grid_ign = ignores.pivot_table(index='y', columns='x', values='label')
            ax[1, 0].imshow(grid_ign, cmap='gray')
        ax[1, 0].set_title('Ignore (255)')

        ax[1, 1].imshow(grid_error, cmap='Reds', vmin=0, vmax=1)
        ax[1, 1].set_title('Error locations')

        plt.tight_layout()
        plt.show()

    def error_table(self, df):
        df = self._calculate_errors(df)
        return df[df['error'] == True].copy()

    
