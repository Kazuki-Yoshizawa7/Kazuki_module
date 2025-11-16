import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt

# 連続変数の検定モジュール
# パラメトリックかノンパラメトリックかを判定してその流れで適切な検定方法を判断していく関数
"""
このclassモジュールで連続変数の検定ができるが、これらは基本的にNaNが存在しないことを想定しているので、事前に
DataFrameを綺麗にする必要がある
"""

class Continuous:
    """
    DataFrameを受け取り、指定された連続変数の正規性を判定し、
    パラメトリック/ノンパラメトリックに応じた統計量を計算するクラス。
    """

    # init 初期化でdfを受け取る
    def __init__(self, df: pd.DataFrame):
    
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be Pandas DataFrame")
        
        self.df = df
        print(f"DataFrame (shape={df.shape}) has been set")
        
        # 各カラムの検定結果をキャッシュする辞書
        self.results_cache = {}

    def check_normalitiy(self, col_name: str, alpha: float=0.05) -> Tuple[bool, float]:

        if col_name not in self.df.columns:
            raise ValueError(f"Variable '{col_name}' does not exist in DataFrame")

        # Shapiro検定

        data = self.df[col_name].dropna().values
        n = len(data)

        if n < 3:
            is_parametric = False
            p_value = np.nan
        
        # サンプルサイズが5000を超える場合 (Shapiro検定の警告)
        elif n > 5000:
            print(f"警告: カラム '{col_name}' のサンプルサイズ ({n}) が5000を超えているため、シャピロ・ウィルク検定の精度が落ちる可能性があります。")
            shapiro_stat, p_value = stats.shapiro(data)
            is_parametric = p_value > alpha
        
        else:
            # シャピロ・ウィルク検定 (H0: 正規分布に従う)
            shapiro_stat, p_value = stats.shapiro(data)
            is_parametric = p_value > alpha
        
        # 結果をキャッシュに保存
        self.results_cache[col_name] = {
            'n': n,
            'is_parametric': is_parametric,
            'shapiro_p_value': p_value
        }
        
        return (is_parametric, p_value)
    
    def _qqplot(self,col_names:  Union[str, List[str]], show=False):  # multiple columns and plot

        data = self.df[col_names].dropna()
        n_cols = len(data.columns)
        fig,ax=plt.subplots(n_cols,1,figsize=(8, 5 * n_cols))

        for ax, col in zip(ax, col_names):
            if data[col].empty:
                ax.text(0.5, 0.5, f"No valid data for {col}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"Q-Q Plot for {col}")
                continue

            sm.qqplot(data[col], fit=True, line="45", ax=ax)
            
            ax.set_title(f"Q-Q Plot for {col}")

        plt.tight_layout() # サブプロットが重ならないように調整

        # 7. show引数の処理
        if show:
            plt.show()

    # 呼び出し側でさらに編集できるよう、figureオブジェクトを返す
        return fig
    

    def summary(self,col_names,alpha=0.05,qqplot=False):

        # 各データについてNormalitycheckをする
        if isinstance(col_names, str):
            cols_to_check = [col_names]
        else:
            cols_to_check = col_names

        print(f"--- Normality Check Summary (Alpha={alpha}) ---")
        

        results_list = []
        for col in cols_to_check:
            try:
                # self で関数を呼び出す必要がある
                self.check_normalitiy(col, alpha=alpha)
                
                # check_normalitiyがキャッシュに保存した結果を取得
                result = self.results_cache[col]
                
                results_list.append({
                    'Column': col,
                    'N': result['n'],
                    'Shapiro p-value': result['shapiro_p_value'],
                    'Is Normal (Parametric)': result['is_parametric']
                })

            except ValueError as e:
                print(f"Skipping {col}: {e}")
            except Exception as e:
                print(f"Error processing {col}: {e}")

        if not results_list:
            print("No valid results to display.")
            return None

        summary_df = pd.DataFrame(results_list)
        summary_df = summary_df.set_index('Column')
        
        # p値の表示形式を揃える
        summary_df['Shapiro p-value'] = summary_df['Shapiro p-value'].apply(
            lambda p: f"{p:.4e}" if (p < 0.001 and not pd.isna(p)) else 
                    f"{p:.4f}" if not pd.isna(p) else 
                    "N/A (N<3)"
        )
        
        print(summary_df.to_string()) # .to_string() でコンソールに綺麗に表示
        
        if qqplot==True:
            try:
                self._qqplot(col_names,show=True)
            except ValueError as e:
                print(f'Value Error {e}')


        
        
        return summary_df
    
