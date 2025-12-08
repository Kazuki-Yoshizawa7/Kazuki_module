"""
このファイルは、Kde Plotやその他のさまざまな計算を行い、Plotするための関数を詰めている
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats 
from scipy.stats import gaussian_kde

from scipy.optimize import brentq
from typing import List, Dict, Callable


"""
Semantic Segmentationで使えるようにしたDensityAnalysis Module
まず、すべてのピクセル情報が詰まったDFと、複数のDataFrameを扱う用のdf_listを与える必要がある
（df_list はHistgramで抽出したような間違いのパターンごとのDataFrameを想定）

"""

class DensityAnalysis:
    """
    このClassでは、DensityPlotに対してさまざまな分析を行えるようなものにする: 例えば、閾値設定をして区分するなど
    """
    def __init__(self,DF,df_list=None):

        self.DF = DF
        self.df_list = df_list  # 複数のDataFrameを扱う場合のリスト


    def _kde_calculator(self,df,column_name):

        data = df[column_name].to_numpy()
        kde = gaussian_kde(data)

        return kde
    

    ### FOR SEMANTIC SEGMENTATION 
    """
    複数のKDEについて処理する DF_list を想定　DF1,２、３、
    """

    def semantic_kde(self):
        
        kde_list = []
        for index_i, i in enumerate(self.df_list): # ★ 変更
            kde_func = self._kde_calculator(i, 'difference')
            
            x_points = np.linspace(0, 10, 1000)

            # 3. KDE関数を評価して、Y軸の値（密度のNumPy配列）を取得
            density_array = kde_func(x_points)

            j = index_i % 3 
            row = index_i // 3

            kde_list.append({f'DF_{j}':density_array,
                             'row':row})
            
        return kde_list
    
    



class PCA_analysis:
    """
    PCA解析を行うクラスで生物学統計やそのほかの分野でさまざまなところで使えるようにしている
    
    """


































"""
以下とりあえずは不要：
"""



class KDE_SS: # for Semantic Segmentation 

    def __init__(self,df,kde_columns): # ColumnでDf内のKDEを行いたいカラムを指定する 
        
        self.df = df
        self.kde_columns = kde_columns

    """
    1: Kde関数をとってそこから特定の範囲内で交点を求める計算
    入力は必ずDataFrameでカラムがあるもの『kde_columnがないとダメ]また、これらの入力はすべてDF中の同じ長さで、NaNが処理されていることが大前提
    """
    
    def _dict_creator(self):
        
        cols = self.kde_columns
        self.data_dict = {} 
        
        for col_name in cols:
            data_array = self.df[col_name].to_numpy()
            
           
            self.data_dict[col_name] = data_array
    
        return self.data_dict  
    


    def _kde_dict_creator(self):

        keys_to_process = self.kde_columns 

        for col_name in keys_to_process:
            # data_dictは _dict_creator で kde_columns に基づいて作成されている前提
            if col_name in self.data_dict: 
                data = self.data_dict[col_name]
                kde = gaussian_kde(data)
                # 辞書にKDE関数を追加
                self.data_dict[col_name + '_kde'] = kde
            else:
                # 処理すべきカラムがdata_dictに見つからなかった場合の警告
                print(f"Warning: Data for column '{col_name}' not found in data_dict.")
        
        return self.data_dict
        # for i in self.data_dict:
        #     data = self.data_dict[i]
        #     kde = gaussian_kde(data)
        #     self.data_dict[i+'_kde'] = kde
        
        # return self.data_dict



    def _get_sorted_median_pairs(self):
        

        
        median_list = []
        
        # data_dict に格納されている元のデータ配列に対して中央値を計算する
        for col_name in self.kde_columns:
            data_array = self.data_dict[col_name]
            median_value = np.median(data_array)
            
            # 中央値と対応するカラム名をリストに格納
            median_list.append({'col_name': col_name, 'median': median_value})


        median_df = pd.DataFrame(median_list)
        sorted_median_df = median_df.sort_values(by='median', ascending=True).reset_index(drop=True)
        
        pairs = []
        col_names = sorted_median_df['col_name'].tolist()
        for i in range(len(col_names) - 1):
            # 隣り合うカラム名 (col1, col2) をペアとする
            pairs.append((col_names[i], col_names[i+1]))
            
        return pairs 

    def create_difference_functions(self) -> List[Dict[str, Callable[[float], float]]]:
        """
        中央値順の隣り合うペアのKDE差分関数 (P(x) - Q(x)) を作成する
        """
        

        self._dict_creator()
        self._kde_dict_creator()
        
        sorted_pairs = self._get_sorted_median_pairs()
        
        difference_functions_list = []
        
        for col_P, col_Q in sorted_pairs:
            # 3. 対応するKDE関数を取得
            # P(x) が中央値が小さい方 (col_P)、Q(x) が大きい方 (col_Q)
            kde_P = self.data_dict[col_P + '_kde']
            kde_Q = self.data_dict[col_Q + '_kde']
            
            def difference_function(x, kde_P=kde_P, kde_Q=kde_Q):
                """f(x) = P(x) - Q(x)"""
                return kde_P(x) - kde_Q(x)
            
            difference_functions_list.append({
                'pair': f'{col_P} vs {col_Q}',
                'func': difference_function
            })

        return difference_functions_list


    def find_intersections_df(self) -> pd.DataFrame:
       
       """
       Difference_functions_listにあるもので、Differenceの符号が変わる区間をそれぞれ求めて、その間で交点の座標を出す
       """
       
       # data_dictからKdeを取り出して、Difference_functions_listの実装

       diff_func_list = self.create_difference_functions()

       intersection_results = []

       for item in diff_func_list:
           func = item['func']
           col_P = item['col_P']
           col_Q = item['col_Q']

           low = self.min(col_P)
           high = self.max(col_Q)

           try:
               # brentq: 指定区間 [low, high] で関数の符号が逆転する場合に根を返す
               # 交点では P(x) - Q(x) = 0 となる
               root = brentq(func, low, high)
               
               intersection_results.append({
                   'pair': item['pair_name'],
                   'col_lower': col_P,
                   'col_upper': col_Q,
                   'intersection_x': root,
                   'density_at_intersection': self.data_dict[col_P + '_kde'](root)[0] # 交点での確率密度
               })
               
           except ValueError:
               # 区間内で符号が変わらない（交点がない、または範囲外）場合
               print(f"Warning: No intersection found between medians for {col_P} and {col_Q}.")
               intersection_results.append({
                   'pair': item['pair_name'],
                   'col_lower': col_P,
                   'col_upper': col_Q,
                   'intersection_x': None,
                   'density_at_intersection': None
               })

       return pd.DataFrame(intersection_results)
       
    




        
    
        
        
