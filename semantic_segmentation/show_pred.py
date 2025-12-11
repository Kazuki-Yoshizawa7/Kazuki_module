import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import numpy as np
from torch.nn import functional as F

from scipy.ndimage import label, find_objects
import networkx as nx
import itertools

from scipy.stats import gaussian_kde # KDE計算用
from kazustat.statplot import KDE_SS
from scipy.optimize import brentq


"""
    ロジットのPred≡Label、その他などのアルゴリズム適用して可視化するコード:
    ”Classification情報が含まれたDataFrame"と元のDataFrame(ロジット情報こみ)を使う

    Instanceをまず設定して初期化すると、その後の関数でいちいちDFやDataFrameを引数として渡さなくて良い



    """

class WithoutGT:

    def __init__(self,df,dataframe): 
        if df.empty:
            raise ValueError("入力されたDataFrameが空です。")
        if dataframe.empty:
            raise ValueError("入力されたDataFrameが空です。")
        self.df = df
        self.dataframe = dataframe


    def logit_show(self,row_start,row_end,type): # 0~n 

        
        data = self.dataframe.iloc[row_start:row_end]

        for _, row in tqdm(data.iterrows(), total = len(data)):

            label = row['label']
            pred = row['pred']
            error_code = row['error_code']
            error_type = row['type']

            subset_df = self.df[self.df['ignore_error_flag'] == False]

            cols = [f'class_{i}' for i in range(0,19)]
            logit_2 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 1]
            logit_3 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 2]
            logit_4 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 3]
            subset_df['second_high']=logit_2
            subset_df['third_high']=logit_3
            subset_df['fourth_high']=logit_4



            DF1 = subset_df[(subset_df['pred'] == subset_df['label']) & (subset_df['pred']==pred) & (subset_df['label']==pred)] # 正解
            DF2 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']==label)] # エラーを起こしていて、ラベルが組み合わせ内でのLabel
            DF3 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']!=label)] # エラーを起こしていて、ラベルが組み合わせのLabel以外のもの


            class_values = DF1[cols].to_numpy()
            rows = np.arange(len(DF1))
            pred_indices = DF1['pred'].to_numpy()
            second_high_indices = DF1['second_high'].to_numpy()
            pred_prob = class_values[rows, pred_indices]
            second_prob = class_values[rows, second_high_indices]

            DF1['difference'] = 0.0
            DF1['ratio'] = 0.0  



            DF1['difference'] = np.where(DF1['ignore_error_flag']==False,
                                            abs(pred_prob - second_prob),
                                            DF1['difference'])
            
            DF1['ratio'] = np.where(DF1['ignore_error_flag']==False,
                                        abs(second_prob / pred_prob),
                                        DF1['ratio'])
    

            if type == 'difference':
                #DF1['difference'] = DF1[f'class_{pred}'] - DF1[f'class_{logit_2}'] # ２番目に高いLogitとの差をとっている
                DF2['difference'] = abs(DF2[f'class_{pred}'] - DF2[f'class_{label}'])
                # --- DF3 の計算 (ループ版) ---
                data_3_list = []
                for i in range(len(DF3)):
                    row = DF3.iloc[i]
                    p = row['pred']
                    l = row['label']
                    # その行ごとのPredとLabelの値を取得
                    val = abs(row[f'class_{p}'] - row[f'class_{l}'])
                    data_3_list.append(val)
                
                # ★ ここで代入します
                DF3['difference'] = np.array(data_3_list)
                
                #DF3['difference'] = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}'])

                fig, ax = plt.subplots(6,1, figsize=(10,18))
                
                ax[0].text(
                    x=-0.1,  # x座標: -0.1 (左端より少し外側)
                    y=1.4,  # y座標: 1.05 (上端より少し外側)          
                    s=f'{error_code} : {error_type}',
                    transform=ax[0].transAxes, 
                    fontsize=12,
                    fontweight='bold',
                    va='center',
                    ha='center' 
                )

                ax[0].text(
                    x=-0.1,  # x座標: -0.1 (左端より少し外側)
                    y=1.1,  # y座標:                   1.05 (上端より少し外側)          
                    s='A',      
                    transform=ax[0].transAxes, 
                    fontsize=16,
                    fontweight='bold',
                    va='center',
                    ha='center' 
                )
                ax[1].text(x=-0.1,y=1.1,s='B',transform=ax[1].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[2].text(x=-0.1,y=1.1,s='C',transform=ax[2].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')          
                ax[3].text(x=-0.1,y=1.1,s='D',transform=ax[3].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[4].text(x=-0.1,y=1.1,s='E',transform=ax[4].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[5].text(x=-0.1,y=1.1,s='F',transform=ax[5].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')

                sns.histplot(DF1['difference'], kde=True, ax=ax[0], color='green')
                ax[0].set_title(f'Correct Predictions: Pred = Label = {pred} ') 
                sns.histplot(DF2['difference'], kde=True, ax=ax[1], color='orange')
                ax[1].set_title(f'Errors with Label {label} Pred {pred}')
                sns.histplot(DF3['difference'], kde=True, ax=ax[2], color='red')
                ax[2].set_title(f'Errors with other Labels than {label}')
    
                ax[3].set_title(f'Combined View')
                combined_data = pd.DataFrame({
                    'Correct Predictions': DF1['difference'],
                    f'Errors with Label {label}': DF2['difference'],
                    'Errors with other Labels': DF3['difference']
                })
                sns.kdeplot(data=combined_data, fill=True, common_norm=False, alpha=0.5, ax=ax[3])  

                ax[4].set_title('Combined View (Histogram)')
                combined_data_hist = pd.DataFrame({
                    'Correct Predictions': DF1['difference'],
                    f'Errors with Label {label}': DF2['difference'],
                    'Errors with other Labels': DF3['difference']
                })
                sns.histplot(data=combined_data_hist, multiple="stack", ax=ax[4])




                ax[5].set_title('Total View')
                total_data = pd.concat([DF1['difference'], DF2['difference'], DF3['difference']], ignore_index=True)
                sns.histplot(total_data, kde=True, ax=ax[5], color='purple')  
                plt.tight_layout()      
                plt.show()
            
            elif type == 'ratio':
                #DF1['ratio'] = DF1[f'class_{logit_2}'] / DF1[f'class_{pred}']
                DF2['ratio'] = abs(DF2[f'class_{label}'] / DF2[f'class_{pred}'])
                DF3['ratio'] = abs(DF3[f'class_{label}'] / DF3[f'class_{pred}'])    

                fig, ax = plt.subplots(6,1, figsize=(10,18))

                ax[0].text(
                    x=-0.1,  # x座標: -0.1 (左端より少し外側)
                    y=1.4,  # y座標: 1.05 (上端より少し外側)          
                    s=f'{error_code} : {error_type}',
                    transform=ax[0].transAxes, 
                    fontsize=12,
                    fontweight='bold',
                    va='center',
                    ha='center' 
                )

                ax[0].text(
                    x=-0.1,  # x座標: -0.1 (左端より少し外側)
                    y=1.1,  # y座標: 1.05 (上端より少し外側)
                    s='A',  
                    transform=ax[0].transAxes, 
                    fontsize=16,
                    fontweight='bold',
                    va='center',
                    ha='center' 
                )
                ax[1].text(x=-0.1,y=1.1,s='B',transform=ax[1].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[2].text(x=-0.1,y=1.1,s='C',transform=ax[2].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[3].text(x=-0.1,y=1.1,s='D',transform=ax[3].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[4].text(x=-0.1,y=1.1,s='E',transform=ax[4].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')
                ax[5].text(x=-0.1,y=1.1,s='F',transform=ax[5].transAxes,fontsize=16,fontweight='bold',va='center',ha='center')  


                sns.histplot(DF1['ratio'], kde=True, ax=ax[0], color='green')
                ax[0].set_title(f'Correct Predictions: Pred = Label = {pred}') 
                sns.histplot(DF2['ratio'], kde=True, ax=ax[1], color='orange')
                ax[1].set_title(f'Errors with Label {label} Pred {pred}')
                sns.histplot(DF3['ratio'], kde=True, ax=ax  [2], color='red')
                ax[2].set_title(f'Errors with other Labels than {label}')
                plt.show()
                ax[3].set_title(f'Combined View')
                combined_data = pd.DataFrame({
                    'Correct Predictions': DF1['ratio'],
                    f'Errors with Label {label}': DF2['ratio'],
                    'Errors with other Labels': DF3['ratio']
                })
                sns.kdeplot(data=combined_data, fill=True, common_norm=False, alpha=0.5, ax=ax[3])


                ax[4].set_title(f'Combined View (Histogram)')
                combined_data_hist = pd.DataFrame({
                    'Correct Predictions': DF1['ratio'],
                    f'Errors with Label {label}': DF2['ratio'],
                    'Errors with other Labels': DF3['ratio']
                })
                sns.histplot(data=combined_data_hist, multiple="stack", ax=ax[4])


                ax[5].set_title('Total View')
                total_data = pd.concat([DF1['ratio'], DF2['ratio'], DF3['ratio']], ignore_index=True)
                sns.histplot(total_data, kde=True, ax=ax[5], color='purple')     
                plt.tight_layout()   
                plt.show()

    
    """
    Difference,ratioそれぞれについてDataFrameを作成する関数 -> 3つのDFを返している
    """
    def logit_return(self,row_start,row_end,type):



        
        data = self.dataframe.iloc[row_start:row_end]


        df_list = []
        for _, row in tqdm(data.iterrows(), total = len(data)):

            label = row['label']
            pred = row['pred']
            error_code = row['error_code']
            error_type = row['type']

            subset_df = self.df[self.df['ignore_error_flag'] == False]

            cols = [f'class_{i}' for i in range(0,19)]
            logit_2 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 1]
            logit_3 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 2]
            logit_4 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 3]
            subset_df['second_high']=logit_2
            subset_df['third_high']=logit_3
            subset_df['fourth_high']=logit_4



            DF1 = subset_df[(subset_df['pred'] == subset_df['label']) & (subset_df['pred']==pred) & (subset_df['label']==pred)] # 正解
            DF2 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']==label)] # エラーを起こしていて、ラベルが組み合わせ内でのLabel
            DF3 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']!=label)] # エラーを起こしていて、ラベルが組み合わせのLabel以外のもの


            class_values = DF1[cols].to_numpy()
            rows = np.arange(len(DF1))
            pred_indices = DF1['pred'].to_numpy()
            second_high_indices = DF1['second_high'].to_numpy()
            pred_prob = class_values[rows, pred_indices]
            second_prob = class_values[rows, second_high_indices]

            DF1['difference'] = 0.0
            DF1['ratio'] = 0.0  



            DF1['difference'] = np.where(DF1['ignore_error_flag']==False,
                                            abs(pred_prob - second_prob),
                                            DF1['difference'])
            
            DF1['ratio'] = np.where(DF1['ignore_error_flag']==False,
                                        abs(second_prob / pred_prob),
                                        DF1['ratio'])
    

            if type == 'difference':
                #DF1['difference'] = DF1[f'class_{pred}'] - DF1[f'class_{logit_2}'] # ２番目に高いLogitとの差をとっている
                DF2['difference'] = abs(DF2[f'class_{pred}'] - DF2[f'class_{label}'])
                # DF3['difference'] = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}'])
                # ここの部分の計算正確に！
                
                # --- DF3 の計算 (ループ版) ---
                data_3_list = []
                for i in range(len(DF3)):
                    row = DF3.iloc[i]
                    p = row['pred']
                    l = row['label']
                    # その行ごとのPredとLabelの値を取得
                    val = abs(row[f'class_{p}'] - row[f'class_{l}'])
                    data_3_list.append(val)
                
                # ★ ここで代入します
                DF3['difference'] = np.array(data_3_list)

                df_list.extend([DF1, DF2, DF3])


                
                
            elif type == 'ratio':
                #DF1['ratio'] = DF1[f'class_{logit_2}'] / DF1[f'class_{pred}']
                DF2['ratio'] = abs(DF2[f'class_{label}'] / DF2[f'class_{pred}'])
                DF3['ratio'] = abs(DF3[f'class_{label}'] / DF3[f'class_{pred}'])    

                df_list.extend([DF1, DF2, DF3])

            
        return df_list





    """
    Density plotを行う: statplot.pyから取得した関数でそれぞれのDFについてDensity Plotを出して交点を求めるような関数を作る
    (実用性はあまりないかも)
    → Densityの大小で閾値設定を行ってからやる方が良い
    """

    def dataframe_for_density_plot(self,row_start,row_end,type):


        data = self.dataframe.iloc[row_start:row_end]

        
        for _, row in tqdm(data.iterrows(), total = len(data)):

            label = row['label']
            pred = row['pred']
            error_code = row['error_code']
            error_type = row['type']

            subset_df = self.df[self.df['ignore_error_flag'] == False]

            cols = [f'class_{i}' for i in range(0,19)]
            logit_2 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 1]
            logit_3 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 2]
            logit_4 = np.argsort(subset_df[cols].values,axis=1)[:, ::-1][:, 3]
            subset_df['second_high']=logit_2
            subset_df['third_high']=logit_3
            subset_df['fourth_high']=logit_4



            DF1 = subset_df[(subset_df['pred'] == subset_df['label']) & (subset_df['pred']==pred) & (subset_df['label']==pred)] # 正解
            DF2 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']==label)] # エラーを起こしていて、ラベルが組み合わせ内でのLabel
            DF3 = subset_df[(subset_df['pred'] != subset_df['label']) &  (subset_df['pred']==pred) &(subset_df['label']!=label)] # エラーを起こしていて、ラベルが組み合わせのLabel以外のもの


            class_values = DF1[cols].to_numpy()
            rows = np.arange(len(DF1))
            pred_indices = DF1['pred'].to_numpy()
            second_high_indices = DF1['second_high'].to_numpy()
            pred_prob = class_values[rows, pred_indices]
            second_prob = class_values[rows, second_high_indices]

            DF1['difference_1'] = 0.0
            DF1['ratio_1'] = 0.0  



            DF1['difference_1'] = np.where(DF1['ignore_error_flag']==False,
                                            abs(pred_prob - second_prob),
                                            DF1['difference_1'])
            
            DF1['ratio_1'] = np.where(DF1['ignore_error_flag']==False,
                                        abs(second_prob / pred_prob),
                                        DF1['ratio_1'])
    

            
            if type == 'difference':
                
                data_1 = DF1['difference_1'].to_numpy()
                data_2 = abs(DF2[f'class_{pred}'] - DF2[f'class_{label}']).to_numpy()


                # ここの部分の計算正確に！

                data_3_list = []
                for i in range(len(DF3)):
                    row = DF3.iloc[i]
                    pred = row['pred']
                    label = row['label']

                    data_3_list.append(abs(row[f'class_{pred}'] - row[f'class_{label}']))
                data_3 = np.array(data_3_list)

                #data_3 = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}']).to_numpy()
            
                raw_datasets = [
                    {'data': data_1, 'label': 'difference_1'},
                    {'data': data_2, 'label': 'difference_2'},
                    {'data': data_3, 'label': 'difference_3'}
                ]

                # --- KDEと中央値の計算 ---
                kde_info = []
                all_data_concat = []

                for item in raw_datasets:
                    data = item['data']
                    # データが空の場合のエラーハンドリングが必要ならここに追加
                    median_val = np.median(data)
                    kde_func = gaussian_kde(data)
                    
                    kde_info.append({
                        'median': median_val,
                        'kde': kde_func,
                        'label': item['label'],
                        'data': data # プロット等で使うなら保持
                    })
                    all_data_concat.append(data)

                # --- 中央値順にソート (A < B < C) ---
                kde_info.sort(key=lambda x: x['median'])
                
                A, B, C = kde_info[0], kde_info[1], kde_info[2]

                # --- 描画用ドメインの定義 ---
                all_data = np.concatenate(all_data_concat)
                x_domain = np.linspace(all_data.min() - 1, all_data.max() + 1, 1000) # 点数を少し増やして精度向上

                # 描画用にdensityを計算しておきたい場合
                for item in kde_info:
                    item['density'] = item['kde'](x_domain)

                def find_optimal_intersection(kde_lower, kde_upper, median_lower, median_upper):
                    """
                    2つの分布の交点を求める。
                    ただし、探索範囲を「2つの中央値の間」に限定することで、
                    裾野のノイズによる誤検知を防ぐ。
                    """
                    # 差分関数
                    diff_func = lambda x: kde_lower(x) - kde_upper(x)
                    
                    # 中央値の間でグリッドサーチを行い、符号が反転する区間を探す
                    # 範囲を限定することで計算コスト削減とノイズ除去を行う
                    search_grid = np.linspace(median_lower, median_upper, 200)
                    diff_vals = diff_func(search_grid)
                    
                    # 符号が変わるインデックスを取得
                    sign_changes = np.where(np.diff(np.sign(diff_vals)))[0]
                    
                    intersections = []
                    for idx in sign_changes:
                        # グリッドの区間 [x0, x1]
                        x0, x1 = search_grid[idx], search_grid[idx+1]
                        
                        # scipy.optimize.brentq で高精度に根（交点）を探す
                        # 異符号であることが保証されているため収束する
                        try:
                            root = brentq(diff_func, x0, x1)
                            intersections.append(root)
                        except ValueError:
                            # 万が一収束しなかった場合は線形補間で代用
                            y0, y1 = diff_vals[idx], diff_vals[idx+1]
                            root = x0 - y0 / (y1 - y0) * (x1 - x0)
                            intersections.append(root)

                    if len(intersections) == 0:
                        # 交点が見つからない場合は、中間点を返すなどのフォールバック
                        return (median_lower + median_upper) / 2
                    
                    # 交点が複数ある場合、最も密度が高い点（＝最も確率が高い競合点）を採用する
                    # あるいは単純に平均をとるなど、目的に応じて変更可能
                    best_intersection = max(intersections, key=lambda x: kde_lower(x))
                    
                    return best_intersection

                # --- 交点の計算 ---
                # AとBの間の交点 (Aの右側裾とBの左側裾のクロス)
                intersection_AB = find_optimal_intersection(
                    A['kde'], B['kde'], A['median'], B['median']
                )
                
                # BとCの間の交点
                intersection_BC = find_optimal_intersection(
                    B['kde'], C['kde'], B['median'], C['median']
                )

                print(f'Intersection between A and B: {intersection_AB}')
                print(f'Intersection between B and C: {intersection_BC}')  
            
                

                # --- Density Plotの描画 ---
                plt.figure(figsize=(20, 10))
                plt.title('KDE Density Plots and Decision Boundaries (Intersections)')
                plt.xlabel('Difference Value')
                plt.ylabel('Density')
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # 1. 各分布のKDEカーブを描画
                colors = ['blue', 'green', 'red']
                labels_map = {A['label']: 'A', B['label']: 'B', C['label']: 'C'}

                for i, item in enumerate([A, B, C]):
                    plt.plot(x_domain, item['density'], 
                            label=f'Distribution {labels_map[item["label"]]} ({item["label"]})', 
                            color=colors[i], 
                            alpha=0.8)
                    
                    # 2. 各分布の中央値を描画
                    plt.axvline(item['median'], 
                                color=colors[i], 
                                linestyle=':', 
                                linewidth=1, 
                                label=f'Median {labels_map[item["label"]]} ({item["median"]:.2f})')
                    
                    # 3. 交点（決定境界）を強調して描画

                # 交点 A-B (Threshold 1)
                plt.axvline(intersection_AB, 
                            color='purple', 
                            linestyle='-', 
                            linewidth=2.5, 
                            label=f'Threshold A-B ({intersection_AB:.2f})')
                # 領域AとBの間の交点を塗りつぶしで強調
                y_inter_AB = A['kde'](intersection_AB)
                plt.plot(intersection_AB, y_inter_AB, 'o', color='purple', markersize=8)

                # 交点 B-C (Threshold 2)
                plt.axvline(intersection_BC, 
                            color='orange', 
                            linestyle='-', 
                            linewidth=2.5, 
                            label=f'Threshold B-C ({intersection_BC:.2f})')
                # 領域BとCの間の交点を塗りつぶしで強調
                y_inter_BC = B['kde'](intersection_BC) # A['kde']でもB['kde']でも値は同じはず
                plt.plot(intersection_BC, y_inter_BC, 's', color='orange', markersize=8)


                # 4. 凡例の表示
                plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize='small')
                plt.tight_layout(rect=[0, 0, 0.8, 1]) # 凡例を外に出すためにレイアウトを調整

                plt.show()



            elif type == 'ratio':
                #DF1['ratio'] = DF1[f'class_{logit_2}'] / DF1[f'class_{pred}']
                DF2['ratio_2'] = abs(DF2[f'class_{label}'] / DF2[f'class_{pred}'])
                DF3['ratio_3'] = abs(DF3[f'class_{label}'] / DF3[f'class_{pred}'])    

                
               
                data_1 = DF1['ratio_1'].to_numpy()
                data_2 = abs(DF2[f'class_{pred}'] - DF2[f'class_{label}']).to_numpy()
                data_3 = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}']).to_numpy()
            
                raw_datasets = [
                    {'data': data_1, 'label': 'ratio_1'},
                    {'data': data_2, 'label': 'ratio_2'},
                    {'data': data_3, 'label': 'ratio_3'}
                ]

                # --- KDEと中央値の計算 ---
                kde_info = []
                all_data_concat = []

                for item in raw_datasets:
                    data = item['data']
                    # データが空の場合のエラーハンドリングが必要ならここに追加
                    median_val = np.median(data)
                    kde_func = gaussian_kde(data)
                    
                    kde_info.append({
                        'median': median_val,
                        'kde': kde_func,
                        'label': item['label'],
                        'data': data # プロット等で使うなら保持
                    })
                    all_data_concat.append(data)

                # --- 中央値順にソート (A < B < C) ---
                kde_info.sort(key=lambda x: x['median'])
                
                A, B, C = kde_info[0], kde_info[1], kde_info[2]

                # --- 描画用ドメインの定義 ---
                all_data = np.concatenate(all_data_concat)
                x_domain = np.linspace(all_data.min() - 1, all_data.max() + 1, 1000) # 点数を少し増やして精度向上

                # 描画用にdensityを計算しておきたい場合
                for item in kde_info:
                    item['density'] = item['kde'](x_domain)

                def find_optimal_intersection(kde_lower, kde_upper, median_lower, median_upper):
                    """
                    2つの分布の交点を求める。
                    ただし、探索範囲を「2つの中央値の間」に限定することで、
                    裾野のノイズによる誤検知を防ぐ。
                    """
                    # 差分関数
                    diff_func = lambda x: kde_lower(x) - kde_upper(x)
                    
                    # 中央値の間でグリッドサーチを行い、符号が反転する区間を探す
                    # 範囲を限定することで計算コスト削減とノイズ除去を行う
                    search_grid = np.linspace(median_lower, median_upper, 200)
                    diff_vals = diff_func(search_grid)
                    
                    # 符号が変わるインデックスを取得
                    sign_changes = np.where(np.diff(np.sign(diff_vals)))[0]
                    
                    intersections = []
                    for idx in sign_changes:
                        # グリッドの区間 [x0, x1]
                        x0, x1 = search_grid[idx], search_grid[idx+1]
                        
                        # scipy.optimize.brentq で高精度に根（交点）を探す
                        # 異符号であることが保証されているため収束する
                        try:
                            root = brentq(diff_func, x0, x1)
                            intersections.append(root)
                        except ValueError:
                            # 万が一収束しなかった場合は線形補間で代用
                            y0, y1 = diff_vals[idx], diff_vals[idx+1]
                            root = x0 - y0 / (y1 - y0) * (x1 - x0)
                            intersections.append(root)

                    if len(intersections) == 0:
                        # 交点が見つからない場合は、中間点を返すなどのフォールバック
                        return (median_lower + median_upper) / 2
                    
                    # 交点が複数ある場合、最も密度が高い点（＝最も確率が高い競合点）を採用する
                    # あるいは単純に平均をとるなど、目的に応じて変更可能
                    best_intersection = max(intersections, key=lambda x: kde_lower(x))
                    
                    return best_intersection

                # --- 交点の計算 ---
                # AとBの間の交点 (Aの右側裾とBの左側裾のクロス)
                intersection_AB = find_optimal_intersection(
                    A['kde'], B['kde'], A['median'], B['median']
                )
                
                # BとCの間の交点
                intersection_BC = find_optimal_intersection(
                    B['kde'], C['kde'], B['median'], C['median']
                )

                print(f'Intersection between A and B: {intersection_AB}')
                print(f'Intersection between B and C: {intersection_BC}')  
            
                

                # --- Density Plotの描画 ---
                plt.figure(figsize=(20, 10))
                plt.title('KDE Density Plots and Decision Boundaries (Intersections)')
                plt.xlabel('Difference Value')
                plt.ylabel('Density')
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # 1. 各分布のKDEカーブを描画
                colors = ['blue', 'green', 'red']
                labels_map = {A['label']: 'A', B['label']: 'B', C['label']: 'C'}

                for i, item in enumerate([A, B, C]):
                    plt.plot(x_domain, item['density'], 
                            label=f'Distribution {labels_map[item["label"]]} ({item["label"]})', 
                            color=colors[i], 
                            alpha=0.8)
                    
                    # 2. 各分布の中央値を描画
                    plt.axvline(item['median'], 
                                color=colors[i], 
                                linestyle=':', 
                                linewidth=1, 
                                label=f'Median {labels_map[item["label"]]} ({item["median"]:.2f})')
                    
                    # 3. 交点（決定境界）を強調して描画

                # 交点 A-B (Threshold 1)
                plt.axvline(intersection_AB, 
                            color='purple', 
                            linestyle='-', 
                            linewidth=2.5, 
                            label=f'Threshold A-B ({intersection_AB:.2f})')
                # 領域AとBの間の交点を塗りつぶしで強調
                y_inter_AB = A['kde'](intersection_AB)
                plt.plot(intersection_AB, y_inter_AB, 'o', color='purple', markersize=8)

                # 交点 B-C (Threshold 2)
                plt.axvline(intersection_BC, 
                            color='orange', 
                            linestyle='-', 
                            linewidth=2.5, 
                            label=f'Threshold B-C ({intersection_BC:.2f})')
                # 領域BとCの間の交点を塗りつぶしで強調
                y_inter_BC = B['kde'](intersection_BC) # A['kde']でもB['kde']でも値は同じはず
                plt.plot(intersection_BC, y_inter_BC, 's', color='orange', markersize=8)


                # 4. 凡例の表示
                plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize='small')
                plt.tight_layout(rect=[0, 0, 0.8, 1]) # 凡例を外に出すためにレイアウトを調整

                plt.show()
