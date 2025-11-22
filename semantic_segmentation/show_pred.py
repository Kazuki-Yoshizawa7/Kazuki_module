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
                DF3['difference'] = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}'])

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
    Density plotからDensity同士の差を計算して、閾値を設定し、そこからErrorの領域を特定するコード
    """

    def kde_analysis(self,row_start,row_end,type):
        

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
                DF3['difference'] = abs(DF3[f'class_{pred}'] - DF3[f'class_{label}'])

                fig, ax = plt.subplots(2,1,figsize=(10,18))


                combined_data = pd.DataFrame({
                    'Correct Predictions': DF1['difference'],
                    f'Errors with Label {label}': DF2['difference'],
                    'Errors with other Labels': DF3['difference']
                })
                sns.kdeplot(data=combined_data, fill=True, common_norm=False, alpha=0.5, ax=ax[0])

                """
                密度差を取得する 
                """  

                # Instance 作成
                DF1_KDE = gaussian_kde(DF1['difference'])
                DF2_KDE = gaussian_kde(DF2['difference'])
                DF3_KDE = gaussian_kde(DF3['difference'])

                # x軸の範囲を設定
                x_min, x_max = combined_data.min() - 0.1, combined_data.max() + 0.1
                x_range = np.linspace(x_min, x_max, 50)

                P_DF1 = DF1_KDE(x_range)
                P_DF2 = DF2_KDE(x_range)
                P_DF3 = DF3_KDE(x_range)


                """
                ここからPlotしていきたい→ 閾値となる点を探したい
                """


                


            
        

    