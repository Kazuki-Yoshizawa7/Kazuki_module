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


class LogitClassify:

    def __init__(self,df,ranking_df):
        if df.empty:
            raise ValueError("入力されたDataFrameが空です。")
        
        # self.df = df
        # self.ranking_df = ranking_df
    

    def _logit_classify(self,df,ranking_df):


        def mis_logit_show(df,label,pred):
            df = df[df['label']==label]
            df = df[df['pred']==pred]

            l = []
            for i in range(len(df)):
                a = df.iloc[i]
                pred_l = a[f'class_{pred}']
                label_l = a[f'class_{label}']
                logit_diff = abs(pred_l -label_l)
                logit_ratio = abs(label_l/pred_l)
                l.append({'x':a['x'],
                        'y':a['y'],
                        'label':label,
                        'pred':pred,
                        'logit_difference':logit_diff,
                        'logit_ratio':logit_ratio})
            l = pd.DataFrame(l)
            fig,ax = plt.subplots(2,1,figsize=(10,6))
            ax[0].set_title(f'label={label} vs pred = {pred} Difference')
            ax[0].hist(l['logit_difference'],bins=30,color='blue',alpha=0.7)
            ax[1].set_title(f'label={label} vs pred = {pred} Ratio')
            ax[1].hist(l['logit_ratio'],bins=30,color='red',alpha=0.7)
            plt.tight_layout()
            # plt.show()
            
            return l
        def logit_display(df,p,start,end):
            p = p[start:end]
            
            # グラフ表示は結合せずに、ループ内で関数を呼び出すだけでOK
            for _, row in tqdm(p.iterrows(),total=len(p),desc="Displaying misclassified logits"):
                
                l = row['label']
                p = row['pred']
                _ = mis_logit_show(df,l,p) # mis_logit_show を呼び出すとグラフが表示される
                
            plt.show() # 全てのグラフを表示するために最後に plt.show() を呼び出す


        def mis_logit(df,label,pred):
            df = df[df['label']==label]
            df = df[df['pred']==pred]

            l = []
            for i in range(len(df)):
                a = df.iloc[i]
                pred_l = a[f'class_{pred}']
                label_l = a[f'class_{label}']
                logit_diff = abs(pred_l -label_l)
                logit_ratio = abs(label_l/pred_l)
                l.append({'x':a['x'],
                        'y':a['y'],
                        'label':label,
                        'pred':pred,
                        'logit_difference':logit_diff,
                        'logit_ratio':logit_ratio})
            l = pd.DataFrame(l)

            return l 
        


        # 修正された最終部分: 新しい関数を呼び出し、結合結果を変数に代入し、それを返す
        logit_display(df,ranking_df,0,10)
        
            

        def classifybydiff(result,logitdiff,pred,label):
            logit_range=logitdiff.max()-logitdiff.min()
            logit_mean = logitdiff.mean()
            if logit_range/2 > logitdiff.quantile(0.85):
                result.append({'label':label,'pred':pred,'ratio':'middle',
                                    'difference':'left'})
            elif logit_range/2 < logitdiff.quantile(0.15):
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'middle',
                                    'difference':'right'})
            else: 
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'middle',
                                    'difference':'middle'})

            return result 

        def classifybydiff_left(result,logitdiff,pred,label):
            logit_range=logitdiff.max()-logitdiff.min()
            logit_mean = logitdiff.mean()
            if logit_range/2 > logitdiff.quantile(0.85):
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'left',
                                    'difference':'left'})
            elif logit_range/2 < logitdiff.quantile(0.15):
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'left',
                                    'difference':'right'})
            else: 
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'left',
                                    'difference':'middle'})

            return result 

        def classifybydiff_right(result,logitdiff,pred,label):
            logit_range=logitdiff.max()-logitdiff.min()
            logit_mean = logitdiff.mean()
            if logit_range/2 > logitdiff.quantile(0.85):
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'right',
                                    'difference':'left'})
            elif logit_range/2 < logitdiff.quantile(0.15):
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'right',
                                    'difference':'right'})
            else: 
                result.append({'label':label,'pred':pred,'result':'not_normal','ratio':'right',
                                    'difference':'middle'})

            return result 



        def classify(df,l):
            g = l[0:20]
            result=[]
            for i in range(len(g)):
                label = g.iloc[i]['label']
                pred = g.iloc[i]['pred']
                data = mis_logit(df,label,pred)
                logitdiff = data['logit_difference']
                logit_ratio=data['logit_ratio']
                
                # result.append({'label':label,'pred':pred})
                #============== 一応標準化しているが、これは必要かどうかの検証
                ld = logitdiff.to_numpy()
                ld_reshaped = ld.reshape(-1, 1)
                standard_scaler = StandardScaler()
                standard_scaled_data = standard_scaler.fit_transform(ld_reshaped)
                #=============

                
                W, shapiro_p_value = stats.shapiro(standard_scaled_data)
                logitdiff_mean=logitdiff.mean()
                ratio_range=logit_ratio.max()-logit_ratio.min()


                if shapiro_p_value < 0.01:
                    # result.append({'W-value':W,'p-value':shapiro_p_value,'result':'not_normal'})
                    if ratio_range/2 > logit_ratio.quantile(0.85):
                        result.append({'label':label,'pred':pred,'W-value':W,
                                    'p-value':shapiro_p_value,'result':'not_normal','ratio':'left'})
                    elif ratio_range/2 < logit_ratio.quantile(0.15):
                        result.append({'label':label,'pred':pred,'W-value':W,
                                    'p-value':shapiro_p_value,'result':'not_normal','ratio':'right'})
                    else:
                        result.append({'label':label,'pred':pred,'W-value':W,
                                    'p-value':shapiro_p_value,'result':'not_normal','ratio':'middle'})
                        

                else:
                    result.append({'label':label,'pred':pred,'W-value':W,'p-value':shapiro_p_value,'result':'normal'})
            
            result=pd.DataFrame(result)
            pd.set_option('display.float_format', '{:.3f}'.format)

            
            return result 
        
        def classify_update(df,l):
            g = l[0:20]
            result=[]
            for i in range(len(g)):
                label = g.iloc[i]['label']
                pred = g.iloc[i]['pred']
                data = mis_logit(df,label,pred)
                logitdiff = data['logit_difference']
                logit_ratio=data['logit_ratio']
                
                # result.append({'label':label,'pred':pred})
                #============== 一応標準化しているが、これは必要かどうかの検証
                ld = logitdiff.to_numpy()
                ld_reshaped = ld.reshape(-1, 1)
                standard_scaler = StandardScaler()
                standard_scaled_data = standard_scaler.fit_transform(ld_reshaped)
                #=============

                
                W, shapiro_p_value = stats.shapiro(standard_scaled_data)
                logitdiff_mean=logitdiff.mean()
                ratio_range = logit_ratio.max()-logit_ratio.min()


                if shapiro_p_value < 0.01:
                    # result.append({'W-value':W,'p-value':shapiro_p_value,'result':'not_normal'})
                    if ratio_range/2 > logit_ratio.quantile(0.85):
                        result = classifybydiff_left(result,logitdiff,pred,label)
                    elif ratio_range/2 < logit_ratio.quantile(0.15):
                        result = classifybydiff_right(result,logitdiff,pred,label)
                    else:
                        result = classifybydiff(result,logitdiff,pred,label)
                        

                else:
                    result.append({'label':label,'pred':pred,'W-value':W,'p-value':shapiro_p_value,'result':'normal'})
            
            result=pd.DataFrame(result)
            pd.set_option('display.float_format', '{:.3f}'.format)

            
            return result 

        result_r = classify_update(df,ranking_df)

# 正規分布ではなく、Logit Diffの単純な形状から分けてた方が良さそう（Quantileを使ってRatioと同様に行う感じか）

        # ll,lm,lr,
# rl,rm,rr,
# ml,mm,mr,

        def classify(df): #ratio-differenceの順に組み合わせていることに注意
            for i in range(len(df)):
                r = df.iloc[i]['ratio']
                d = df.iloc[i]['difference']
                if r=='right':
                    if d=='right':
                        df.loc[i, 'type'] = 'rr'
                    elif d=='middle':
                        df.loc[i, 'type'] = 'rm'
                    elif d=='left':
                        df.loc[i, 'type'] = 'rl'

                elif r=='middle': # ratioがmiddleの場合
                    if d=='right':
                        df.loc[i, 'type'] = 'mr'
                    elif d=='middle':
                        df.loc[i, 'type'] = 'mm'
                    elif d=='left':
                        df.loc[i, 'type'] = 'ml'

                elif r=='left': # ratioがleftの場合
                    if d=='right':
                        df.loc[i, 'type'] = 'lr'
                    elif d=='middle':
                        df.loc[i, 'type'] = 'lm'
                    elif d=='left':
                        df.loc[i, 'type'] = 'll'
                
            return df

        result_update = classify(result_r)
        



    
        
        return result_update

# x = mis_logit(df,9,8)


    """

    TypeをGT、Object、Border,NAに分類する Typeに分類されたDataFrameを受け取ってさらに追加していく形

    """

    def error_pattern(self,df):
        types = df['type']
        object_error = set(['lm','ll'])
        gt_error = set(['rl'])
        border_error = set(['rm','mm','mr'])
        nan = set(['rr','lr','ml'])

        df['error_code']=df['type'].apply(lambda x: 'Object' if x in object_error else ('GT' if x in gt_error else ('Border' if x in border_error else 'NA')))

        return df

    





    
    