import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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





class LogitRanking:

    def __init__(self,df):
        if df.empty:
            raise ValueError("入力されたDataFrameが空です。")
        

        



    def _ranking_creation(self,df):
        lp = df[['x','y','label','pred','ignore_error_flag']]
        lp= lp[lp['ignore_error_flag']==False]

        # crosstab 
        cross = pd.crosstab(lp['label'],lp['pred'])
        ar = pd.crosstab(lp['label'],lp['pred']).to_numpy()
        
        def combi(cross):

            results_list = []
            for label, row_data in cross.iterrows():
        
                sorted_series = row_data.sort_values(ascending=False)
                
            
                results_list.append({
                    'label': label,
                    'pred_1': sorted_series.index[0],
                    'count_1': sorted_series.iloc[0],
                    'pred_2': sorted_series.index[1],
                    'count_2': sorted_series.iloc[1],
                    'pred_3': sorted_series.index[2],
                    'count_3': sorted_series.iloc[2],
                    'pred_4': sorted_series.index[3],
                    'count_4': sorted_series.iloc[3]
                })
            

            return pd.DataFrame(results_list)


        p = combi(cross)
    
        def ranking(p):
            l=[]
            for i in range(3):
                pp = p[['label',f"pred_{i+1}",f"count_{i+1}"]]
                pp = p.sort_values(by=f"count_{i+1}",ascending=False)
                
                for j in range(len(pp)):
                    x = pp.iloc[j]
                    if x['label']!= x[f'pred_{i+1}']:
                        l.append({'label':x['label'],'pred': x[f'pred_{i+1}'],'count':x[f'count_{i+1}']})

            l = pd.DataFrame(l).sort_values(by='count',ascending=False)
            
            return l 

    
        ranking_df = ranking(p)

        #DAG 
        daglist = ranking_df[['pred','label','count']]
        daglist_arr = daglist.to_numpy(list)
        # daglist[0:10]
        G = nx.from_pandas_edgelist(daglist[0:10], 'label', 'pred', create_using=nx.DiGraph())
        # グラフのノードとエッジを出力
        print("ノード:", G.nodes)
        print("エッジ:", G.edges)

        # グラフを可視化
        pos = nx.spring_layout(G)  # レイアウトの計算
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, edge_color='k', arrows=True)
        plt.title("DAG (Top10)")
        plt.show()



        return ranking_df


# --- ヘルパー関数 (ループの外で定義) ---

    def combi(self,cross):
        """
        Crosstabからラベルごとの予測トップ4を取得する
        """
        results_list = []
        for label, row_data in cross.iterrows():
            sorted_series = row_data.sort_values(ascending=False)
            
            preds = sorted_series.index
            counts = sorted_series.values
            
            # IndexErrorを回避するため、len()でチェック
            results_list.append({
                'label': label,
                'pred_1': preds[0] if len(preds) > 0 else None,
                'count_1': counts[0] if len(counts) > 0 else 0,
                'pred_2': preds[1] if len(preds) > 1 else None,
                'count_2': counts[1] if len(counts) > 1 else 0,
                'pred_3': preds[2] if len(preds) > 2 else None,
                'count_3': counts[2] if len(counts) > 2 else 0,
                'pred_4': preds[3] if len(preds) > 3 else None,
                'count_4': counts[3] if len(counts) > 3 else 0
            })
        return pd.DataFrame(results_list)

    def ranking(self,p):
        """
        combiの結果から、誤分類をカウント順にランク付けする
        """
        l=[]

        for i in range(3): 
            pred_col = f"pred_{i+1}"
            count_col = f"count_{i+1}"

            if pred_col not in p.columns or count_col not in p.columns:
                continue
                
            pp = p[['label', pred_col, count_col]].copy()
            pp = pp.sort_values(by=count_col, ascending=False)
            
            for j in range(len(pp)):
                x = pp.iloc[j]
            
                if pd.notna(x['label']) and pd.notna(x[pred_col]) and x['label'] != x[pred_col]:
                    l.append({'label':x['label'], 'pred': x[pred_col], 'count':x[count_col]})

        if not l:
            return pd.DataFrame(columns=['label', 'pred', 'count'])
            
        l = pd.DataFrame(l).sort_values(by='count', ascending=False)
        return l 



    def ranking_error(self,DF):
        
    
        lp_base = DF[['x','y','label','pred','ignore_error_flag','boundary','interior']]
        lp_base = lp_base[lp_base['ignore_error_flag']==False]

        interior = lp_base[lp_base['interior']==True]
        boundary = lp_base[lp_base['boundary']==True]


        results = {}

        for name, lp in [('interior', interior), ('boundary', boundary)]:
            
            print(f"--- Processing: {name} ---")

            # crosstab 
            cross = pd.crosstab(lp['label'],lp['pred'])
            
            if cross.empty:
                print(f"No data for {name}, skipping.")
                results[name] = pd.DataFrame(columns=['label', 'pred', 'count']) # 空のDFを格納
                continue

        
            
            p = self.combi(cross)
            ranking_df = self.ranking(p)

            results[name] = ranking_df

            #DAG 
            daglist = ranking_df[['pred','label','count']]
            

            if daglist.empty:
                print(f"No ranking errors found for {name}, skipping DAG.")
                continue
                
            G = nx.from_pandas_edgelist(daglist.head(10), 'label', 'pred', create_using=nx.DiGraph())
            
            print(f"ノード ({name}):", G.nodes)
            print(f"エッジ ({name}):", G.edges)

    
            pos = nx.spring_layout(G) 
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, edge_color='k', arrows=True)

            plt.title(f"DAG (Top10 Errors) - {name}")
            plt.show()

        return results
    

    

