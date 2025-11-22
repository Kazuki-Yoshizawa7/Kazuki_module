import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class GMMfunction:

    """
    GMMの実行を行う（連続データに関して）関数
    """


    def GMM_identifier(self,logitrat):
        n_components = np.arange(1, 10)
        gmm_diff = [GaussianMixture(n,n_init=3,random_state=42).fit(logitrat.values.reshape(-1, 1))
                    for n in tqdm(n_components, desc="Fitting GMMs", leave=True)]
        #gmm_diff.fit(logitdiff.values.reshape(-1, 1))
        #fig, ax = plt.subplots(figsize=(9,7))
        #ax.plot(n_components, [m.bic(logitrat.values.reshape(-1,1)) for m in gmm_diff], label='BIC')
        #ax.plot(n_components, [m.aic(logitrat.values.reshape(-1,1)) for m in gmm_diff], label='AIC')
        aic = [m.aic(logitrat.values.reshape(-1,1))for m in gmm_diff]
        bic = [m.bic(logitrat.values.reshape(-1,1))for m in gmm_diff]


        # AIC/BIC が最小となるインデックス (0-8) を見つける
        #best_aic_idx = np.argmin(aic)
        #best_bic_idx = np.argmin(bic)
        
        # より良い方のインデックスを採用
        #best_idx = min(best_aic_idx, best_bic_idx)
        
        #components = n_components[best_idx]

        try:
            aic_diff2 = np.diff(aic, n=2)
            bic_diff2 = np.diff(bic, n=2)

            # 2. 二階差分が「最大」になるインデックスを探す
            best_aic_idx = np.argmax(aic_diff2) + 1
            best_bic_idx = np.argmax(bic_diff2) + 1

            
            best_idx = min(best_aic_idx, best_bic_idx)

        except ValueError:
        
            best_idx = min(np.argmin(aic), np.argmin(bic))

        components = n_components[best_idx]
        
        return components

        # plt.legend(loc='best')
        # plt.xlabel('n_components')

    def GMM_classifier(self,df,tabelle):
        
        l = []
        df_error = df[df['error']==True]
            
        for i, row in tqdm(tabelle.iterrows(), total=len(tabelle), desc="GMM Classification"):
            
                # 'i' は元のインデックス, 'row' は行のデータ
            pred = row['pred']
            label = row['label']
            ratio = row['ratio']
            difference = row['difference']
            
            logitratio = abs(df_error[f'class_{label}'] / df_error[f'class_{pred}'])
            logitdiff = abs(df_error[f'class_{pred}'] - df_error[f'class_{label}'])
            t = row['type']
                
            if (ratio == 'middle') & (difference != 'middle'):
                components = self.GMM_identifier(logitratio)
                l.append({'Index': i, 'pred': pred, 'label': label, 'type': t, 'ratio_components': components,
                            'diff_components': 0})

            elif (ratio != 'middle') & (difference == 'middle'):
                components = self.GMM_identifier(logitdiff)
                l.append({'Index': i, 'pred': pred, 'label': label, 'type': t, 'ratio_components': 0,
                            'diff_components': components})

            elif (ratio == 'middle') & (difference == 'middle'):
                r_com = self.GMM_identifier(logitratio)
                d_com = self.GMM_identifier(logitdiff)
                l.append({'Index': i, 'pred': pred, 'label': label, 'type': t, 'ratio_components': r_com,
                            'diff_components': d_com})
                    
            else: 
                l.append({'Index': i, 'pred': pred, 'label': label, 'type': t, 'ratio_components': 0,
                            'diff_components': 0})
                
        return pd.DataFrame(l)



            






