import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np  
import statsmodels.stats.multitest as smm
import re 






class LogiticRegression:
    def __init__(self, df): 

        if df.empty:
            raise ValueError("入力されたDataFrameが空です。")
        self.df = df

    """
    Categoryデータに関しては、変数をLevel0~4のようにRefが0となるような順番にコーディングすること: 
    """

    def logistic_regression_creator(self,continuous_col,categorical_col,target_col):
        
        sample_df = self.df[target_col+continuous_col+categorical_col]
        
        features = []
        features.extend(continuous_col)

        for col in categorical_col:
            features.append(f'C({col}, Treatment(reference=0))')

        formula = f'{target_col} ~ {" + ".join(features)}'
        print(f"Generated Formula: {formula}")

        model = sm.logit(formula = formula, data = sample_df)
        result = model.fit()

        result = model.fit()
        ci_exp = np.exp(result.conf_int())
        
        # ここからOrganizeしていく
        df_result = pd.DataFrame({
        'ORs': np.exp(result.params),
        'Coef (係数)': result.params,
        'P-value': result.pvalues,
        'CI_Upper':ci_exp[1],
        'CI_Lower':ci_exp[0],
        'Std.Err': result.bse
    
        })


        return ci_exp,df_result 


    """
    Formulaを事前に指定していれるVersion 
    """


    def logistic_regression_formula(self,formula):


        model = sm.Logit.from_formula(formula, data=self.df)
        result = model.fit()
        ci_exp = np.exp(result.conf_int())
        
        # ここからOrganizeしていく
        df_result = pd.DataFrame({
        'ORs': np.exp(result.params),
        'Coef (係数)': result.params,
        'P-value': result.pvalues,
        'CI_Upper':ci_exp[1],
        'CI_Lower':ci_exp[0],
        'Std.Err': result.bse
    
        })
        return df_result
    

    """
    Logistic RegressiongあとのDataFrame作成
    """


    def result_show(self,result):


        res = []
        for i in range(len(result)):
            r = result.iloc[i:i+1]
            index_name = r.index[0]

            if index_name == 'Intercept':
                continue

            find = re.findall(pattern, index_name)
            # res.append({'Variable': find[0][0],
            #             'Category': find[0][2],
            #             'Treatment': find[0][1],
            #             'ORs': r['ORs'].values[0],
            #             'CI_Lower': r['CI_Lower'].values[0],
            #             'CI_Upper': r['CI_Upper'].values[0],
            #             'P-value': r['P-value'].values[0]
            #             })
            ors_val = r['ORs'].values[0]
            lower_val = r['CI_Lower'].values[0]
            upper_val = r['CI_Upper'].values[0]
            p_val = r['P-value'].values[0]

            # P値の整形ロジック
            if p_val < 0.001:
                p_str = '< 0.001'
            else:
                p_str = f'{p_val:.3f}'

            
            res.append({
            'Variable': find[0][0],
            'Category': find[0][2],
            'Treatment': find[0][1],
            
            # :.3f をつけることで小数点3桁に固定
            # 形: "1.234 [0.900-1.500]"
            'ORs': f'{ors_val:.3f} [{lower_val:.3f}-{upper_val:.3f}]',
            'Odds': ors_val,
            'CI_Lower': lower_val,
            'CI_Upper': upper_val,
            # P値も同様に3桁にする場合
            'P-value': p_str
            })

        data = pd.DataFrame(res)

        """
        Bonferroni 
        """
        data['P-value']=pd.to_numeric(data['P-value'], errors='coerce')
        p_values = data['P-value']


        corrected_result = smm.multipletests(p_values, alpha=0.05, method='bonferroni')
        pval = corrected_result[1]
        data['Bonferroni_P-value'] = pval
        

        """
        Holm 
        """
        corrected_result = smm.multipletests(p_values, alpha=0.05, method='bonferroni')
        pval = corrected_result[1]

        data['Holm_P-value']=pval

        return data

    def forestplot(self, df):

        # errorbar 作成したい
        
        df['err_lower']=df['ORs']-df['CI_Lower']
        df['err_upper']=df['CI_Upper']-df['ORs']


        