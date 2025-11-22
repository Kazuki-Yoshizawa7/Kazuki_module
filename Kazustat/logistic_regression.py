import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np  





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
        return result
    



    def forestplot(self, df):

        # errorbar 作成したい
        
        df['err_lower']=df['ORs']-df['CI_Lower']
        df['err_upper']=df['CI_Upper']-df['ORs']

        