import numpy as np 
import pandas as pd
import seaborn as sns 
from scipy import stats


# continuous -> para or non-para 




class VariableTester: 
    """
    連続変数とカテゴリ変数の統計検定を行うクラス。 
    """
    def __init__(self, data: pd.DataFrame):
        """
        検定対象のデータフレームを初期化時に受け取る。
        :param data: pandas.DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("データはPandasのDataFrameである必要があります。")
        self.data = data
        print("VariableTesterがデータフレームで初期化されました。")

    
    