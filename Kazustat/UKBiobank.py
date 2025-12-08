
import pandas as pd 
import numpy as np 


"""
UKBiobank Data Analysis Module
"""

class UKBiobank:

    """
    df とは変数のFieldIDとDescription、RenameするためのColumnNameをカラムにもつCSVファイルのこと（適宜Excelか何かで作っていくこと）
    """

    def __init__(self, df,participant,dataset,engine): 
        if df.empty:
            raise ValueError("入力されたDataFrameが空です。")
        self.df = df
        self.participant = participant
        self.dataset = dataset  
        self.engine = engine

    """
    UKBiobank用のデータを読み込むための、変数リストを作りたい（事前にCSVで作ったものに対してそれをDict型に変形して、読み込むためのコード）”
    """

    


    def variable_loader(self):

        var_list = self.df['FieldID'].astype(str) #{'43000':'age'} こんな感じだっけな List型で返す
        desc = self.df['Rename']
        d = dict(zip(desc, var_list))
        print(d)
        
        return d


    def dataloader(self):

        cov_ids = self.variable_loader()
        field_objs = []

        def get_field_objs(fid):
            return self.participant.find_fields(name_regex=rf"^p{fid}(_i\d+)?(_a\d+)?$")
        
        field_objs += self.participant.find_fields(name_regex="^eid$")
        # 各Covariateの全インスタンス／配列を取得
        for fid in cov_ids.values():
            field_objs.extend(get_field_objs(fid))
        main_entity = self.dataset.primary_entity

        ctv_main = main_entity.retrieve_fields(fields = field_objs,engine=self.engine)

        data_df = ctv_main.toPandas()

        return data_df
    
    

    
    def rename_columns(self,data):

        data_renamed = data.rename(columns=dict(zip('p'+self.df['FieldID'].astype(str))))



    




