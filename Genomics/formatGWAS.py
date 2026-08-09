import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


"""
Class for reshaping GWAS summary statstics to run command tools such as TwoSampleMR, ANNOVAR, etc
"""

class GWAS_Prep():

    def __init__(self,input_file):
        self.input_file = input_file
        

    def _standarized_snp(self,df):

        df['standarized_SNP'] = 'chr' + df['CHR'].astype(str) + ':' + df['BP'].astype(str)

        return df 
    

    # Have to speficy columns: 
    def rename_cols(self, SNP: str=None, CHR: str, BP: str,A1: str, A2: str, BETA: str, SE: str, FREQ: str, NEG_LOG10_P: str) -> pd.DataFrame:


    #     df_std = df_std.rename(columns={
    #     'chromosome': 'CHR',
    #     'base_pair_location': 'BP',
    #     'effect_allele': 'A1',
    #     'other_allele': 'A2',
    #     'beta': 'BETA',
    #     'standard_error': 'SE',
    #     'effect_allele_frequency': 'FREQ',
    #     'neg_log_10_p_value': 'NEG_LOG10_P'
    # })

        df = pd.read_csv(self.input_file, sep=None, engine='python')
        col_mapping = {
            SNP: "SNP",
            CHR: "CHR",
            BP: "BP",
            A1: "A1",
            A2: "A2",
            BETA: "BETA",
            SE: "SE",
            FREQ: "FREQ",
            NEG_LOG10_P: "NEG_LOG10_P",

            }
        
        existing_mapping = {
            old: new 
            for old, new in col_mapping.items()
            if old in df.columns
        }

        missing = set(col_mapping.keys()) - set(existing_mapping.keys())
        if missing:
            print(f"[WARNING] :Missing Columns: {missing}")

        df = df.rename(columns=existing_mapping)

        # NEG_LOG10_P → P値に変換
        if "NEG_LOG10_P" in df.columns:
            df["P"] = 10 ** (-df["NEG_LOG10_P"])


        # adding standarized_SNP col
        df = self._standarized_snp(df)

        return df 
    
    
    
        

"""
Class for harmonizing several GWAS Summary Statistics: 
"""

class Harmonize_GWAS():

    def __init__(self,df1,df2):
            
        self.df1 = df1 
        self.df2 = df2


    def harmonize(self):

        df1_work = self.df1.copy()
        df2_work = self.df2.copy()
        # Setting merge_key to merge several gwas statistics 
        df1_work['merge_key'] = df1_work['CHR'].astype(str) + ':' + df1_work['BP'].astype(str)
        df2_work['merge_key'] = df2_work['CHR'].astype(str) + ':' + df2_work['BP'].astype(str)



        common_keys = set(df1_work['merge_key']) & set(df2_work['merge_key'])
        print(f"Common SNPs between studies: {len(common_keys)}")

        

        

