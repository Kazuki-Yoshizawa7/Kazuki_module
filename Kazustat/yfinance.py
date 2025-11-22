import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from typing import Union, List, Tuple
from tqdm import tqdm
import seaborn as sns 
import warnings 
import japanize_matplotlib
from sklearn.preprocessing import MinMaxScaler

class YFinanceAnalysis:

 
    def __init__(self,codes,start,end):
        
        self.codes = codes
        self.start = start
        self.end = end 


    
    def _analyze_table(self):

        l = []
        for code in self.codes:
            dat = yf.Ticker(code)
            info = dat.info
            past_earnings = dat.earnings_dates
            per = info['trailingPE']
            per_future = info['forwardPE']
            pbr = info['priceToBook']
            eps = info['trailingEps']
            divid = info['dividendYield']
            margin = info['operatingMargins']
            promargin = info['profitMargins']
            marketcap = info['marketCap']

            l.append({'Code':code,'PER':per,'Future_PER':per_future,
                    'PBR':pbr,'EPS':eps,'Dividend_Yield':divid,'Operating_Margin':margin
                    ,'Profit_Margin':promargin,'Market_cap':marketcap})
    
        return pd.DataFrame(l)
    
    def _plot_ticks(self,code,start,end):
        df = yf.download(code, start, end)
        dat = yf.Ticker(code)
    
        df = df.reset_index().copy()
        df.columns = ['Date', "Close", "High", "Low", "Open", "Volume"]

        # (A) 過去の決算日を取得
        past_earnings = dat.earnings_dates
        if 'Event Type' in past_earnings.columns:
            earnings_only_df = past_earnings[past_earnings['Event Type'] == 'Earnings']
            earnings_dates_list = earnings_only_df.index
        else:
            earnings_dates_list = past_earnings.index

        past_dividends = dat.dividends
        dividend_dates_list = past_dividends.index
        dividend_dates_list = dividend_dates_list[35:]
        
        
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        # プロット
        plt.figure(figsize=(20,10))
        plt.plot(df['Date'],df["Close"], label="Close Price")
        plt.plot(df['Date'],df["SMA_20"], label="20-day SMA", linestyle='--')
        plt.title(f"{code} Stock Price with 20-day SMA")
        for earnings_date in earnings_dates_list:
            
            # タイムゾーン情報を除去
            earnings_date_naive = earnings_date.tz_localize(None)
            
            plt.axvline(x=earnings_date_naive, color='red', linestyle=':', 
                            label=f"Earnings ({earnings_date_naive.strftime('%Y-%m-%d')})")
            
        for div_date in dividend_dates_list:
            
            # タイムゾーン情報を除去
            div_date_naive = div_date.tz_localize(None)

            plt.axvline(x=div_date_naive, color='green', linestyle=':', 
                            label=f"Ex-Dividend ({div_date_naive.strftime('%Y-%m-%d')})")
                

        plt.xlabel("Date")
        plt.xlim()
        plt.ylabel("Price (JPY)")
        plt.legend()
        plt.grid(True)
        plt.show()


    def describe_codes(self):
        table = self._analyze_table()
        for code in self.codes:
            print(f"Plotting for {code}")
            self._plot_ticks(code,self.start,self.end)
        
        per_ref_line = 15.0      # PERの基準線
        future_per_ref_line = 15.0 # Future PERの基準線
        pbr_ref_line = 1.0       # PBRの基準線 (例: 1倍)
        eps_ref_line = 100.0     # EPSの基準線 (例: 100円)





        # --- 2. 4行2列のサブプロットを作成 ---
        # (レイアウト (4, 2) は正しいです)
        fig, ax = plt.subplots(4, 2, figsize=(15, 22)) # figsizeを (15, 22) に調整
        fig.suptitle('Stock Indicators Comparison', fontsize=18, y=1.02)

        # --- 1行目 (基準線あり) ---
        # プロット 1: PER (0,0)
        sns.barplot(x='Code', y='PER', data=table, ax=ax[0,0], palette='viridis')
        ax[0,0].set_title('PER (実績)')
        ax[0,0].set_ylabel('PER (倍)')
        ax[0,0].axhline(y=per_ref_line, color='red', linestyle='--', label=f'基準: {per_ref_line}')
        ax[0,0].legend() 

        # プロット 2: Future PER (0,1)
        sns.barplot(x='Code', y='Future_PER', data=table, ax=ax[0,1], palette='plasma')
        ax[0,1].set_title('Future PER (予想)')
        ax[0,1].set_ylabel('PER (倍)')
        ax[0,1].axhline(y=future_per_ref_line, color='red', linestyle='--', label=f'基準: {future_per_ref_line}')
        ax[0,1].legend()

        # --- 2行目 (基準線あり) ---
        # プロット 3: PBR (1,0)
        sns.barplot(x='Code', y='PBR', data=table, ax=ax[1,0], palette='coolwarm')
        ax[1,0].set_title('PBR (実績)')
        ax[1,0].set_ylabel('PBR (倍)')
        ax[1,0].axhline(y=pbr_ref_line, color='red', linestyle='--', label=f'基準: {pbr_ref_line}')
        ax[1,0].legend()

        # プロット 4: EPS (1,1)
        sns.barplot(x='Code', y='EPS', data=table, ax=ax[1,1], palette='Set2')
        ax[1,1].set_title('EPS (実績)')
        ax[1,1].set_ylabel('EPS (円)')
        ax[1,1].axhline(y=eps_ref_line, color='red', linestyle='--', label=f'基準: {eps_ref_line}')
        ax[1,1].legend()

        # --- 3行目 (基準線なし) ---
        # ▼ (修正) y= に正しい列名 'Dividend_Yield (%)' を指定
        sns.barplot(x='Code', y='Dividend_Yield', data=table, ax=ax[2,0], palette='YlGn')
        ax[2,0].set_title('配当利回り')
        ax[2,0].set_ylabel('利回り (%)')

        # ▼ (修正) y= に正しい列名 'Operating_Margin (%)' を指定
        sns.barplot(x='Code', y='Operating_Margin', data=table, ax=ax[2,1], palette='RdYlBu')
        ax[2,1].set_title('営業利益率')
        ax[2,1].set_ylabel('利益率 (%)')

        # --- 4行目 (基準線なし) ---
        # ▼ (修正) y= に正しい列名 'Profit_Margin (%)' を指定
        sns.barplot(x='Code', y='Profit_Margin', data=table, ax=ax[3,0], palette='RdYlBu_r')
        ax[3,0].set_title('純利益率')
        ax[3,0].set_ylabel('利益率 (%)')

        # ▼ (修正) 構文エラー (ax[3,]) を (ax[3,1]) に修正
        # ▼ (修正) y= に正しい列名 'Market_cap (億円)' を指定
        sns.barplot(x='Code', y='Market_cap', data=table, ax=ax[3,1], palette='Oranges')
        ax[3,1].set_title('時価総額')
        ax[3,1].set_ylabel('時価総額 (億円)')


        # --- 5. レイアウトの調整 ---
        for subplot in ax.flat:
            # グリッドを追加
            subplot.grid(axis='y', linestyle='--', alpha=0.7)
            # X軸のラベルを回転
            subplot.set_xticklabels(subplot.get_xticklabels(), rotation=60, ha='right')
            subplot.set_xlabel('') # 各プロットのX軸ラベルは削除

        # ▼ (修正) 一番下のプロット (4行目) にだけX軸ラベルを表示
        ax[3,0].set_xlabel('Stock Code')
        ax[3,1].set_xlabel('Stock Code')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    
    def scatter_show(self):
        table = self._analyze_table()
        min = table['EPS'].min()
        max = table['EPS'].max()
        scaler = MinMaxScaler(feature_range=(min, max)) # range chousei

        table['size']=scaler.fit_transform(table[['EPS']])
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=table, 
                        x='PBR', 
                        y='PER', 
                        hue='Code',
                        size='size',
                        sizes = (min,max),
                        palette='viridis',     # カラーマップ
                        alpha=0.7,             # 透明度
                        edgecolor='w',         # 点の縁の色
                        linewidth=0.5)
        plt.title('PBR vs PER Scatter Plot (Size=EPS, Color=Code)', fontsize=16)
        plt.xlabel('PBR', fontsize=12)
        plt.ylabel('PER', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # サイズの凡例と色の凡例が自動で表示されます
        plt.legend(title='EPS Range / Category', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # 凡例がはみ出さないように調整
        plt.show()


        