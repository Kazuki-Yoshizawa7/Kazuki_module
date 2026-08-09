import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import japanize_matplotlib



class StockValuation:
    def __init__(self, ticker_list, main_ticker, csv_data=None):
        self.ticker_list = ticker_list
        self.main_ticker = main_ticker
        self.csv_data = csv_data

    def extract_info(self, ticker_symbol: str) -> dict:
        """単一のティッカーから財務データを取得・算出する"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')

            _ITEMS = {
                'ticker': ticker_symbol,
                'current_price': price,
                'revenue': info.get('totalRevenue'),
                'historical_per': info.get('trailingPE'),
                'forward_per': info.get('forwardPE'),
                'pbr': info.get('priceToBook'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'ev': info.get('enterpriseValue'),
                'ebitda': info.get('ebitda'),
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'bps': info.get('bookValue')
            }

            # 欠損値（None）がある場合の自己計算処理（フォールバック）
            if _ITEMS['historical_per'] is None and price and _ITEMS['trailing_eps']:
                _ITEMS['historical_per'] = price / _ITEMS['trailing_eps']

            if _ITEMS['forward_per'] is None and price and _ITEMS['forward_eps']:
                _ITEMS['forward_per'] = price / _ITEMS['forward_eps']

            if _ITEMS['pbr'] is None and price and _ITEMS['bps']:
                _ITEMS['pbr'] = price / _ITEMS['bps']

            if _ITEMS['ev_ebitda'] is None and _ITEMS['ev'] and _ITEMS['ebitda']:
                _ITEMS['ev_ebitda'] = _ITEMS['ev'] / _ITEMS['ebitda']

            return _ITEMS

        except Exception as e:
            print(f"[{ticker_symbol}] 取得エラー: {e}")
            return {'ticker': ticker_symbol}

    def get_dataframe(self) -> pd.DataFrame:
        """ticker_list 全体のデータをループ取得し、DataFrame化する"""
        records = []

        for item in self.ticker_list:
            # [(ticker, sector), ...] のタプル形式と ['ticker', ...] の単一文字列リストの両方に対応
            if isinstance(item, (tuple, list)):
                ticker_symbol = item[0]
                sector = item[1] if len(item) > 1 else None
            else:
                ticker_symbol = item
                sector = None

            # データ取得
            info_dict = self.extract_info(ticker_symbol)

            # セクター情報の追加とメイン銘柄フラグの設定
            if sector:
                info_dict['sector'] = sector
            info_dict['is_main'] = (ticker_symbol == self.main_ticker)

            records.append(info_dict)

        df = pd.DataFrame(records)
        return df

    def calculate_summary(self, by_sector: bool = False) -> pd.DataFrame:
        """
        セクターごと、または類似企業全体の平均（Mean）および中央値（Median）を算出する
        (列構造をマルチインデックスに統一)
        """
        dataframe = self.get_dataframe()
        num_cols = ['current_price', 'historical_per', 'forward_per', 'pbr', 'ev_ebitda']

        if by_sector:
            # 1. セクターごとにグループ化
            summary_df = dataframe.groupby('sector')[num_cols].agg(['mean', 'median'])
        else:
            # 2. 評価対象(main_ticker)を除外した「類似企業全体」
            peer_df = dataframe[~dataframe['is_main']]
            s = peer_df[num_cols].agg(['mean', 'median'])
            
            # 列構造を (指標, 統計量) のマルチインデックスに統一して1行のDataFrameを作成
            summary_df = pd.DataFrame([s.unstack()], index=['All Peers'])

        return summary_df

    def estimation(self, use_sector_summary: bool = False) -> pd.DataFrame:
        """
        類似企業の倍率（Mean / Median）と main_ticker の財務データを用いて推定時価総額を算出する
        """
        # 1. 類似企業の倍率（Mean / Median）を取得
        summary = self.calculate_summary(by_sector=use_sector_summary)

        # 2. 分析対象(main_ticker)の財務データを yfinance から取得
        main_obj = yf.Ticker(self.main_ticker)
        info = main_obj.info

        # --- 財務数値の取得 ---
        shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')

        # ① 自己資本（純資産）
        bps = info.get('bookValue')
        total_equity = info.get('totalStockholderEquity')
        equity = total_equity if total_equity is not None else (bps * shares if bps and shares else None)

        # ② 当期純利益（実績・予想）
        trailing_eps = info.get('trailingEps')
        forward_eps = info.get('forwardEps')
        net_income_hist = (trailing_eps * shares) if (trailing_eps and shares) else info.get('netIncomeToCommon')
        net_income_fwd = (forward_eps * shares) if (forward_eps and shares) else None

        # ③ EBITDA
        ebitda = info.get('ebitda')

        # ④ 調整項目（非事業価値・債権者価値など）
        non_operating_assets = info.get('totalCash') or 0
        creditor_value = info.get('totalDebt') or 0
        other_net_assets = 0

        # 3. 倍率参照用の行（Series）を取り出す
        if use_sector_summary:
            main_sector = None
            for item in self.ticker_list:
                if isinstance(item, (tuple, list)) and item[0] == self.main_ticker:
                    main_sector = item[1] if len(item) > 1 else None
                    break
            
            if main_sector and main_sector in summary.index:
                multiples = summary.loc[main_sector]
            else:
                multiples = summary.iloc[0]
        else:
            multiples = summary.loc['All Peers'] if 'All Peers' in summary.index else summary.iloc[0]

        # 4. 各手法による時価総額の試算
        results = {}
        for stat in ['mean', 'median']:
            pbr_mult = multiples.get(('pbr', stat))
            per_hist_mult = multiples.get(('historical_per', stat))
            per_fwd_mult = multiples.get(('forward_per', stat))
            ev_ebitda_mult = multiples.get(('ev_ebitda', stat))

            # PBR法: PBR * 自己資本
            mcap_pbr = (pbr_mult * equity) if (pd.notna(pbr_mult) and equity) else np.nan

            # PER法 (実績): PER * 当期純利益 (実績)
            mcap_per_hist = (per_hist_mult * net_income_hist) if (pd.notna(per_hist_mult) and net_income_hist) else np.nan

            # PER法 (予想): PER * 当期純利益 (予想)
            mcap_per_fwd = (per_fwd_mult * net_income_fwd) if (pd.notna(per_fwd_mult) and net_income_fwd) else np.nan

            # EV/EBITDA法
            if pd.notna(ev_ebitda_mult) and ebitda:
                estimated_ev = ev_ebitda_mult * ebitda
                mcap_ev_ebitda = estimated_ev + non_operating_assets - creditor_value - other_net_assets
            else:
                mcap_ev_ebitda = np.nan

            results[stat] = {
                'PBR法': mcap_pbr,
                'PER法(実績)': mcap_per_hist,
                'PER法(予想)': mcap_per_fwd,
                'EV/EBITDA法': mcap_ev_ebitda
            }

        # 転置して億円単位（10^8）で返す
        estimation_df = pd.DataFrame(results)
        return estimation_df / 1e8

    def plot_estimation_range(self, estimation_df: pd.DataFrame = None):
        """
        Mean と Median による推定時価総額のレンジを縦方向の線グラフでプロットする
        """
        if estimation_df is None:
            estimation_df = self.estimation()

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = estimation_df.index
        x_positions = np.arange(len(methods))

        # 各手法ごとに Mean と Median の間を縦線で結ぶ
        for i, method in enumerate(methods):
            val_mean = estimation_df.loc[method, 'mean']
            val_median = estimation_df.loc[method, 'median']

            # 欠損値スキップ
            if pd.isna(val_mean) or pd.isna(val_median):
                continue

            low = min(val_mean, val_median)
            high = max(val_mean, val_median)

            # 1. Mean と Median を結ぶ垂直線 (レンジ)
            ax.vlines(x=i, ymin=low, ymax=high, color='gray', linestyle='-', linewidth=3, alpha=0.6, zorder=2)

            # 2. Mean (青) と Median (赤) のデータポイント
            ax.scatter(i, val_mean, color='blue', s=120, label='Mean（平均値）' if i == 0 else "", zorder=3)
            ax.scatter(i, val_median, color='red', s=120, label='Median（中央値）' if i == 0 else "", zorder=3)

            # 3. 数値のテキスト注記 (数値が見やすいよう左右に配置)
            ax.annotate(f"Mean: {val_mean:,.0f}億円", (i, val_mean), textcoords="offset points", 
                        xytext=(12, 0), ha='left', va='center', fontsize=9, color='blue', fontweight='bold')
            ax.annotate(f"Med: {val_median:,.0f}億円", (i, val_median), textcoords="offset points", 
                        xytext=(-12, 0), ha='right', va='center', fontsize=9, color='red', fontweight='bold')

        # 4. 参考ライン：現在の実際の時価総額を横点線で追加
        try:
            main_info = yf.Ticker(self.main_ticker).info
            actual_mcap = main_info.get('marketCap')
            if actual_mcap:
                actual_mcap_oku = actual_mcap / 1e8
                ax.axhline(actual_mcap_oku, color='green', linestyle='--', linewidth=1.5,
                           label=f'現在の実際の時価総額 ({actual_mcap_oku:,.0f}億円)', zorder=1)
        except Exception:
            pass

        # グラフの見た目調整
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
        ax.set_ylabel('推定時価総額（億円）', fontsize=11)
        ax.set_title(f'マルチプル法による推定時価総額レンジ比較 ({self.main_ticker})', fontsize=14, pad=15)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        ax.legend(loc='best', frameon=True)

        # 軸の余白調整
        ax.set_xlim(-0.8, len(methods) - 0.2)
        
        plt.tight_layout()
        plt.show()




# --- 実行例 ---
if __name__ == "__main__":
    # ティッカーとセクターのリスト（タプル形式）
    ticker_list = [
        ("6954.T", "FA/ロボット"),  # ファナック
        ("6506.T", "FA/ロボット"),  # 安川電機
        ("6645.T", "制御機器"),     # オムロン
        ("6861.T", "FA/センサー")   # キーエンス
    ]
    main_ticker = "6954.T"

    # インスタンス化とDataFrame取得
    valuation = StockValuation(ticker_list, main_ticker)
    df_result = valuation.get_dataframe()

    # 主要マルチプルのみを表示
    cols = ['ticker', 'sector', 'is_main', 'current_price', 'historical_per', 'forward_per', 'pbr', 'ev_ebitda']
    print(df_result[cols])