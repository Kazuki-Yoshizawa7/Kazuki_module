
# analysis functions:
#
# yFinance を用いて、指定した複数銘柄(ticker)の株価指標・財務指標を取得し、
# 業界(sector)ラベルを付与して統合、比較用に可視化するクラス。
#
# 実現Flow:
#   1. __init__        : ticker_list = [(ticker, sector), ...] を受け取る
#   2. obtain_all       : 各tickerについてyFinance経由で指標を取得・計算しDataFrameに格納
#   3. merge            : 取得済みDataFrameに業界ラベルを付けて1つのDataFrameに統合
#   4. plot             : by_sector=True/False に応じて指標ごとに比較プロットする

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import japanize_matplotlib

# 日本語(業界名など)がグラフ中で文字化けしないよう、利用可能な日本語フォントを優先的に設定する。
# 該当フォントが環境になければ何もせず、matplotlibの既定フォントにフォールバックする。
# plt.rcParams["font.family"] = [
#     "Hiragino Sans", "Hiragino Kaku Gothic Pro",  # macOS
#     "Yu Gothic", "Meiryo",                         # Windows
#     "IPAexGothic", "TakaoPGothic", "Noto Sans CJK JP",  # Linux
#     "sans-serif",
# ]
plt.rcParams["axes.unicode_minus"] = False


class YFinance_analysis():
    """
    yFinance を利用して複数銘柄の株価指標・財務指標を取得し、比較・可視化するクラス。

    Parameters
    ----------
    ticker_list : list[tuple[str, str]]
        (ticker, sector) のタプルのリスト。
        例:
            [
                ("6954.T", "機械(ロボット)"),
                ("6506.T", "機械(ロボット)"),
                ("7203.T", "自動車"),
                ("7270.T", "自動車"),
            ]
    """

    # 実際のfinancial statementの行名はyFinanceのバージョンや銘柄によって
    # 表記ゆれがあるため、候補を複数持たせて安全に取得する。
    _ITEM_ALIASES = {
        "revenue": ["Total Revenue", "TotalRevenue"],
        "cogs": ["Cost Of Revenue", "CostOfRevenue", "Reconciled Cost Of Revenue"],
        "operating_income": ["Operating Income", "OperatingIncome"],
        "net_income": ["Net Income", "NetIncome", "Net Income Common Stockholders"],
        "ebit": ["EBIT", "Ebit"],
        "interest_expense": ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"],
        "total_assets": ["Total Assets", "TotalAssets"],
        "total_equity": [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "StockholdersEquity",
            "Common Stock Equity",
        ],
        "current_assets": ["Current Assets", "CurrentAssets", "Total Current Assets"],
        "current_liabilities": ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"],
        "inventory": ["Inventory", "Inventories"],
        "receivables": ["Accounts Receivable", "Receivables", "Net Receivables", "Gross Accounts Receivable"],
        "payables": ["Accounts Payable", "Payables", "Payables And Accrued Expenses", "Accounts Payable And Accrued Expense"],
        # 固定資産 = 非流動資産合計。"Net PPE"は有形固定資産のみで固定資産全体
        # (投資その他の資産・無形固定資産等を含む)を大幅に過小評価するため、
        # 優先度を下げてフォールバック専用とする。"Net Tangible Assets"は
        # 有形純資産(総資産-無形資産-総負債)であり固定資産とは別概念のため対象外。
        "fixed_assets": ["Total Non Current Assets", "Net PPE"],
        "long_term_liabilities": [
            "Long Term Debt",
            "Total Non Current Liabilities Net Minority Interest",
            "Long Term Debt And Capital Lease Obligation",
        ],
        "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    }

    def __init__(self, ticker_list):
        self.ticker_list = ticker_list
        self.tickers = {}        # ticker -> yf.Ticker object
        self.metrics_df = {}     # ticker -> 年度別指標DataFrame
        self.merged_df = None    # 業界ラベル付き統合DataFrame

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(df, keys, col):
        """財務諸表DataFrameから該当する行を安全に取得する(見つからなければNaN)。"""
        if df is None or df.empty:
            return np.nan
        for key in keys:
            if key in df.index:
                try:
                    val = df.loc[key, col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    return val
                except Exception:
                    continue
        return np.nan

    @staticmethod
    def _safe_div(numerator, denominator):
        if numerator is None or denominator is None:
            return np.nan
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    # ------------------------------------------------------------------
    # 1. Obtaining / Calculating
    # ------------------------------------------------------------------
    def obtain_and_calculate(self, ticker):
        """
        1銘柄について財務諸表(年次)を取得し、各種指標を年度別に計算してDataFrameとして返す。
        """
        tk = yf.Ticker(ticker)

        bs = tk.balance_sheet   # 貸借対照表(年次、列=決算期、新しい年ほど左)
        fs = tk.financials      # 損益計算書(年次)

        if bs is None or fs is None or bs.empty or fs.empty:
            raise ValueError(f"{ticker}: 財務データが取得できませんでした。")

        # 貸借対照表・損益計算書に共通する決算期のみを使用し、古い順に並べる。
        columns = [c for c in bs.columns if c in fs.columns]
        columns = sorted(columns)

        # 各決算期末の終値を取得し、その期のEPS・BPSからPER・PBRを近似計算する
        # (yFinanceの無料枠では過去時点のPER/PBRそのものは取得できないため)。
        price_hist = tk.history(
            start=columns[0] - pd.Timedelta(days=10),
            end=columns[-1] + pd.Timedelta(days=10),
        )["Close"]
        if not price_hist.empty:
            price_hist.index = price_hist.index.tz_localize(None)

        A = self._ITEM_ALIASES
        records = []

        for col in columns:
            revenue = self._safe_get(fs, A["revenue"], col)
            cogs = self._safe_get(fs, A["cogs"], col)
            operating_income = self._safe_get(fs, A["operating_income"], col)
            net_income = self._safe_get(fs, A["net_income"], col)
            ebit = self._safe_get(fs, A["ebit"], col)
            interest_expense = self._safe_get(fs, A["interest_expense"], col)

            total_assets = self._safe_get(bs, A["total_assets"], col)
            total_equity = self._safe_get(bs, A["total_equity"], col)
            current_assets = self._safe_get(bs, A["current_assets"], col)
            current_liabilities = self._safe_get(bs, A["current_liabilities"], col)
            inventory = self._safe_get(bs, A["inventory"], col)
            receivables = self._safe_get(bs, A["receivables"], col)
            payables = self._safe_get(bs, A["payables"], col)
            fixed_assets = self._safe_get(bs, A["fixed_assets"], col)
            long_term_liabilities = self._safe_get(bs, A["long_term_liabilities"], col)
            shares_outstanding = self._safe_get(bs, A["shares_outstanding"], col)

            if pd.isna(ebit):
                # EBITが取得できない場合は営業利益で代用
                ebit = operating_income

            # ---- 各種指標計算 ----
            roe = self._safe_div(net_income, total_equity)
            roa = self._safe_div(net_income, total_assets)
            ato = self._safe_div(revenue, total_assets)                          # 総資産回転率
            financial_leverage = self._safe_div(total_assets, total_equity)      # 財務レバレッジ
            icr = self._safe_div(ebit, abs(interest_expense) if pd.notna(interest_expense) else np.nan)  # インタレスト・カバレッジ・レシオ
            profit_margin = self._safe_div(net_income, revenue)
            ros = self._safe_div(operating_income, revenue)                      # 売上高営業利益率(ROS)
            fixed_asset_turnover = self._safe_div(revenue, fixed_assets)
            working_capital = (
                current_assets - current_liabilities
                if pd.notna(current_assets) and pd.notna(current_liabilities)
                else np.nan
            )

            dio = self._safe_div(inventory, cogs)
            dio = dio * 365 if pd.notna(dio) else np.nan
            dso = self._safe_div(receivables, revenue)
            dso = dso * 365 if pd.notna(dso) else np.nan
            dpo = self._safe_div(payables, cogs)
            dpo = dpo * 365 if pd.notna(dpo) else np.nan
            ccc = (dio if pd.notna(dio) else 0) + (dso if pd.notna(dso) else 0) - (dpo if pd.notna(dpo) else 0)

            current_ratio = self._safe_div(current_assets, current_liabilities)
            quick_ratio = (
                self._safe_div(current_assets - inventory, current_liabilities)
                if pd.notna(current_assets) and pd.notna(inventory)
                else np.nan
            )
            fixed_ratio = self._safe_div(fixed_assets, total_equity)
            fixed_long_term_ratio = self._safe_div(
                fixed_assets,
                (total_equity + long_term_liabilities)
                if pd.notna(total_equity) and pd.notna(long_term_liabilities)
                else np.nan,
            )
            equity_ratio = self._safe_div(total_equity, total_assets)            # 自己資本比率

            # PER, PBR: 決算期末に最も近い終値と、その期のEPS・BPSから算出
            price = price_hist.asof(col) if not price_hist.empty else np.nan
            eps = self._safe_div(net_income, shares_outstanding)
            bps = self._safe_div(total_equity, shares_outstanding)
            per = self._safe_div(price, eps)
            pbr = self._safe_div(price, bps)

            records.append({
                "date": col,
                "revenue": revenue,
                "roe": roe,
                "roa": roa,
                "ato": ato,
                "financial_leverage": financial_leverage,
                "icr": icr,
                "profit_margin": profit_margin,
                "ros": ros,
                "fixed_asset_turnover": fixed_asset_turnover,
                "working_capital": working_capital,
                "inventory": inventory,
                "receivables": receivables,
                "payables": payables,
                "dio": dio,
                "dso": dso,
                "dpo": dpo,
                "ccc": ccc,
                "current_ratio": current_ratio,
                "quick_ratio": quick_ratio,
                "fixed_ratio": fixed_ratio,
                "fixed_long_term_ratio": fixed_long_term_ratio,
                "equity_ratio": equity_ratio,
                "per": per,
                "pbr": pbr,
            })

        df = pd.DataFrame(records).set_index("date").sort_index()

        # CAGR: 取得できた期間全体での売上高成長率
        if len(df) >= 2 and pd.notna(df["revenue"].iloc[0]) and df["revenue"].iloc[0] != 0:
            years = len(df) - 1
            ratio = df["revenue"].iloc[-1] / df["revenue"].iloc[0]
            df["cagr"] = ratio ** (1 / years) - 1 if ratio > 0 else np.nan
        else:
            df["cagr"] = np.nan

        df["ticker"] = ticker

        self.tickers[ticker] = tk
        self.metrics_df[ticker] = df
        return df

    def obtain_all(self):
        """ticker_list に含まれる全銘柄について指標を取得・計算する。"""
        for ticker, sector in self.ticker_list:
            try:
                self.obtain_and_calculate(ticker)
            except Exception as e:
                print(f"[WARN] {ticker} の取得に失敗しました: {e}")
        return self.metrics_df

    # ------------------------------------------------------------------
    # 2. Merging
    # ------------------------------------------------------------------
    def merge(self):
        """
        取得済みの各銘柄DataFrameに業界(sector)ラベルを付与して1つのDataFrameに統合する。
        """
        if not self.metrics_df:
            self.obtain_all()

        sector_map = dict(self.ticker_list)
        merged = []
        for ticker, df in self.metrics_df.items():
            tmp = df.copy()
            tmp["sector"] = sector_map.get(ticker, "unknown")
            tmp = tmp.reset_index()
            merged.append(tmp)

        if not merged:
            raise ValueError("統合できるデータがありません。obtain_all() の結果を確認してください。")

        self.merged_df = pd.concat(merged, ignore_index=True)
        return self.merged_df

    # ------------------------------------------------------------------
    # 3. Plotting
    # ------------------------------------------------------------------
    DEFAULT_METRICS = [
        "per", "pbr", "roe", "roa", "revenue", "ato",
        "financial_leverage", "icr", "profit_margin", "ros",
        "fixed_asset_turnover", "working_capital", "ccc",
        "current_ratio", "quick_ratio", "fixed_ratio",
        "fixed_long_term_ratio", "equity_ratio", "cagr",
    ]

    def plot(self, by_sector=False, metrics=None, ncols=3, figsize=(5, 4)):
        """
        指標ごとにサブプロットを作成し、銘柄間で比較できるように1枚の図にまとめてプロットする。
        `fig, axes = plt.subplots(...)` でグリッドを作り、各指標を1つのaxに描画する。

        Parameters
        ----------
        by_sector : bool
            True の場合、行=指標・列=業界(sector) のグリッドで、業界ごとに分けて表示する。
            False の場合、指標ごとに1つのaxを割り当て、全銘柄を重ねて表示する
            (グリッドは ncols 列で折り返す)。
        metrics : list[str] or None
            プロットしたい指標名のリスト。Noneの場合は主要指標をすべてプロットする。
        ncols : int
            by_sector=False のときの1行あたりの列数(サブプロット数)。
        figsize : tuple
            サブプロット1つあたりの(幅, 高さ)。グリッド全体のサイズはこれに行数・列数を掛けて決まる。

        Returns
        -------
        fig, axes : matplotlib の Figure と Axes(ndarray)
        """
        if self.merged_df is None:
            self.merge()

        df = self.merged_df
        metrics = [m for m in (metrics or self.DEFAULT_METRICS) if m in df.columns]
        if not metrics:
            raise ValueError("プロット可能な指標がありません。metrics引数を確認してください。")

        if by_sector:
            sectors = list(df["sector"].unique())
            nrows, ncols_ = len(metrics), len(sectors)
            fig, axes = plt.subplots(
                nrows, ncols_,
                figsize=(figsize[0] * ncols_, figsize[1] * nrows),
                squeeze=False,
            )
            for i, metric in enumerate(metrics):
                for j, sector in enumerate(sectors):
                    ax = axes[i][j]
                    sub = df[df["sector"] == sector]
                    for ticker in sub["ticker"].unique():
                        t_df = sub[sub["ticker"] == ticker]
                        ax.plot(t_df["date"], t_df[metric], marker="o", label=ticker)
                    ax.set_title(f"{metric} - {sector}")
                    ax.legend(fontsize=8)
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(True)
        else:
            n = len(metrics)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(figsize[0] * ncols, figsize[1] * nrows),
                squeeze=False,
            )
            flat_axes = axes.flatten()
            for ax, metric in zip(flat_axes, metrics):
                for ticker in df["ticker"].unique():
                    t_df = df[df["ticker"] == ticker]
                    ax.plot(t_df["date"], t_df[metric], marker="o", label=ticker)
                ax.set_title(metric)
                ax.legend(fontsize=8)
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True)
            # 使わなかった余りのaxは非表示にする
            for ax in flat_axes[n:]:
                ax.axis("off")

        fig.tight_layout()
        plt.show()
        return fig, axes
