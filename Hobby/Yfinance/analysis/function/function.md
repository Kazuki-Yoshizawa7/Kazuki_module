# Function 仕様書

function/function.pyで実現したいものである。これをanalysis/00_notebook.ipynb上で東京証券所のファナック、安川電機、及びトヨタとスバルについてTestしたい。

## 可視化したいもの：

yFinance moduleを用いて行う。

- PBR 
- PER

- 財務指標：
    - ROE
    - ROA 
    - 売上高
    - 純資産回転率（ATO）
    - 財務レバレッジ ICR
    - Profit margin 
    - return of sales (ROS)
    - 固定資産回転率
    - 運転資本の計算
    - CCC(Cash conversion cycle) (棚卸資産や売上債権と仕入債権もそれぞれ表示)
    - 流動比率
    - 当座比率
    - 固定比率・固定長期適合率
    - 自己資本比率
    - 成長率(CAGR)

現段階で、これらの指標をticker_list で入力したものについて比較してPlotできるようにしたい。
実現Flow:
- init 定義：ticker_list = (ticker,sector(業界))
- obtaining / calculating: 上にあげた指標をyFinance経由で取得・計算してDataFrameに格納して返す
- merging dataframe: Obtain/calculating で得られるDataFrameについてTickerごとに取得したものを、業界ラベルをつけて統合する；
- Plotting:　業界で分けるというTrue /False入力があった場合、業界ごとにPlotしていく。比較できるように各指で一つのグラフとしてプロットしていく。
