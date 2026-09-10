# 00 Notebook: 



### Content: 


- July 26th 2026 


    *assert関数の使い方*

    以下のような例をあげる：
    x = 5
    ```python
    # x は 0 以上 10 以下であるはずだ、という確認
    assert x >= 0 and x <= 10, "xの値がおかしいです"
    # continues if True 
    # stops and sends error message when False (Error)

    ```

- Sep 10th 2026

  Plotting の裏技：
  どのようにしてレイアウトを作るか

  ```python
     def auto_subplots(n, n_cols=3, figsize_per=(4, 3)):
       n_rows = math.ceil(n / n_cols)
       fig, axes = plt.subplots(
           n_rows, n_cols,
           figsize=(figsize_per[0]*n_cols, figsize_per[1]*n_rows),
           squeeze=False
       )
       axes = axes.flatten()
       for ax in axes[n:]:
           ax.set_visible(False)
       return fig, axes[:n]
   
   fig, axes = auto_subplots(7)
   for i, ax in enumerate(axes):
       ax.plot(data[i])
       ax.set_title(f"Plot {i}")
   plt.tight_layout()
  ```


  - math.ceil()
    引数として与えた数値の小数点以下を切り上げて、その数値以上の最小の整数を返す関数として使える

  - 書いた図の一部分を拡大させたい時：

    ```python
      from mpl_toolkits.axes_grid1.inset_locator import inset_axes
      from mpl_toolkits.axes_grid1.inset_locator import mark_inset
      import matplotlib.pyplot as plt
      import numpy as np
      
      x = np.linspace(0, 10, 500)
      y = np.sin(x) * np.exp(-x/5)
      
      fig, ax = plt.subplots()
      ax.plot(x, y)
      
      # 拡大用の小さいaxesを作成（親axの30%サイズ、右上に配置）
      axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
      axins.plot(x, y)
      axins.set_xlim(1, 2)   # 拡大したい範囲を指定
      axins.set_ylim(0.3, 0.6)
      
      
      mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray")
      axins.set_title("拡大部分", fontsize=8)
      axins.annotate("ピーク", xy=(1.5, 0.55), xytext=(1.7, 0.5),
                     arrowprops=dict(arrowstyle="->"))
      
      axins.set_xticklabels([])
      axins.set_yticklabels([])
      # または
      axins.tick_params(labelsize=6)  # 文字サイズだけ小さくする
  ```

  


