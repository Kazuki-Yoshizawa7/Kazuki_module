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




