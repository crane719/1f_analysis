# 1f_analysis

- データが1/fゆらぎに従っているのか分析

## usage

- 必要なものをinstall

```
sh install.sh
```

- static/datasetの作成
    - static/dataset directory内で分析したいデータを以下のように作成
        - ``static/dataset/分類名/曲名.png[or wav]``
        - ``static/dataset/分類名/人名/曲名.png[or wav]``

- 実行

```
python main.py
```
