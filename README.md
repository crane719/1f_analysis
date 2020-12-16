# 1f_analysis

- データが1/fゆらぎに従っているのか分析

## usage

- 必要なものをinstall
    - ``install.sh``はbrew入っている前提なため注意

```
sh install.sh
```

- static/datasetの作成
    - static/dataset directory内で分析したいデータを以下のように作成
        - ``static/dataset/分類名/曲名.mp3[or wav]``

- 実行

```
source venv/bin/activate
python main.py
```

- 可視化
    - ``http://localhost:5000``にアクセス

```
source venv/bin/activate
python run.py
```
