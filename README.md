# covid19-detection

CTスキャンの画像から、コロナウィルスに罹患しているのかどうかを検出するディープラーニングモデルです。
精度は　ROC AUC で 96%程度です。

## 学習方法

どこかに、以下のリポジトリを cloneします
```
git clone https://github.com/ieee8023/covid-chestxray-dataset
```

src/config.tomlの DATADIR_HEREの部分に、先ほどクローンしたリポジトリへのパスを書きます。
```
[data]
base_dir = "DATADIR_HERE"
```

train.pyを実行します。

```
python train.py
```


## Dataset

COVID-19 image data collection

Joseph Paul Cohen and Paul Morrison and Lan Dao
COVID-19 image data collection, arXiv:2003.11597, 2020
https://github.com/ieee8023/covid-chestxray-dataset


## 精度

```
[I 200424 21:16:31 train:92] Epoch 200 Train loss 0.56104, Acc 74.3%, F1 66.3%, AUC 75.95%, grad 0.9589924216270447
[I 200424 21:16:33 train:112] Epoch 200 Test loss 0.25031, Acc 86.0%, F1 74.4%, AUC 96.84%
```
