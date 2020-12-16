import copy
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import json
import pandas as pd

import utils
import signal_process as sp
import visualize as vis

# result directoryの作成
result_dir="static/analysis_result"
required_dirs = [
        result_dir,
        result_dir+"/psd",
        result_dir+"/fitting",
        ]
utils.make_dir(required_dirs)

# datasetの各データのdirectoryの取得
# ラベルをkey, 各データのdirectoryのlistをvalueとしたdictionaryで保存
dataset_dir="static/dataset"
all_dirs=utils.current_all_dir(dataset_dir)
all_dirs=sorted(all_dirs)
dir_dict={}
name_dict={}
for dir_ in all_dirs:
    dir_list=dir_.split("/")
    if len(dir_list)==3:
        dir_dict[dir_list[2]]=[]
        name_dict[dir_list[2]]=[]
    else:
        dir_dict[dir_list[2]].append(dir_)
        name=dir_.split("/")[-1].split(".")[0]
        name_dict[dir_list[2]].append(name)

# スペクトログラムに変換
data_dict=copy.deepcopy(dir_dict) # dictionaryの初期化が面倒なのでcopy
data_dict={k:[] for k in data_dict.keys()}
print("PSD Calculation...")
min_th=20
max_th=8000
for k, v in dir_dict.items():
    create_dir=[
            result_dir+"/psd/"+k
            ]
    utils.make_dir(create_dir)
    for i in tqdm.tqdm(range(len(v))):
        directory=v[i]
        rate, data=sp.read_signal(directory)
        f, psd=sp.psd(data, fs=rate)
        # 低周波の削除
        psd=psd[min_th<f]
        f=f[min_th<f]
        # 高周波の削除
        psd=psd[f<max_th]
        f=f[f<max_th]
        # save
        data_dict[k].append([f, psd])

        # psdの可視化
        name=directory.split("/")[-1].split(".")[0]
        outputdir=result_dir+"/psd/"+k+"/"
        data={"PSD": [f, psd]}
        vis.times_plot(data, outputdir+name+".png", x_label="frequency", y_label="Power", log=False)
        vis.times_plot(data, outputdir+"log_"+name+".png", x_label="frequency", y_label="Power", log=True)

# fitting
print("Fitting...")
fitting_dict=copy.deepcopy(dir_dict)
fitting_dict={k: [] for k in fitting_dict.keys()}
for k, v in data_dict.items():
    create_dir=[
            result_dir+"/fitting/"+k
            ]
    utils.make_dir(create_dir)
    for i in tqdm.tqdm(range(len(v))):
        # fitting
        x,y=v[i]
        param, cov=sp.fitting(x,y)
        fitting_y=[sp.function(tmp, param[0]) for tmp in x]
        data={
            "data":[x, y],
            "fitting data":[x, fitting_y]
            }
        directory=dir_dict[k][i]
        name=directory.split("/")[-1].split(".")[0]
        outputdir=result_dir+"/fitting/"+k+"/"
        # 可視化
        # non log
        vis.times_plot(data, outputdir+"log_"+name+".png", x_label="frequency", y_label="Power", log=True)
        vis.times_plot(data, outputdir+name+".png", x_label="frequency", y_label="Power", log=False)

        # save
        std=np.sqrt(np.diag(cov))
        fitting_dict[k].append([param[0], std[0]])

# json形式に変換する前に整形
result_dict={
        k: {name: {"param":result[0], "std":result[1]} for name, result in zip(dir_dict[k], fitting_dict[k])}
        for k in fitting_dict.keys()}

# json形式で結果の出力
f=open("result.json", "w")
jsondata=json.dump(result_dict, f)

# csvで出力
tmp=[]
for k, v in result_dict.items():
    for directory, data in v.items():
        name=directory.split("/")[-1].split(".")[0]
        tmp.append([name, k, directory, data["std"]])
columns=["name", "category", "directory", "std"]
df=pd.DataFrame(tmp, columns=columns)
df.to_csv("result.csv")
