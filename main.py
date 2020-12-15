import copy
import matplotlib.pyplot as plt
import tqdm

import utils
import signal_process as sp

# result directoryの作成
result_dir="analysis_result"
required_dirs = [
        result_dir,
        result_dir+"/psd",
        ]
utils.make_dir(required_dirs)

# datasetの各データのdirectoryの取得
# ラベルをkey, 各データのdirectoryのlistをvalueとしたdictionaryで保存
dataset_dir="dataset"
all_dirs=utils.current_all_dir(dataset_dir)
all_dirs=sorted(all_dirs)
dir_dict={}
for dir_ in all_dirs:
    dir_list=dir_.split("/")
    if len(dir_list)==2:
        dir_dict[dir_list[1]]=[]
    else:
        dir_dict[dir_list[1]].append(dir_)
#prtmpint(dir_dict)

# スペクトログラムに変換
data_dict=copy.deepcopy(dir_dict) # dictionaryの初期化が面倒なのでcopy
data_dict={k:[] for k in data_dict.keys()}
for k, v in dir_dict.items():
    print("PSD Calculation...")
    create_dir=[
            result_dir+"/psd/"+k
            ]
    utils.make_dir(create_dir)
    for i in tqdm.tqdm(range(len(v))):
        directory=v[i]
        rate, data=sp.read_signal(directory)
        f, psd=sp.psd(data, fs=rate)
        # 低周波の削除
        f=f[1:]
        psd=psd[1:]
        # 高周波の削除
        psd=psd[f<rate//2]
        f=f[f<rate//2]
        # save
        data_dict[k].append([f, psd])

        # psdの可視化
        name=directory.split("/")[-1].split(".")[0]
        outputdir=result_dir+"/psd/"+k+"/"
        # log-log
        plt.figure()
        ax=plt.gca()
        ax.set_yscale("log")
        ax.set_xscale("log")
        plt.plot(f, psd)
        plt.savefig(outputdir+"log_"+name+".png")
        plt.close()
        # 通常
        plt.figure()
        plt.plot(f, psd)
        plt.savefig(outputdir+name+".png")
        plt.close()
