from flask import Flask, render_template, jsonify, send_file, request
import json
from collections import OrderedDict
import shutil

import utils

# app, jinjaの設定変更
app=Flask(__name__)
jinja_options=app.jinja_options.copy()
jinja_options.update({
    "variable_start_string": "<<",
    "variable_end_string": ">>",
    })
app.jinja_options=jinja_options

f=open("result.json", "r")
result_dict=json.load(f)

# directoryをkey, 標準偏差をvalueとしたorderddict(sorted)を作成
tmps={}
for _, v in result_dict.items():
    for music_dir, v1 in v.items():
        music_dir=music_dir.split("/")
        if len(music_dir)==4:
            #music_dir[3]="log_"+music_dir[3].split(".")[0]+".png"
            music_dir[3]=music_dir[3].split(".")[0]+".png"
        tmps["static/analysis_result/fitting/"+music_dir[2]+"/"+music_dir[3]]=\
                [v1["rmse"], v1["param"]]

pic_dict={"dirs":[], "rmse":[], "param":[]}
for k, v in sorted(tmps.items(), key=lambda item: item[1]):
    pic_dict["dirs"].append(k)
    pic_dict["rmse"].append(v[0])
    pic_dict["param"].append(v[1])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_pic")
def get_pic_dir():
    return jsonify(pic_dict)

@app.route("/result.csv")
def get_csv():
    return send_file("./result.csv")

@app.route("/save")
def save():
    #th_std=float(request.args.get("th_std"))
    min_th=float(request.args.get("min_th"))
    max_th=float(request.args.get("max_th"))

    result_dir=["result"]
    shutil.rmtree("./result")
    mix_dir="result/mix"
    utils.make_dir(["result", mix_dir])
    for k, v in result_dict.items():
        copy_dir="result/"+k
        utils.make_dir([copy_dir])
        for directory, v1 in v.items():
            if v1["param"]<=max_th and min_th<=v1["param"]:
                tmp=directory.split("/")[-1]
                shutil.copyfile(directory, copy_dir+"/"+tmp)
                shutil.copyfile(directory, mix_dir+"/"+tmp)
    return


if __name__=="__main__":
    app.run(debug=True)

