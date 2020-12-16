from flask import Flask, render_template, jsonify, send_file
import json
from collections import OrderedDict

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
            music_dir[3]="log_"+music_dir[3].split(".")[0]+".png"
        tmps["static/analysis_result/fitting/"+music_dir[2]+"/"+music_dir[3]]=v1["std"]

pic_dict={"dirs":[], "std":[]}
for k, v in sorted(tmps.items(), key=lambda item: item[1]):
    pic_dict["dirs"].append(k)
    pic_dict["std"].append(v)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_pic")
def get_pic_dir():
    return jsonify(pic_dict)

@app.route("/get_std")
def get_std():
    return jsonify(result_dict)

@app.route("/result.csv")
def get_csv():
    return send_file("./result.csv")

if __name__=="__main__":
    app.run(debug=True)

