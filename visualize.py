import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import warnings
import imgkit
import glob
import os
import copy
import tqdm

class TableVis():
    def __init__(self, df=None, directory=None, correct_column=None, dropcolumns=None):
        if not df is None:
            self.df=df.copy()
        if not dropcolumns is None:
            self.df=self.df.drop(dropcolumns, axis=1)
        self.directory=directory
        self.correct_column=correct_column

    def make_dir(self, required_dirs):
        def current_all_dir(directory="."):
            tmp = []
            for tmp_directory in glob.glob(directory+"/*"):
                tmp.append(tmp_directory)
                tmp.extend(current_all_dir(tmp_directory))
            return tmp
        dirs = current_all_dir()
        for required_dir in required_dirs:
            required_dir = "./"+required_dir
            if not required_dir in dirs:
                print("generate file in current dir...")
                print("+ "+required_dir)
                os.mkdir(required_dir)
                print("\n")

    def vis_all(self):
        print("Start TableVisualization....")
        required_dir=[
                "nan",
                "dist",
                "compare_dist",
                "ratio",
                "countdist",
                "cross_count"
                ]
        required_dir=[self.directory+dir_ for dir_ in required_dir]
        self.make_dir(required_dir)
        required_dir=[dir_+"/" for dir_ in required_dir]

        self.vis_nan(self.df, required_dir[0])
        self.vis_dist(self.df, required_dir[1])
        self.vis_compare_dist(self.df,required_dir[2],correct_column=self.correct_column)
        self.vis_ratio(self.df, required_dir[3])
        self.vis_count_dist(self.df, required_dir[4])
        self.vis_cross_count(self.df, required_dir[5])

    def vis_nan(self, df, directory):
        """
        Args:
            df: pandas dataframe形式のdeta
            directory: 画像などを出力するdirectory. "*/"
        """
        print("     Visualize Nan...")
        warnings.filterwarnings("ignore")
        df.isnull().sum().to_csv(directory+"nan_items.csv")

        plt.style.use('ggplot')
        plt.figure()
        sns.heatmap(df.isnull(),cbar=False,cmap='YlGnBu')
        plt.tight_layout()
        plt.savefig(directory+"nan_items.png")

    def vis_cross_count(self, df, directory):
        print("     Visualize Cross Count....")
        column_names=list(df.columns)
        combs=list(itertools.combinations(column_names, 2))
        for column1, column2 in combs:
            print("         [%s, %s]"%(column1, column2))
            tmp=pd.crosstab(
                    df[column1],
                    df[column2]
                    ).T.style.background_gradient(cmap='summer_r')
            html=tmp.render()
            imgkit.from_string(html, directory+'%s_%s.png'%(column1, column2))

    def vis_count_dist(self, df, directory):
        print("     Visualize Count Distribution....")
        for column in df.columns:
            plt.figure()
            sns.countplot(df[column])
            plt.tight_layout()
            plt.savefig(directory+column+".png")

    def vis_ratio(self, df, directory):
        import yaml
        print("     Visualize Ratio....")
        for column in df.columns:
            plt.figure()
            tmp=df[column].value_counts()
            plt.pie(list(tmp.values),labels=list(tmp.index))
            plt.savefig(directory+column+".png")
            sum_=sum(list(tmp.values))
            tmp={k: v/sum_ for k,v in tmp.items()}
            f=open(directory+column+".yaml", "w")
            f.write(yaml.dump(tmp))
            f.close()

    def vis_compare_dist(self, df, directory, correct_column):
        print("     Visualize Correct Compared Distribution....")
        # hist plot
        for column in df.columns:
            plt.figure()
            sns.countplot(correct_column,hue=column,linewidth=2.5,edgecolor=".2",data=df)
            plt.savefig(directory+"hist_"+column+".png")

        # stripplot
        for column in df.columns:
            plt.figure()
            sns.catplot(x=correct_column,y=column,data=df,kind="swarm")
            plt.savefig(directory+"dist_"+column+".png")

        # boxplot
        tmp_df=pd.DataFrame()
        tmp_df[correct_column]=list(df[correct_column])
        for column in df.columns:
            # 数値じゃない場合、boxplotできないので、数値に変換
            # ラベルエンコーディング
            if df[column].dtype==str or df[column].dtype==object:
                tmp_df[column]=self.label_encoding(df[column])
            else:
                tmp_df[column]=list(df[column])
            plt.figure()
            sns.catplot(x=correct_column,y=column,data=tmp_df,kind="box")
            plt.savefig(directory+"box_"+column+".png")

        # 二変数と教師ラベルの関係
        column_names=set(df.columns)
        column_names=column_names-set([correct_column])
        column_names=list(column_names)
        perms=list(itertools.permutations(column_names, 2))
        for column1, column2 in perms:
            plt.figure()
            sns.catplot(x=column1, y=correct_column, hue=column2, data=df, kind="point")
            plt.savefig(directory+"point_"+column1+"_"+column2+".png")
            plt.close()

    def vis_dist(self, df, directory):
        print("     Visualize Multi Distribution....")
        # stringだと、処理できないため
        df=df.copy()
        columns=list(df.columns)
        for column in columns:
            if df[column].dtype==str or df[column].dtype==object:
                df[column]=self.label_encoding(df[column])
        # kde plot
        for column in df.columns:
            plt.figure()
            sns.distplot(df[column])
            plt.tight_layout()
            plt.savefig(directory+"kde_"+column+".png")
        # 二変数間の散布図
        combs=list(itertools.combinations(columns, 2))
        for i in tqdm.tqdm(range(len(combs))):
            column1, column2=combs[i]
            plt.figure()
            sns.jointplot(x=column1, y=column2, data=df, kind="kde")
            plt.tight_layout()
            plt.savefig(directory+"double_"+column1+"_"+\
                    column2+".png")
            plt.close()
        # 三変数間の散布図
        combs=list(itertools.combinations(columns, 2))
        for i in tqdm.tqdm(range(len(combs))):
            column1, column2=combs[i]
            tmp=set(df.columns)-set([column1, column2])
            for column3 in tmp:
                plt.figure()
                sns.relplot(x=column1,y=column2,data=df,hue=column3)
                plt.tight_layout()
                plt.savefig(directory+"triple_"+column1+"_"+\
                        column2+"_"+column3+".png")
                plt.close()

    def label_encoding(self, column):
        import copy
        column=list(column)
        ele_set=set(column)
        d={k:num for k,num in zip(list(ele_set), range(len(ele_set)))}
        column=[d[ele] for ele in column]
        return column

def times_plot(data_dict, directory, x_label="", y_label="", log=False):
    """
    dataをdictionaryにして渡す. kがlegendのラベル. vは(x, y)のlist. x,yはそれぞれ時系列
    """
    plt.figure()
    ax=plt.gca()
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    for k, (x,y) in data_dict.items():
        plt.plot(x, y, label=k)
    plt.savefig(directory)
    plt.close()
    return

def compare_box_plot(data_dict, directory, datanum=None, xlabel="", ylabel="", palette="Set3", dodge=True, is_sort=False):
    """
    二つ以上の分布を比較するためのboxplot
    Args:
        data_dict: dataのdictionary. {key1: {key2: data}, ....}を想定. key1はboxの種類(legendで分けるやつ). key2は横軸の種類. dataはlistを想定
        directory: boxplotの比較を出力するdicrectory
        datanum: 各々の分布のデータ点数. Noneであれば小さいものに合わせる.
        xlabel: 横軸のラベル
        ylabel: 縦軸のラベル
        palette: boxplotの色palette. originalを指定する場合は, {data_dictのkey1: カラーコード, ...}で指定
    """
    # 分布のデータ点数の最小数
    if datanum is None:
        min_datanum=1e10
        for v1 in data_dict.values():
            for v2 in v1.values():
                min_datanum=min(min_datanum, len(v2))
        datanum=min_datanum
    # 分布のデータ点数を揃える
    data_dict={k1: {k2: v2[:datanum] for k2, v2 in data_dict[k1].items()} for k1 in data_dict.keys()}
    # boxplot
    fig = plt.figure()
    sns.set_context("paper", 1.2)
    ax = fig.add_subplot(1, 1, 1)
    df=pd.DataFrame()
    for key1, v in data_dict.items():
        tmp_df=pd.DataFrame(v)
        tmp_melt=pd.melt(tmp_df)
        tmp_melt["species"]=key1
        df=pd.concat([df, tmp_melt], axis=0).reset_index(drop=True)
    if is_sort:
        df = df.sort_values("variable", ascending=False) # x軸でsort
    sns.boxplot(x='variable', y='value', data=df, hue='species', dodge=dodge, showfliers=False, palette=palette, ax=ax)
    sns.stripplot(x='variable', y='value', data=df, hue='species', dodge=dodge, jitter=True, color='black', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:2], labels[0:2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(directory)

