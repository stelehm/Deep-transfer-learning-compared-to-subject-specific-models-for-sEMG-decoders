# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:42:25 2021

Plot the percentage change and safe as svg

@author: StephanLehmler
"""


import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.ioff()

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

sns.set(font_scale=3)
df_list = []
basefilename = "../3 Experiments/results/results_{}_{}.csv"
for db in ["DB2","DB3","DB4"]:
    for i in ["features","raw"]:
        filename = basefilename.format(db,i)
        df = pd.read_csv(filename)
        df["db"] = db
        df["inputs"] = i
        df_list.append(df)
    df = pd.concat(df_list)
    df = df[df["retraining_method"] != " new"]
    df = df[df["retraining_samples"].isin([" 83Perc", " 60Perc", " 30Perc", " 16Perc"])]
    df = df.replace([" 83Perc", " 60Perc", " 30Perc", " 16Perc"], ["5 Repetitions", "4 Repetitions","2 Repetitions", "1 Repetition"])

    df["Percentage Improvement"] = (df['r_test_accuracy'] - df['pre_retraining_test_accuracy'] ) / df['pre_retraining_test_accuracy'] *100


    g = sns.catplot(#x="retraining_method", 
                y="Percentage Improvement",
                col="retraining_samples", 
                x="inputs",
                hue="inputs",
                aspect=1,
                dodge=False,
                kind="box",
                palette="coolwarm",
                data=df)

    g.set_titles("{col_name} for retraining")
    
    g.fig.suptitle("Test set accuracy percentage change on {}".format(db))

    for ax in g.axes.ravel():
        ax.hlines(0,*ax.get_xlim())
        ax.set_xlabel("")
    
    #plt.get_current_fig_manager().full_screen_toggle()
    #plt.get_current_fig_manager().window.showMaximized()
    g.fig.set_size_inches(30,18)
    
    g.savefig("{}_PercentageChange_HD.svg".format(db), dpi=600,format="svg")
    g.savefig("{}_PercentageChange_HD.pdf".format(db), dpi=600,format="pdf")
    
    
