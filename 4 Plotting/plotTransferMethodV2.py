# -*- coding: utf-8 -*-
"""
plot accuracy during training for different retraining methods 
plot raw and features in one figure V2
@author: StephanLehmler
"""

#plot logs
basedirectory_logs = "../3 Experiments/models/{}_{}/logs/"

basefilename_retrain_logs = {"DB2features": "model{}_83Perc_{}_retrain_history_log.csv",
                              "DB2raw": "model{}_samples_83Perc_method_{}_retrain_history_log.csv",
                              "DB3features": "model{}_samples_83Perc_method_{}_retrain_history_log.csv",
                              "DB3raw": "model{}_samples_83Perc_method_{}_retrain_history_log.csv",
                              "DB4features": "model{}_83Perc_{}_retrain_history_log.csv",
                              "DB4raw": "model{}_samples_83Perc_method_{}_retrain_history_log.csv"
                              }


import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()
sns.set(font_scale=3
        )
df_list = []
for db,startS, nbSubject in [("DB2",11,41),("DB3",1,12),("DB4",1,11)]:
    for i,color in [("raw", "Reds"), ("features","Blues")]:
        #df_list = []
        for m in ["all","first","last"]:
            for s in range(startS,nbSubject):
                dirname = basedirectory_logs.format(db,i)
                df = pd.read_csv(dirname + basefilename_retrain_logs[db+i].format(s,m))
                df["db"] = db
                df["inputs"] = i
                df["subject"] = s
                df["method"] = m
                df_list.append(df)
    currentresults = pd.concat(df_list)
        
        #only every fifths epoch
    #currentresults = currentresults[currentresults["epoch"].isin(list(range(0,currentresults.epoch.max(),2)))]
    currentresults.rename(columns = {'val_accuracy':'Validation Accuracy',"epoch":"Epoch"}, inplace = True)
        

    
    g = sns.relplot(#x="retraining_method", 
                y="Validation Accuracy",
                col="method",
                row_order=["features","raw"],
                x="Epoch",
                hue="inputs",
                style="inputs",
                markers=True, dashes=True,
                mew=0,
                kind="line",
                ci="sd",
                palette="coolwarm_r",
                linewidth=3,
                height=4,
                aspect=1.5,
                data=currentresults)
    
    g.set_titles("Retrain {col_name} layer")
    g.fig.legend(fontsize=25,title_fontsize=25,title="Input",bbox_to_anchor=[.97, 0.4],loc="right")
    g._legend.remove()
    #change layer to layers on firsat axis
    g.axes[0][0].set_title(g.axes[0][0].title.get_text().replace("layer","layers"))


    g.fig.set_size_inches(20,12)
    
    g.savefig("{}_TransferMethodV2_HD.svg".format(db), dpi=600,format="svg")
    g.savefig("{}_TransferMethodV2_HD.pdf".format(db), dpi=600,format="pdf")


