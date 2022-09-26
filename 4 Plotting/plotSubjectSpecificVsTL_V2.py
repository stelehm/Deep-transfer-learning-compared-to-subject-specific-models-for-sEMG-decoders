# -*- coding: utf-8 -*-
"""
subject specific model vs fine tuning
@author: StephanLehmler
"""

#plot logs
basedirectory_logs = "../3 Experiments/models/{}_{}/logs/"

basefilename_retrain_logs = {"DB2features": "model{}_83Perc_all_retrain_history_log.csv",
                              "DB2raw": "model{}_samples_83Perc_method_all_retrain_history_log.csv",
                              "DB3features": "model{}_samples_83Perc_method_all_retrain_history_log.csv",
                              "DB3raw": "model{}_samples_83Perc_method_all_retrain_history_log.csv",
                              "DB4features": "model{}_83Perc_all_retrain_history_log.csv",
                              "DB4raw": "model{}_samples_83Perc_method_all_retrain_history_log.csv"
                              }
basefilename_new_logs = {"DB2features": "model{}_83Perc_newModel_history_log.csv",
                              "DB2raw": "model{}_samples83Perc_newModel_history_log.csv",
                              "DB3features": "model{}_samples83Perc_newModel_history_log.csv",
                              "DB3raw": "model{}_samples83Perc_newModel_history_log.csv",
                              "DB4features": "model{}_83Perc_newModel_history_log.csv",
                              "DB4raw": "model1_samples83Perc_newModel_history_log.csv"
                              }

import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()
sns.set(font_scale=3)

for db,startS, nbSubject in [("DB2",11,41),("DB3",1,12),("DB4",1,11)]:
    df_list=[]
    for i,color in [("raw", "Reds"), ("features","Blues")]:
        #df_list = []
        for s in range(startS,nbSubject):
            dirname = basedirectory_logs.format(db,i)
            df = pd.read_csv(dirname + basefilename_retrain_logs[db+i].format(s))
            df["db"] = db
            df["inputs"] = i
            df["subject"] = s
            df["Model"] ="retrained"
            df_list.append(df)
            df_new = pd.read_csv(dirname + basefilename_new_logs[db+i].format(s))
            df_new["db"] = db
            df_new["inputs"] = i
            df_new["subject"] = s
            df_new["Model"] = "subject-specific"
            df_list.append(df_new)
    currentresults = pd.concat(df_list)
        
        #only every third epoch
        #currentresults = currentresults[currentresults["epoch"].isin(list(range(0,currentresults.epoch.max(),3)))]
    currentresults.rename(columns = {'val_accuracy':'Validation Accuracy', "inputs": "Input", "epoch":"Epoch"}, inplace = True)
        
    fig= plt.figure()
        #sns.boxplot( data=currentresults, x='epoch', y='Validation Accuracy', hue='Model',saturation=.5,palette=color).set(title='Inputs: {}'.format(i))
        

    g = sns.lineplot(x="Epoch", y="Validation Accuracy",
             hue="Input", style="Model",
             palette="coolwarm_r", #saturation=.5,
             ci="sd",
             markers=True, dashes=True,
             mew=0,
             linewidth = 3,
             legend= "brief",
             data=currentresults)
    g.legend(fontsize=25)
        #don't show all ticks
    # for xtick in fig.get_axes()[0].get_xticklabels():
    #     if int(xtick.get_text()) % 5 != 0:
    #         xtick.set_visible(False)

    fig.set_size_inches(20,12)
    
    fig.savefig("{}_SSvsTL_V2_HD.svg".format(db), dpi=600,format="svg")
    fig.savefig("{}_SSvsTL_V2_HD.pdf".format(db), dpi=600,format="pdf")



