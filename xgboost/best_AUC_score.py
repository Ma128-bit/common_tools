import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
n_splits = 5
out_path = "results/ROCurvs/"
in_path = "results/BDT/"

def best_ROC_in_category(name, category, ax, type="all", sel = "max", weights = True):
    if category=="A":
        trs_test = 0.915
    elif category=="B":
        trs_test = 0.904
    else:
        trs_test = 0.933
    if weights == True:
        w = "-w"
        w2 = "_w"
        w3 = "weighted_"
    else:
        w=""
        w2=""
        w3=""
    file = f"-Category_Cat_{category}-validation-roc"+w+".npz"
    fpr_test = []
    tpr_test = []
    fpr_train = []
    tpr_train = []
    if sel=="min":
        auc_test = 1
    else:
        auc_test = -1
    auc_train = -1
    for i in range(n_splits):
        file_name = in_path + name +"/" + f"Run2022-Event{i}" + file
        loaded_data = np.load(file_name)
        auc_test_temp = loaded_data[w3+"auc_test"]
        if((auc_test_temp>auc_test and sel=="max") or (auc_test_temp<auc_test and sel=="min")):
            auc_test = auc_test_temp
            auc_train = loaded_data[w3+"auc_train"]
            fpr_test = loaded_data["fpr_test"+w2]
            tpr_test = loaded_data["tpr_test"+w2]
            fpr_train = loaded_data["fpr_train"+w2]
            tpr_train = loaded_data["tpr_train"+w2]
        del loaded_data

    if auc_test>trs_test and sel == "max":
        if type == "train":
            ax.plot(fpr_train, tpr_train, label="train "+name+" AUC = {:.6f}".format(auc_train))
        elif type == "test":
            ax.plot(fpr_test, tpr_test, label="test "+name+" AUC = {:.6f}".format(auc_test))
        else:
            ax.plot(fpr_train, tpr_train, label="train "+name+" AUC = {:.6f}".format(auc_train))
            ax.plot(fpr_test, tpr_test, label="test "+name+" AUC = {:.6f}".format(auc_test))
    elif ((auc_test>trs_test-0.005 and category!="C") or (auc_test>trs_test-0.01 and category=="C")) and sel == "min":
        if type == "train":
            ax.plot(fpr_train, tpr_train, label="train "+name+" AUC = {:.6f}".format(auc_train))
        elif type == "test":
            ax.plot(fpr_test, tpr_test, label="test "+name+" AUC = {:.6f}".format(auc_test))
        else:
            ax.plot(fpr_train, tpr_train, label="train "+name+" AUC = {:.6f}".format(auc_train))
            ax.plot(fpr_test, tpr_test, label="test "+name+" AUC = {:.6f}".format(auc_test))

def best_AUC_in_category(name, category, weights = True):
    if weights == True:
        w = "-w"
        w2 = "_w"
        w3 = "weighted_"
    else:
        w=""
        w2=""
        w3=""
    file = f"-Category_Cat_{category}-validation-roc"+w+".npz"
    auc_test_mean = 0
    auc_train_mean = 0
    for i in range(n_splits):
        file_name = in_path + name +"/" + f"Run2022-Event{i}" + file
        loaded_data = np.load(file_name)
        auc_test_temp = loaded_data[w3+"auc_test"]
        auc_train_temp = loaded_data[w3+"auc_train"]
        auc_test_mean = auc_test_mean + auc_test_temp
        auc_train_mean = auc_train_mean + auc_train_temp
        del loaded_data
        
    auc_test_mean = auc_test_mean/n_splits
    auc_train_mean = auc_train_mean/n_splits

    return [auc_train_mean, auc_test_mean]

def draw_category(names, category, type="all", sel = "max", weights = True):
    fig, ax = plt.subplots()
    plt.grid(which='both', axis='both')
    for name in names:
        best_ROC_in_category(name, category, ax, type, sel, weights)
    ax.set_xlabel("bkg eff")
    ax.set_ylabel("sig eff")
    ax.set_xlim([0.05, 1])
    ax.set_ylim([0.6, 1])
    ax.set_yscale("log")
    ax.set_title("ROC curves")
    ax.legend()
    #fig.set_tight_layout(True)
    fig.savefig(out_path +"Category_"+category+"_type_"+type+"_sel_"+sel+"-roc_"+ names[0].split('-')[0]+".pdf")

def histo_AMS(names, category, type="test", weights = True):
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)
    plt.subplots_adjust(bottom=0.25)
    #plt.subplots_adjust(top=0.05)
    #plt.grid(which='both', axis='both')
    index = 1 #test
    if type=="train":
        index = 0
    auc_test = []
    #auc_train = []
    for name in names:
        vector = best_AUC_in_category(name, category, weights)
        if type == "all":
            auc_test.append(vector[1]/(vector[0]-vector[1]))
        else:
            auc_test.append(vector[index])
        #auc_train.append(vector[0])
    colori = ['orange' if (valore > max(auc_test)-(max(auc_test)-min(auc_test))/20 and valore!=max(auc_test)) else 'red' if valore==max(auc_test) else 'blue' for valore in auc_test]
    plt.bar(names, auc_test, color=colori)
    plt.xticks(rotation='vertical')
    #ax.set_xlabel("File")
    ax.set_ylabel("AUC")
    ax.set_ylim([min(auc_test)-0.015, max(auc_test)+0.015])
    #ax.set_yscale("log")
    ax.set_title("AUC score")
    
    #ax.legend()
    fig.savefig(out_path +"Category_"+category+"_type_"+type+"-auc_"+ names[0].split('-')[0]+".pdf")


if __name__ == "__main__":
    #files = ["20240124-104401", "20240124-122044", "20240124-132322", "20240124-134302", "20240124-134256", "20240124-134305", "20240124-134308", "20240124-134311", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ]
    
    list = os.listdir(in_path)
    files = [i for i in list if os.path.isdir(os.path.join(in_path, i)) and "20240124" in i]
    histo_AMS(files, "A", type="test")
    histo_AMS(files, "B", type="test")
    histo_AMS(files, "C", type="test")
    histo_AMS(files, "A", type="train")
    histo_AMS(files, "B", type="train")
    histo_AMS(files, "C", type="train")
    histo_AMS(files, "A", type="all")
    histo_AMS(files, "B", type="all")
    histo_AMS(files, "C", type="all")
    """
    draw_category(files, "A", type="test", sel ="min")
    draw_category(files, "B", type="test", sel ="min")
    draw_category(files, "C", type="test", sel ="min")
    draw_category(files, "A", type="train", sel ="min")
    draw_category(files, "B", type="train", sel ="min")
    draw_category(files, "C", type="train", sel ="min")
    """


