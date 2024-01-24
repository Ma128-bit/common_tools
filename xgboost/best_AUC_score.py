import numpy as np
import matplotlib.pyplot as plt
n_splits = 5
out_path = "results/BDT/"

def best_AUC_in_category(name, category, ax, sel = "max", weights = True):
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
        file_name = out_path + name +"/" + f"Run2022-Event{i}" + file
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
        
    ax.plot(fpr_train, tpr_train, label="train AUC = {:.6f}".format(auc_train))
    ax.plot(fpr_test, tpr_test, label="test AUC = {:.6f}".format(auc_test))

def draw_category(names, category, sel = "max", weights = True):
    fig, ax = plt.subplots()
    plt.grid(which='both', axis='both')
    for name in names:
        best_AUC_in_category(name, category, ax, sel, weights)
    ax.set_xlabel("bkg eff")
    ax.set_ylabel("sig eff")
    ax.set_xlim([0.05, 1])
    ax.set_ylim([0.5, 1])
    ax.set_title("ROC curves")
    ax.legend('lower right')
    fig.set_tight_layout(True)
    fig.savefig(out_path +"global-roc_"+ names[0] + names[len(names)-1]+".pdf")


if __name__ == "__main__":
    draw_category(["20240124-104401"], "A")



