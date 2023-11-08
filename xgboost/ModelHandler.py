import sys, os, subprocess, json
from datetime import datetime
import numpy as np
import pandas as pd
import uproot
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
from pprint import pprint
import matplotlib as mpl
# https://matplotlib.org/faq/usage_faq.html
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import cos
import random
import ROOT
from ROOT import *

class ModelHandler:
    """XGBoost Model training and validation"""
    def __init__(self, config):
        with open(config, 'r') as file:
            json_file = json.load(file)
        self.do_validation_plots = json_file['validation_plots']
        self.output_path = json_file['output_path']
        self.features = json_file['feature_names']
        self.all_branches = json_file['feature_names'] + json_file['other_branches']
        self.do_weight = json_file['do_weight']
        self.weight_column = json_file['weight_column']
        self.tree_name = json_file['tree_name']
        self.Y_column = json_file['Y_column']
        self.BDT_0 = json_file['BDT_0']
        self.data = None #ALL data
        self.train_data = None #Data with all the branches
        self.test_data = None
        self.x_train = None # Data for train and test with only feature_names branches for x
        self.x_test  = None
        self.y_train = None # Data for train and test with only Y_column branches for x
        self.y_test  = None
        self.model = None
        self.roc_bkg_min = 0.2
        self.roc_bkg_max = 1.0
        self.roc_sig_min = 0.4
        self.roc_sig_max = 1.0
        self.train_weights = None
        self.test_weights = None
        self.y_test_all = None
        self.y_pred_all = None
        self.do_cat = False
        self.save_ntuple=False

    def load_data(self, file_name):
        """Load ROOT data and turn tree into a pd dataframe"""
        print("Loading data from", file_name)
        f = uproot.open(file_name)
        tree = f[self.tree_name]
        branches = self.all_branches
        data = tree.arrays(branches ,library="pd")
        return data
    
    @staticmethod
    def merge_data(datasets):
        """Merge multiple datasets into one dataset."""
        if len(datasets) < 1:
            return pd.DataFrame() # empty DataFrame
        
        dataset = pd.concat(datasets, ignore_index=True, join='inner') # join='inner' eliminates any columns that are not common to all df
        return dataset
    
    @staticmethod
    def plot_histogram(dataframe, column_name, lable, num_bins=10, xlabel=None, ylabel=None, title=None):
        """Crea un istogramma a partire dai dati in una colonna di un DataFrame di pandas. """
        data = dataframe[column_name]
        plt.hist(data, bins=num_bins, edgecolor='k', alpha=0.65)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.savefig("./plot_"+column_name+lable+".png", format='png', dpi=300)

    def load_datasets(self, input_list, config):
        """Load and merge ROOT trees with MVA data into a single dataset."""
        with open(config, 'r') as file:
            json_file = json.load(file)
        fraction = json_file['event_fraction']
        if fraction!=0:
            event_sel = int(1/fraction)
        if fraction==0:
            print("Error, no events!")
        datasets = []
        if self.data:
            datasets.append(self.data)
        batch_index=0
        
        for entry in input_list:
            files = subprocess.check_output("find %s -type f -name '*root'" % entry, shell=True)
            for f in files.splitlines():
                if batch_index%event_sel==0:
                    datasets.append(self.load_data(f.decode()))
                batch_index += 1
        print("Done!")
        #print("datasets[0]", datasets[0])
        self.data = self.merge_data(datasets)
        print("Initial length of data: ", (self.data).shape[0])
        print("self.data", self.data["isMC"])
        #print("Total number of events:", len(self.data[self.event_branch_name]))

    def apply_selection(self, config, selection_name='selections_1', mask2 = None):
        with open(config, 'r') as file:
            json_file = json.load(file)
        selections = json_file[selection_name]
        print(selections)
        mask = pd.Series(True, index=(self.data).index)
        for selection in selections:
            print("selection",selection)
            group_mask = pd.Series(False, index=(self.data).index)
            for key, value in selection.items():
                print("key, value", key, value)
                operator = value[0]
                threshold = float(value[1])
                #print(operator, " ",threshold)
                if operator == "<":
                    group_mask = group_mask | (self.data[key] < threshold)  # OR
                elif operator == ">":
                    group_mask = group_mask | (self.data[key] > threshold)  # OR
                elif operator == "==":
                    group_mask = group_mask | (self.data[key] == threshold)  # OR
                elif operator == "!=":
                    group_mask = group_mask | (self.data[key] != threshold)  # OR
                #print(group_mask)
            mask = mask & group_mask  # AND

        if selection_name.endswith('!'):
            mask = ~mask
        
        if mask2 is not None:
            mask = mask | mask2
        
        # Applica la maschera ai dati
        selected_data = self.data[mask]
        selected_data = selected_data.reset_index(drop=True)
        self.data = selected_data
        print("Length of data after selections: ", (self.data).shape[0])
    
    def prepare_train_and_test_datasets(self, config, n=4, index=0):
        """Split dataset into train and test subsets using event number."""

        mask_train = self.data.index % n != index #La maschera è fatta su tutto il df senza separare bkg e sig
        mask_test = self.data.index % n == index
        self.train_data = self.data[mask_train]
        self.test_data = self.data[mask_test]
        #print("self.data: ", self.data)
        
        self.x_train = self.train_data[self.features]
        self.x_test = self.test_data[self.features]
        self.y_train = self.train_data[self.Y_column] #This is a pd series
        self.y_test = self.test_data[self.Y_column]
        
        self.train_weights = self.train_data[self.weight_column]
        self.test_weights = self.test_data[self.weight_column]
        #print(self.train_weights)
        
        #Rimuovere dopo il test:
        self.train_weights = self.train_weights.clip(upper=1)
        self.test_weights = self.test_weights.clip(upper=1)
                
        self.y_train = self.y_train.apply(lambda x: x != self.BDT_0)
        self.y_test = self.y_test.apply(lambda x: x != self.BDT_0)
        
    def get_parameters(self, config):
        """Get parameters for XGBoost."""
        with open(config, 'r') as file:
            json_file = json.load(file)
        parameters = json_file["xgb_parameters"]
        param = {}
        for entry in parameters:
            key = list(entry.keys())[0]
            value = entry[key]
            param[key] = value
        #return param
        return param

    def train(self, config, model_name="test", num_boost_round=5000, target_s_over_b=None):
        """Train a model"""
        
        param = self.get_parameters(config)
        pprint(param)
        #print("item=",list(param.items()))
        
        self.model = xgb.XGBRegressor(**param)
        
        if self.do_weight:
            self.model.fit(self.x_train, self.y_train,
                            verbose               = True,
                            sample_weight = self.train_weights,
                            eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)]
                        )
                        
        else:
            self.model.fit(self.x_train, self.y_train,
                            verbose               = True,
                            eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)]
                        )
                    
        booster = (self.model).get_booster()
        importance_temp = booster.get_score(importance_type="gain")
                
        max_length = max([len(s) for s in self.features])
        feature_format = "%" + "%u" % max_length + "s"

        print("Importance scores:")
        for name in self.features:
            try:
                print((feature_format + " %0.1f") % (name, importance_temp[name]))
            except KeyError:
                print((feature_format + " unused") % name)
                    
        with open("%s/%s.pkl" % (self.output_path, model_name), 'wb') as f:
            pickle.dump(self.model, f)
                                  
        if self.do_validation_plots:
            self.make_validation_plots(model_name)
            
    def load_models(self, config, model_name="test", date ="20231107-1847"):
        """Load BDT models"""
        lable_name = model_name+"-"+date+"-Event"
        file_list = os.listdir(self.output_path)
        models = [file for file in file_list if file.startswith(lable_name) and file.endswith(".pkl")]
        for i in models:
            print(cartella + i)
        
        

    def make_validation_plots(self, model_name="test"):
        """Plot ROC curves and score distributions"""
        y_pred_train = pd.Series(self.model.predict(self.x_train), name='y_pred')
        y_pred_test = pd.Series(self.model.predict(self.x_test), name='y_pred')

        y_pred_train.index = self.y_train.index
        y_pred_test.index = self.y_test.index
        
        fig, ax = plt.subplots()
        
        preds_bkg_test  = y_pred_test[self.y_test == False]
        preds_sig_test  = y_pred_test[self.y_test == True]
        preds_bkg_train = y_pred_train[self.y_train == False]
        preds_sig_train = y_pred_train[self.y_train == True]

        density = True
        bins = np.linspace(0.0, 1, 50)
        edges = 0.5 * (bins[:-1] + bins[1:])

        from scipy import stats
        pks_bkg = stats.ks_2samp(preds_bkg_train, preds_bkg_test)[1]
        pks_sig = stats.ks_2samp(preds_sig_train, preds_sig_test)[1]

        #print(preds_bkg_train)
        if self.do_weight:
            counts_train_bkg, _, _ = ax.hist(
            preds_bkg_train,
            bins=bins,
            histtype="stepfilled",
            alpha=0.45,
            density=True,
            label="bkg, train",
            color=["b"],
            weights=self.train_weights[self.y_train==0])
        else:
            counts_train_bkg, _, _ = ax.hist(
            preds_bkg_train,
            bins=bins,
            histtype="stepfilled",
            alpha=0.45,
            density=True,
            label="bkg, train",
            color=["b"])

        counts_train_sig, _, _ = ax.hist(
            preds_sig_train,
            bins=bins,
            histtype="stepfilled",
            alpha=0.45,
            density=True,
            label="sig, train",
            color=["r"])
        if self.do_weight:
            counts_test_bkg, _, _ = ax.hist(
            preds_bkg_test,
            bins=bins,
            histtype="step",
            alpha=1.0,
            density=True,
            label="bkg, test (KS prob = {:.2f})".format(pks_bkg),
            color=["b"],
            lw=1.5,
            linestyle="solid",
            weights=self.test_weights[self.y_test==0])
        else:
            counts_test_bkg, _, _ = ax.hist(
            preds_bkg_test,
            bins=bins,
            histtype="step",
            alpha=1.0,
            density=True,
            label="bkg, test (KS prob = {:.2f})".format(pks_bkg),
            color=["b"],
            lw=1.5,
            linestyle="solid")

        counts_test_sig, _, _ = ax.hist(
            preds_sig_test,
            bins=bins,
            histtype="step",
            alpha=1.0,
            density=True,
            label="sig, test (KS prob = {:.2f})".format(pks_sig),
            color=["r"],
            lw=1.5,
            linestyle="solid")

        ax.set_yscale("log")
        ax.legend()

        ax.set_ylim([0.01, ax.get_ylim()[1]])
        fig.set_tight_layout(True)
        fig.savefig("%s/%s-validation-disc.pdf" % (self.output_path, model_name))
        print("%s-validation-disc.pdf has been saved in %s" % (model_name, self.output_path))

        fpr_test, tpr_test, _ = roc_curve(self.y_test, y_pred_test)
        print("fpr_test=",fpr_test.shape)
        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_pred_train)
        auc_test = roc_auc_score(self.y_test, y_pred_test)
        auc_train = roc_auc_score(self.y_train, y_pred_train)
        fig, ax = plt.subplots()
        plt.grid(which='both', axis='both')
        ax.plot(fpr_test, tpr_test, label="test AUC = {:.6f}".format(auc_test))
        ax.plot(fpr_train, tpr_train, label="train AUC = {:.6f}".format(auc_train))
        ax.set_xlabel("bkg eff")
        ax.set_ylabel("sig eff")
        ax.set_xlim([self.roc_bkg_min, self.roc_bkg_max])
        ax.set_ylim([self.roc_sig_min, self.roc_sig_max])
        ax.set_title("ROC curves")

        if self.do_weight:
            fpr_test_w, tpr_test_w, _ = roc_curve(self.y_test, y_pred_test,sample_weight=self.test_weights)
            fpr_train_w, tpr_train_w, _ = roc_curve(self.y_train, y_pred_train,sample_weight=self.train_weights)
            weighted_auc_test = roc_auc_score(self.y_test, y_pred_test,sample_weight=self.test_weights)
            weighted_auc_train = roc_auc_score(self.y_train, y_pred_train,sample_weight=self.train_weights)
            ax.plot(fpr_test_w, tpr_test_w, label="test AUC (weighted) = {:.6f}".format(weighted_auc_test))
            ax.plot(fpr_train_w, tpr_train_w, label="train AUC (weighted) = {:.6f}".format(weighted_auc_train))

        ax.legend()
        fig.set_tight_layout(True)
        #ax.set_yscale("log")
        #ax.set_xscale("log")
        fig.savefig("%s/%s-validation-roc.pdf" % (self.output_path, model_name))
        print("%s-validation-roc.pdf has been saved in %s" % (model_name, self.output_path))
