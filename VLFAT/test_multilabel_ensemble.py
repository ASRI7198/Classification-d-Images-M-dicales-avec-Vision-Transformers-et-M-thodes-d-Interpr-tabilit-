#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import inspect
import os
import random
import sys

import json
import time

import re

import yaml

import math
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader

from utils.utils import transform_custom, showLR, draw_results
from utils.channel_wise_aug import augmentations, available_augmentations
from utils.preprocess import random_idxs, middle_idxs
from utils.scheduler_utils import create_scheduler
from utils.optimizer_utils import create_optimizer
from utils.model_utils import *
from utils.load_config import read_conf_file, create_logger
from utils.dataset import BrestOCT_Binary_DS, BrestOCT_Multilabel_DS

from test_files.test import draw_conMatrix, draw_results, draw_roc
from test_files.plot_roc import binary_auc, multiclass_auc

import prettycm as pcm

import time
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics

import matplotlib.pylab as plt
import torchvision.transforms as T
from matplotlib.pyplot import figure

# In[2]:

# In[3]:
torch.manual_seed(99)

# config_path = "./config/YML_files/VLFATBREST_MULTILABEL_TEST.yaml"


config_path = "./config/YML_files/VLFAT_VarIN_BREST_MULTILABEL_TEST.yaml"

model_config, dataset_info, train_config, log_info, where, config, \
    device, model_layout, check_point_path, model_name, results_path, logger = read_conf_file(config_path)

# In[3]:


gray_scale = True if model_config['channels'] == 1 else False

logger.info('Load OCTBREST set for training ...')

train_set = BrestOCT_Multilabel_DS(loader_type=dataset_info['loader_type'],
                                   annotation_path=where + '/' + dataset_info['annotation_path'],
                                   mode="train", val_fold=dataset_info['val'], test_fold=dataset_info['test'],
                                   augment=True,
                                   augmentation_list=available_augmentations,
                                   image_size=model_config['image_size'],
                                   categories=["MLA", "DMLA E", "DMLA A", "OMC diabètique", "OMC", "IVM",
                                               "Autres patho"],
                                   model_type=model_config['model_type'],
                                   gray_scale=gray_scale,
                                   n_frames=model_config['num_frames'],
                                   var_input=model_config['var_input'],
                                   logger=logger, where=where)

# In[5]:


val_set = BrestOCT_Multilabel_DS(loader_type=dataset_info['loader_type_val'],
                                 annotation_path=where + '/' + dataset_info['annotation_path'],
                                 mode="val", val_fold=dataset_info['val'], test_fold=dataset_info['test'],
                                 augment=False,
                                 augmentation_list=available_augmentations,
                                 image_size=model_config['image_size'],
                                 categories=["MLA", "DMLA E", "DMLA A", "OMC diabètique", "OMC", "IVM", "Autres patho"],
                                 model_type=model_config['model_type'],
                                 gray_scale=gray_scale,
                                 n_frames=model_config['num_frames'],
                                 var_input=model_config['var_input'],
                                 logger=logger, where=where)

test_set = BrestOCT_Multilabel_DS(loader_type=dataset_info['loader_type_test'],
                                  annotation_path=where + '/' + dataset_info['annotation_path'],
                                  mode="test", val_fold=dataset_info['val'], test_fold=dataset_info['test'],
                                  augment=False,
                                  augmentation_list=available_augmentations,
                                  image_size=model_config['image_size'],
                                  categories=["MLA", "DMLA E", "DMLA A", "OMC diabètique", "OMC", "IVM",
                                              "Autres patho"],
                                  model_type=model_config['model_type'],
                                  gray_scale=gray_scale,
                                  n_frames=model_config['num_frames'],
                                  var_input=model_config['var_input'],
                                  logger=logger, where=where)

# In[6]:


# Afficher la taille des ensembles de données
print("Taille de l'ensemble d'entraînement :", len(train_set))
print("Taille de l'ensemble de validation :", len(val_set))
print("Taille de l'ensemble de test :", len(test_set))

# In[7]:


train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_config['batch_size'],
    #            batch_size=2,
    shuffle=True,
    num_workers=dataset_info['num_workers'],
    pin_memory=True,
)

val_loader = DataLoader(
    dataset=val_set,
    batch_size=1,
    shuffle=False,
    num_workers=dataset_info['num_workers'],
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False,
    num_workers=dataset_info['num_workers'],
    pin_memory=True,
)

VLFAT = False
if model_config['model_type'] == 'ViT_VaR':
    VLFAT = True

if model_config['model_type'] == 'ViT_VaR_VarIN':
    VLFAT = True

if model_config['weighted']:
    weight_type = 'balanced'
else:
    weight_type = None

""" Setting up the loss function, optimizer, and schedulers"""
"""Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer"""
# set the loss function and the optimizers
cycle_momentum = True
if train_config['optimizer'] == 'adam':
    cycle_momentum = False

if model_config['weighted']:
    weight_type = 'balanced'
else:
    weight_type = None

classes_weights = train_loader.dataset.cal_cls_weight()
# loss_fn = nn.CrossEntropyLoss(weight=classes_weights).to(device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=classes_weights).to(device)


####################################################################################
################ Model Binary Ensemble ##################################################
####################################################################################


class Model_Ensemble(nn.Module):
    def __init__(self, load_weights, model_config, model_layout, logger, path1, path2, path3, path4):
        super(Model_Ensemble, self).__init__()

        self.model1 = ViT_create_model(model_config, model_layout, logger)
        if load_weights == True:
            self.model1.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
            self.model1.eval()

        self.model2 = ViT_create_model(model_config, model_layout, logger)
        if load_weights == True:
            self.model2.load_state_dict(torch.load(path2, map_location=torch.device('cpu')))
            self.model2.eval()

        self.model3 = ViT_create_model(model_config, model_layout, logger)
        if load_weights == True:
            self.model3.load_state_dict(torch.load(path3, map_location=torch.device('cpu')))
            self.model3.eval()

        self.model4 = ViT_create_model(model_config, model_layout, logger)
        if load_weights == True:
            self.model4.load_state_dict(torch.load(path4, map_location=torch.device('cpu')))
            self.model4.eval()

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)

        # torch.softmax(val_pred_logit1,dim =1)

        #         print('out1=',out1)
        #         print('out2=',out2)
        #         print('out=',out)

        out = (out1 + out2 + out3 + out4) / 4
        return out


class model_multilabel:
    def __init__(self):
   
        self.checkpoint1 = "/home2/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_MULTILABEL/20240701-150344/bestmodel_auc.pth"
        self.checkpoint2 = "/home2/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_MULTILABEL/20240701-150400/bestmodel_auc.pth"
        self.checkpoint3 = "/home2/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_MULTILABEL/20240701-150415/bestmodel_hamming.pth"
        self.checkpoint4 = "/home2/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_MULTILABEL/20240701-150438/bestmodel_hamming.pth"

        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cuda")

    def load(self, model_config, model_layout, logger):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        # self.model = Model_ensemble(load_weights=False)
        # join paths
        checkpoint_path1 = os.path.join(self.checkpoint1)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint1))
        checkpoint_path2 = os.path.join(self.checkpoint2)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint2))
        checkpoint_path3 = os.path.join(self.checkpoint3)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint3))
        checkpoint_path4 = os.path.join(self.checkpoint4)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint4))

        self.model = Model_Ensemble(True, model_config, model_layout, logger, checkpoint_path1, checkpoint_path2,
                                    checkpoint_path3, checkpoint_path4)
        self.model.to(self.device)

    def predict(self, x):
        with torch.no_grad():
            output = self.model(x)

        return output


###############################################################################
### test fonction for ensemble 
###############################################################################

from sklearn import metrics

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve


# Function defining the Modified Hamming loss JustRAIGS

def hamming_loss(true_labels, predicted_labels):
    """Calculate the Hamming loss for the given true and predicted labels."""
    # Convert to numpy arrays for efficient computation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate the hamming distance that is basically the total number of mismatches
    Hamming_distance = np.sum(np.not_equal(true_labels, predicted_labels))
    # print("Hamming distance", Hamming_distance)

    # Calculate the total number of labels
    total_corrected_labels = true_labels.size

    # Compute the Modified Hamming loss
    loss = Hamming_distance / total_corrected_labels
    return loss


def test_multilabel(loader, model, loss_fn, logger, phase, device='cuda'):
    # model.eval()
    running_loss = 0
    all_trues = []
    all_probs = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.float().to(device)
            logits = model.predict(images)
            probs = torch.sigmoid(logits)  # Utilisez sigmoid pour obtenir les probabilités
            preds = (probs > 0.5).float()  # Utilisez un seuil de 0.5 pour obtenir des prédictions binaires
            all_trues.append(labels.cpu())
            all_probs.append(probs.cpu())

            # Pour le multilabel, nous utilisons la perte de Hamming comme mesure de précision
            running_loss += loss_fn(logits, labels).item() * images.size(0)

    # Concaténez tous les batches pour calculer les métriques
    all_trues = torch.cat(all_trues, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = (all_probs > 0.5).astype(int)  # Convertissez les probabilités en prédictions binaires

    # Calculez les métriques pour le multilabel
    auc = roc_auc_score(all_trues, all_probs, average='macro', multi_class='ovr')  # AUC pour le multilabel
    average_accuracy = accuracy_score(all_trues, all_preds)  # Précision moyenne
    hamming_loss_value = hamming_loss(all_trues, all_preds)  # Perte de Hamming
    # Si vous avez des classes déséquilibrées, vous pourriez vouloir utiliser le score F1 au lieu de l'accuracy
    f1 = f1_score(all_trues, all_preds, average='weighted')

    logger.info('[INFO] {} loss, accuracy, AUC, Hamming Loss, f1: {}, {}, {}, {}, {}'.format(
        phase, running_loss / len(loader.dataset), average_accuracy, auc, hamming_loss_value, f1))

    return average_accuracy, running_loss / len(loader.dataset), auc, hamming_loss_value, f1


def test_complete_multilabel(loader, model, loss_fn, logger, phase, save_path, device='cuda', n_test=0):
    # model.eval()
    running_loss = 0
    all_trues = []
    all_probs = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.float().to(device)
            logits = model.predict(images)

            # Pour la classification multilabel, utilisez sigmoid au lieu de softmax
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()  # Seuil pour déterminer les prédictions

            # Stockez les vraies étiquettes et les probabilités pour le calcul ultérieur des métriques
            all_trues.append(labels.cpu())
            all_probs.append(probs.cpu())

            # Pour la classification multilabel, calculez la perte de Hamming ou la perte BCEWithLogits
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    # Concaténez toutes les données
    all_trues = torch.cat(all_trues, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = (all_probs > 0.5).astype(int)

    # Calcul des métriques pour la classification multilabel
    hamming_loss_value = hamming_loss(all_trues, all_preds)  # Perte de Hamming
    average_f1 = f1_score(all_trues, all_preds, average='weighted')  # Score F1 moyen

    # Rapport de classification pour les problèmes multilabel
    classification_rpt = classification_report(all_trues, all_preds, target_names=loader.dataset.categories)

    logger.info(f"[INFO] {phase} Hamming Loss: {hamming_loss_value}")
    logger.info(f"[INFO] {phase} Average F1 Score: {average_f1}")
    logger.info(f"[INFO] {phase} Classification Report:\n {classification_rpt}")

    # Enregistrer les prédictions dans un fichier CSV
    output_df = pd.DataFrame({
        'true_labels': list(all_trues),
        'pred_probs': list(all_probs),
        'pred_labels': list(all_preds)
    })
    output_file = save_path + "/pred_outputs.csv"
    output_df.to_csv(output_file, index=False)

    # Vous devrez mettre à jour votre fonction de matrice de confusion pour qu'elle fonctionne avec le multilabel
    # draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories)
    # Même chose pour la courbe ROC si applicable
    # auc = draw_roc(trues, y_prob, save_path, logger)  # Pas directement applicable au multilabel

    return average_f1, running_loss / len(loader.dataset), hamming_loss_value


# In[23]:


model = model_multilabel()
model.load(model_config, model_layout, logger)

test_avg_accuracy, test_loss, test_auc, test_hamming, test_f1 = test_multilabel(test_loader, model, loss_fn, logger,
                                                                                phase='test')
_test_avg_f1, _test_loss, _test_hamming = test_complete_multilabel(test_loader, model, loss_fn, logger, phase="test",
                                                                   save_path=results_path)
