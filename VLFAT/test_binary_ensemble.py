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

import seaborn as sns
import yaml

import math
import os
from captum.attr import IntegratedGradients, Occlusion, GradientShap, Saliency, DeepLift
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import random
import matplotlib
from numpy import mean
from torch.utils.data import Dataset, DataLoader

from utils_files.utils import transform_custom, showLR, draw_results
from utils_files.channel_wise_aug import augmentations, available_augmentations
from utils_files.preprocess import random_idxs, middle_idxs
from utils_files.scheduler_utils import create_scheduler
from utils_files.optimizer_utils import create_optimizer
from utils_files.optimizer_utils import create_optimizer
from utils_files.model_utils import *
from utils_files.load_config import read_conf_file, create_logger
from utils_files.dataset import BrestOCT_Binary_DS, BrestOCT_Multilabel_DS, OCTDL_Binary_DS
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


# config_path = "./config/YML_files/VLFATBREST_BINARY_TEST.yaml"
config_path = "./config/YML_files/VLFAT_VarIN_BREST_BINARY_TEST.yaml"

path_slice = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/data_slice_resampling_info.xls"
df_dataSlice = pd.read_excel(path_slice, index_col=None, header=0)

model_config, dataset_info, train_config, log_info, where, config, \
    device, model_layout, check_point_path, model_name, results_path, logger = read_conf_file(config_path)

# In[4]:
device = torch.device("cuda")

gray_scale = True if model_config['channels'] == 1 else False

test_set = BrestOCT_Binary_DS(loader_type=dataset_info['loader_type_test'],
                              annotation_path=where + '/' + dataset_info['annotation_path'],
                              mode="test", val_fold=dataset_info['val'], test_fold=dataset_info['test'],
                              augment=False,
                              augmentation_list=available_augmentations,
                              image_size=model_config['image_size'],
                              categories=["normal", "anormal"],
                              model_type=model_config['model_type'],
                              gray_scale=gray_scale,
                              n_frames=model_config['num_frames'],
                              var_input=model_config['var_input'],
                              logger=logger, where=where,
                              db_main_path='/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/OCTBREST/',
                              slice_df=df_dataSlice)

# In[6]:


# Afficher la taille des ensembles de données
print("Taille de l'ensemble de test :", len(test_set))

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

classes_weights = test_loader.dataset.cal_cls_weight(weight_type=weight_type)
loss_fn = nn.CrossEntropyLoss(weight=classes_weights).to(device)

print("device", device)


####################################################################################
################ Model Binary Ensemble ##################################################
####################################################################################


class Model_Ensemble(nn.Module):
    def __init__(self, load_weights, model_config, model_layout, logger, path1):
        super(Model_Ensemble, self).__init__()

        self.model1 = ViT_create_model(model_config, model_layout, logger)
        if load_weights:
            self.model1.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
            self.model1.eval()
        '''
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
         '''

    def forward(self, x):
        out1 = self.model1(x)
        '''
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)
        '''
        # torch.softmax(val_pred_logit1,dim =1)

        #         print('out1=',out1)
        #         print('out2=',out2)
        #         print('out=',out)

        #   out = (out1 + out2 + out3 + out4) / 4
        return out1


class model_binary:
    def __init__(self):
        # baseline VLFAT

        # self.checkpoint1 = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/train_output/VLFAT_BINARY/20240430-042649/bestmodel_auc.pth"
        # self.checkpoint2 = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/train_output/VLFAT_BINARY/20240430-042716/bestmodel_auc.pth"
        # self.checkpoint3 = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/train_output/VLFAT_BINARY/20240430-042749/bestmodel_auc.pth"
        # self.checkpoint4 = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/train_output/VLFAT_BINARY/20240430-042936/bestmodel_auc.pth"

        # VLFATVIVP baseline
        self.checkpoint1 = "/home/rasri/PycharmProjects/VLFAT_INTERPRETABILITY/train_output/VLFAT_VI_VP_BINARY/20240517-164008/bestmodel_auc.pth"
        """
        self.checkpoint1 = "/home2/pzhang/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_BINARY/20240517-164008/bestmodel_auc.pth"
        self.checkpoint2 = "/home2/pzhang/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_BINARY/20240517-173754/bestmodel_auc.pth"
        self.checkpoint3 = "/home2/pzhang/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_BINARY/20240519-034914/bestmodel_auc.pth"
        self.checkpoint4 = "/home2/pzhang/VLFAT3/train_output/ViT_VaR_VarIN_224_512_OCT_BREST_BINARY/20240519-184700/bestmodel_auc.pth"
        """

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
        ''''
        checkpoint_path2 = os.path.join(self.checkpoint2)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint2))
        checkpoint_path3 = os.path.join(self.checkpoint3)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint3))
        checkpoint_path4 = os.path.join(self.checkpoint4)
        logger.info('[INFO] Load checkpoint {} \n'.format(self.checkpoint4))
        '''
        self.model = Model_Ensemble(True, model_config, model_layout, logger, checkpoint_path1)
        self.model.to(self.device)

    def predict(self, x):
        # with torch.no_grad():
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


def test(loader, model, loss_fn, logger, phase, device='cpu'):
    # model.eval()
    running_corrects = 0
    running_loss = 0
    trues = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits = model.predict(images)

            prob, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            trues.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            y_prob.extend(prob.cpu())
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    auc = metrics.roc_auc_score(trues, y_prob)
    accuracy = running_corrects / len(loader.dataset)
    'y_true, y_pred'
    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info(
        '[INFO] {} acc, loss, balanced accuracy and auc: {}, {}, {}, {}'.format(phase, accuracy, loss, balanced_acc,
                                                                                auc))
    return accuracy, loss, balanced_acc, auc


def test_complete(loader, model, loss_fn, logger, phase, save_path, device='cpu', n_test=0):
    # model.eval()
    auc = 0
    running_corrects = 0
    running_loss = 0
    trues = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)  # accelerator handles it
            logits = model.predict(images)  # change here

            prob, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            trues.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            y_prob.extend(prob.cpu())
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    # Convertir chaque tenseur en entier
    trues_ = [true.item() for true in trues]
    y_prob_ = [prob.item() for prob in y_prob]

    # save csv target/prediction for stats
    data = {'y_target': trues_, 'y_prediction': y_prob_}
    df = pd.DataFrame(data)

    # Save DataFrame to a CSV file
    print("save_path :", save_path)
    df.to_csv('predict_output.csv', index=False)

    accuracy = running_corrects / len(loader.dataset)
    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info(
        '[INFO] running corrects:{}, {} acc, balanced accuracy,  and loss, n_test: {}, {}, {}, {}'.format(
            running_corrects, phase, accuracy, balanced_acc,
            loss, n_test))
    "get the classification report"
    classification_report = metrics.classification_report(y_true=trues, y_pred=y_pred,
                                                          target_names=loader.dataset.categories)
    logger.info('[INFO] classification report \n')
    logger.info(classification_report)
    #
    n_classes = len(loader.dataset.categories)
    categories = loader.dataset.categories
    draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories)

    if len(loader.dataset.categories) == 2:
        "cal auc for binary classification"
        auc = draw_roc(trues, y_prob, save_path, logger)

    return balanced_acc, loss, auc


# In[23]:


model = model_binary()
model.load(model_config, model_layout, logger)

'''
test_accuracy, test_loss, test_balanced_acc, test_auc = test(test_loader, model, loss_fn, logger, phase='test')
test_bacc, test_loss, test_auc = test_complete(test_loader, model, loss_fn, logger, phase="test",
                                               save_path=results_path)
'''


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # Convertir chaque numpy array en tensor PyTorch et transférer sur le bon device
    # all_layer_matrices = [torch.tensor(layer_matrix, device="cuda" if torch.cuda.is_available() else "cpu")
    #                      for layer_matrix in all_layer_matrices]
    num_tokens = all_layer_matrices[0].shape[1]
    device = all_layer_matrices[0].device
    # Créer une matrice identité augmentée pour chaque batch
    eye = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(1, 1, 1)

    # Ajouter la matrice identité à chaque matrice d'attention pour inclure l'auto-attention résiduelle
    all_layer_matrices = [layer_matrix + eye for layer_matrix in all_layer_matrices]

    # Normaliser les matrices d'attention
    matrices_aug = [layer_matrix / layer_matrix.sum(dim=-1, keepdim=True) for layer_matrix in all_layer_matrices]
    # Calcul er la matrice d'attention globale en partant de la couche spécifiée
    joint_attention = matrices_aug[start_layer]
    for matrix in matrices_aug[start_layer + 1:]:
        joint_attention = torch.bmm(matrix, joint_attention)

    return joint_attention


def forward_wrapper(input_tensor):
    outputs = model.predict(input_tensor)
    # print("Output : ", outputs)
    return outputs


class ClassificationModelWrapper(nn.Module):
    def __init__(self, model):
        super(ClassificationModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model.predict(x)
        return output


wrapped_model = ClassificationModelWrapper(model)

liste = []
test_labels_slices2 = []
for i, test_data in enumerate(test_loader):
    test_images, test_labels, test_labels_slices = test_data[0].to(device), test_data[1].to(device), test_data[2].to(
        device)
    test_labels_slices = test_labels_slices.squeeze().squeeze().detach().cpu().tolist()

    #------------------------ calcul la AUC pour chaque methde de Captum pour les deux methode------------------------------------------------------------------------

    '''
    # Methode des Captum 
    
    ig = IntegratedGradients(forward_wrapper)
    attributions, _ = ig.attribute(input_tensor, target=1, return_convergence_delta=True) 
    
    saliency = Saliency(forward_wrapper)
    attributions = saliency.attribute(input_tensor, target=1)  # torch.Size([1, N, 3, 224, 224])
    
    deeplift = DeepLift(wrapped_model)
    attributions = deeplift.attribute(input_tensor, baselines=baseline_tensor, target=1)
    gradient_shap = GradientShap(forward_wrapper)
    attributions = gradient_shap.attribute(
        inputs=input_tensor,
        baselines=baseline_tensor,
        target=1,
        n_samples=10,  # Nombre d'échantillons pour SmoothGrad
        stdevs=0.09  # Écart type pour le bruit ajouté
    )
    
    occ = Occlusion(forward_wrapper)
    attributions = occ.attribute(input_tensor, target=1, sliding_window_shapes=(1, 3, 32, 32))
    
    # Pour la methode de VLFAT_Var_IN
    
    Nbcoupes = test_images.shape[1]
    H = test_images.shape[3]
    W = test_images.shape[4]
    Attribution = torch.empty(H, W, Nbcoupes)

    for j in range(Nbcoupes):
        Attribution[:, :, j] = attributions[0, j, 0, :, :]

    # print("Attribution.shape : ", Attribution.shape)  # torch.Size([  H, W, N])

    attributions_np = Attribution.detach().cpu().numpy()
    attributions_np = attributions_np.mean(axis=(0, 1))  # size = N

    print("attributions_np : ", attributions_np)
    print("test_labels_slices : ", test_labels_slices)

    y_targets.extend(test_labels_slices)
    y_predictions.extend(attributions_np)
    
    
    # Pour la methode de VLFAT
    
    Nbcoupes = test_images.shape[1]

    Attribution = torch.empty(224, 224, Nbcoupes)

    for j in range(Nbcoupes):
        Attribution[:, :, j] = attributions[0, j, 0, :, :]

    # print("Attribution.shape : ", Attribution.shape)  # torch.Size([H, 224, 224])

    attributions_np = Attribution.detach().cpu().numpy()
    attributions_np = attributions_np.mean(axis=(0, 1))  # size = N

    print("attributions_np : ", attributions_np)
    print("test_labels_slices : ", test_labels_slices)

    y_targets.extend(test_labels_slices)
    y_predictions.extend(attributions_np)
    
    '''
    #------------------------ Calcul de AUC pour chaque volume juste pour les Volumes avec label 1 et 0 ---------------------------------------

    # pour filtrer les volumes avec des labele de 1 et 0
    somme = 0
    for element in test_labels_slices: somme += element
    N = test_images.shape[1]

    if somme != N and somme != 0:
        Volume = test_images
        Name = test_data[3][0]
        name_volume = Name
        Name = Name.replace("/", "|")
        classe, atten = model.predict(Volume.float())

        for j in range(len(atten)):
            atten[j] = atten[j].mean(dim=1)

        Matrice_Rollout = compute_rollout_attention(atten)
        Matrice_Rollout = Matrice_Rollout.squeeze(0)

        mask = Matrice_Rollout[0, 1:]

        mask = mask.detach().cuda().cpu().numpy()
        metric2 = metrics.roc_auc_score(test_labels_slices, mask)
        liste.append(metric2)


    #-------------------------- Creation des Heatmap--------------------------------------------------------------------
    '''
    attention_scores_2d = np.array(attributions).reshape(1, -1)
    plt.figure(figsize=(14, 4))  # Ajustez la taille pour une meilleure visualisation horizontale
    ax = sns.heatmap(attention_scores_2d, annot=True, cmap='viridis', vmin=0.014322852,
                     vmax=0.06828364, cbar=True, linewidths=0.5,
                     linecolor='black',
                     fmt='.18f')

    for text in ax.texts:
        text.set_rotation(90)

    ax.text(-0.5, -0.02, 'Slices :', ha='right', va='top', fontsize=12, color='black',
            transform=ax.get_xaxis_transform())

    plt.title('Attention Scores Heatmap')

    ax.text(-0.5, -0.2, 'labels :', ha='right', va='top', fontsize=12, color='black',
            transform=ax.get_xaxis_transform())

    for j, label in enumerate(test_labels_slices):
        ax.text(j + 0.5, -0.2, str(label), ha='center', va='top', color='black', transform=ax.get_xaxis_transform())

    # ax.text(-0.5, -0.4, 'labels 2 :', ha='right', va='top', fontsize=12, color='black',
    #        transform=ax.get_xaxis_transform())

    taille = len(mask)
    # Ajuster les indices de l'axe x pour commencer à 1
    ax.set_xticks(np.arange(taille) + 0.5)
    ax.set_xticklabels(range(1, taille+1))

    plt.savefig(os.path.join("Visualisation/Rollout VLFAT Binary avent softmax", Name), bbox_inches='tight',
                pad_inches=0)

    # Extraire les couleurs de la heatmap
    colors = []
    for k in range(attention_scores_2d.shape[1]):
        # Extraire la couleur de chaque cellule de la heatmap
        color = ax.collections[0].get_facecolors()[k]
        colors.append(matplotlib.colors.rgb2hex(color))

    # Créer un DataFrame avec les couleurs
    colors_df = pd.DataFrame(colors, columns=['colors'])

    # Sauvegarder les couleurs dans un fichier CSV
    colors_df.to_csv(os.path.join(f"Visualisation/fichier csv avent softmax/{Name}"), index=False)
    plt.close()
    '''


print("Values de AUC est : ", liste)
print("Moyenne de AUC est : ", mean(liste))



'''

print("taille de y_targets : ", len(y_targets))
print("taille de y_predictions : ", len(y_predictions))

metric2 = metrics.roc_auc_score(y_targets, y_predictions)

print("AUC on data test (Rollout VLFAT Binary) = ", metric2)
'''

'''
VLFAT_BINARY : 

AUC on data test (saliency) = 0.31448354603369433
AUC on data test (DeepLift) = 0.5082699827800715
IG ==> Processus arrêté en CPU / Out of Memory en GPU
Gradient Shap ==> Processus arrêté en CPU
Occlusion ==> Processus arrêté en CPU

VLFAT_VI_VP_BINARY : 

AUC on data test (saliency) = 0.5249173637514146
AUC on data test (DeepLift) = 0.36330803326732247
IG ==> Processus arrêté en CPU / Out of Memory en GPU
Gradient Shap ==> Processus arrêté en CPU / Out of Memory en GPU
Occlusion ==> Processus arrêté en CPU / Out of Memory en GPU
'''

