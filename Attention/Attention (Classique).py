import time

import matplotlib
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import DataLoader
import aug
import os
import numpy as np
import torch
import ViT_3D.VisionTransformerBase
import seaborn as sns

# AddChannel,AsChannelFirst

torch.manual_seed(0)

num_epochs = 500
batch_size = 4
val_fold = 0
test_fold = 4

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "/home/rasri/PycharmProjects/Project/data_volume_resampling_info.xlsx"
df_dataset = pd.read_excel(path, index_col=None, header=0)

root = "/home/rasri/PycharmProjects/Project/OCT_BREST_RE2023/"
df_dataset.path1 = root + df_dataset.path1
df_dataset.path2 = root + df_dataset.path2

df_train_val = df_dataset[df_dataset['fold'] != test_fold]
df_train = df_train_val[df_train_val['fold'] != val_fold]
df_val = df_train_val[df_train_val['fold'] == val_fold]
df_test = df_dataset[df_dataset['fold'] == test_fold]

train_list = df_train['path2'].tolist()
train_label_list = df_train['vol_label'].tolist()
train_label_list = torch.nn.functional.one_hot(torch.as_tensor(train_label_list)).float()

val_list = df_val['path2'].tolist()
val_label_list = df_val['vol_label'].tolist()
val_label_list = torch.nn.functional.one_hot(torch.as_tensor(val_label_list)).float()

test_list = df_test['path2'].tolist()
test_label_list = df_test['vol_label'].tolist()
test_label_list = torch.nn.functional.one_hot(torch.as_tensor(test_label_list)).float()

print('train nombre:', len(train_list))
print('val nombre:', len(val_list))
print('test nombre:', len(test_list))

# OCTDATASET INTERPOLATION : REMOVE OR ADD BLACK SLICES


indices = list(range(1, 20))

print("indices : ", indices)


class OCTDataset(Dataset):
    def __init__(self,
                 patient_list,
                 label_list,
                 slice_df,
                 db_root,
                 aug_3d=None):
        self.patient_list = patient_list
        self.label_list = label_list
        self.slice_df = slice_df  # dataframe
        self.root = db_root
        self.aug_3d = aug_3d

    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        suffix_to_remove = "/oct_vol_preprocess.nii.gz"
        # Retirer le suffixe de la chaîne
        if patient.endswith(suffix_to_remove):
            patient_id = patient[:-len(suffix_to_remove)]
            patient_id = patient_id[len(self.root):]

        slicef = self.slice_df[self.slice_df['patient'] == patient_id]

        slice = slicef["slice_label"].tolist()

        label = self.label_list[idx]

        # load the volume
        volume_oct = nib.load(patient).get_fdata().astype(np.float32)
        volume_oct = volume_oct.transpose(1, 2, 0)

        # print the shape -> 19,200,1024 , D,H,W
        #         print('original shape ' , volume_oct.shape)
        slices_dest = 19

        # resize W
        volume_oct = resize(volume_oct, (volume_oct.shape[0], 192, 512), anti_aliasing=False)
        nb_slices = volume_oct.shape[0]

        if nb_slices > slices_dest:
            spacing = int((nb_slices - slices_dest) / 2)
            volume_input = volume_oct[spacing:spacing + slices_dest, :, :]
            sliceR = slice[spacing:spacing + slices_dest]
        elif nb_slices < slices_dest:
            spacing = int((slices_dest - nb_slices) / 2)
            volume_input = np.zeros(shape=(slices_dest, 192, 512))
            volume_input[spacing:spacing + nb_slices, :, :] = volume_oct
            vecteur = np.zeros(19)
            vecteur[spacing:spacing + nb_slices] = slice
            sliceR = vecteur
        else:
            volume_input = volume_oct
            sliceR = slice
        volume_input = volume_input[:, :, :, np.newaxis]

        if self.aug_3d is not None:
            data = {'image': volume_input}
            aug_data = aug(**data)
            volume_input = aug_data['image']

        volume_input = volume_input.transpose(3, 2, 1, 0)

        volume_input = torch.from_numpy(volume_input).float()

        # your data augementation here

        #         print('final shape ' , volume_input.shape)

        # return volume_input, label, torch.tensor(sliceR)

        return volume_input, label, torch.tensor(sliceR), patient_id, torch.tensor(slice)

    def __len__(self):
        return len(self.patient_list)


start = time.time()

summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda', torch.cuda.is_available())
print('gpu number', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)

path_slice = "/home/rasri/PycharmProjects/Project/OCT_BREST_RE2023/data_slice_resampling_info(1).xls"
df_dataSlice = pd.read_excel(path_slice, index_col=None, header=0)

test_ds = OCTDataset(patient_list=test_list, label_list=test_label_list, slice_df=df_dataSlice, db_root=root)
test_loader = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

inputs = torch.rand(1, 1, 512, 192, 19).to(device)
model = ViT_3D.VisionTransformerBase.ViT(in_channels=1, img_size=(512, 192, 19), patch_size=(512, 192, 1),
                                         hidden_size=768,
                                         pos_embed='perceptron', num_heads=12, classification=True,
                                         post_activation=None).to(device)
model.eval()
outputs = model(inputs)[0]
print(outputs.shape)

print(model)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
post_pred = torch.nn.Softmax(dim=1)
post_pred(outputs)

# start a typical PyTorch training
val_interval = 1

print('############# TRAIN FINISH，START TEST 1 ################')

print("Load weight with best auc")

best_model2_path = '/home/rasri/PycharmProjects/Stage_ASRI_RIDA_INTERPRETABILITY/Train/weights/bestmodel2.pth'

model.load_state_dict(torch.load(best_model2_path))
model.cuda()

model.eval()

y_targets = []
y_predictions = []

for i, test_data in enumerate(test_loader):
    test_images, test_labels, test_labels_slices = test_data[0].to(device), test_data[1].to(device), test_data[2].to(
        device)
    test_labels_slices = test_labels_slices.squeeze().squeeze().detach().cpu().tolist()

    Name = test_data[3][0]
    Name = Name.replace("/", "|")
    Volume = test_images

    classe, atten = model(Volume.float())

    # 12 blco Transformers chaque bloc il y a une matrice de [20,20]
    for j in range(len(atten)):
        atten[j] = atten[j].mean(dim=1)

    # Convertir la liste de matrices en un tenseur de dimensions [12, 20, 20]
    attention = torch.stack(atten)

    # Calculer la moyenne le long de la première dimension (les 12 blocs)
    attention = attention.mean(dim=0)

    attention = attention.squeeze()

    print("dimosnion de matrice est : ", attention.shape)

    mask = attention[0, 1:]
    mask = mask.detach().cuda().cpu().numpy()

    print("dimosnion de matrice est : ", mask.shape)

    y_targets.extend(test_labels_slices)
    y_predictions.extend(mask)

    # ------------------------------- Visualisation par Carte chaleur --------------------------------------------------

    attention_scores_2d = np.array(mask).reshape(1, -1)
    # Créer la heatmap
    plt.figure(figsize=(14, 4))  # Ajustez la taille pour une meilleure visualisation horizontale
    ax = sns.heatmap(attention_scores_2d, annot=True, cmap='viridis', vmin=-9.805855613715454e-06,
                     vmax=4.767875086315225e-06, cbar=True, linewidths=0.5,
                     linecolor='black',
                     fmt='.30f')
    for text in ax.texts:
        text.set_rotation(90)

    ax.text(-0.5, -0.02, 'Slices :', ha='right', va='top', fontsize=12, color='black',
            transform=ax.get_xaxis_transform())

    plt.title('Attribution Scores Heatmap')

    ax.text(-0.5, -0.2, 'labels :', ha='right', va='top', fontsize=12, color='black',
            transform=ax.get_xaxis_transform())

    for j, label in enumerate(test_labels_slices):
        ax.text(j + 0.5, -0.2, str(label), ha='center', va='top', color='black', transform=ax.get_xaxis_transform())

    ax.set_xticks(np.arange(19) + 0.5)
    ax.set_xticklabels(range(1, 20))

    plt.savefig(
        os.path.join("/home/rasri/PycharmProjects/Stage_ASRI_RIDA_INTERPRETABILITY/Heatmap/Attention/Attention ("
                     "Classique)", Name),
        bbox_inches='tight',
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
    colors_df.to_csv(os.path.join(f"/home/rasri/PycharmProjects/Stage_ASRI_RIDA_INTERPRETABILITY/Fichier CSV color "
                                  f"/Attention/Attention (Classique)/{Name}"), index=False)
    plt.close()

# AUC
print("y_targets : ", len(y_targets))
print("y_predictions : ", len(y_predictions))

metric2 = metrics.roc_auc_score(y_targets, y_predictions)

print("AUC on data test = ", metric2)
