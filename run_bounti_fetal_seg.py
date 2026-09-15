""" 
Script for Patched-based training/testing of BOUNTI-TR with transient regions included: 
 - Trains with patches of size 128x128x128 of t2w MRI from 21-36 weeks gestational age
 - Model trains to segment 31 regions including Subplate, Ganglionic Eminence, Ventricular Zone and PVWM (R and L)

"""

from __future__ import print_function
import sys
import os
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib 
from sklearn.model_selection import train_test_split
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandAffined,
    ToTensord, 
    ToNumpy,
    ToTensor,
    RandSpatialCropd,
    RandAdjustContrastd
)
from monai.metrics import DiceMetric
from monai.networks.nets import AttentionUnet
from monai.data import (
    DataLoader,
    CacheDataset, decollate_batch)
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

############## DEFINE TRAIN / TEST SETTINGS ############## 
train_dataset = sys.argv[1] # path for folder with training subjects
train_labels = sys.argv[2] # path for folder with training labels

test_dataset = sys.argv[3] # path for folder with test subjects

check_path = sys.argv[4] # checkpoint path : /.../checkpoints-BOUNTI 
results_path = sys.argv[5] # path to save data results 

res = int(sys.argv[6]) # patch-size used: 128 
cl_num = int(sys.argv[7]) # label number: 31

status_train_proc = int(sys.argv[8]) # if I want to train or not (1) : use 1 to train, 0 to test
status_load_check = int(sys.argv[9]) # if I want to load checkpoint or not (1): use 0 for trainnig from scratch or 1 to load checkpoint
max_iterations = int(sys.argv[10]) # iterations for training (e.g: 200000)

############## DEFINE TRAIN / TEST TRANSFORMATIONS ############## 
# ROTATION
degree_min = -0.6
degree_max = 0.6

# # BIAS-FIELD 
coef_min = -0.5
coef_max = 0.5
# #
# ADJUST CONTRAST 
gam_min = 0.5
gam_max = 2.5

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandBiasFieldd(
            keys=["image"],
            degree=3, 
            coeff_range=(coef_min, coef_max), 
            prob=0.40,
        ),
        RandSpatialCropd(keys= ["image", "label"], roi_size= (128,128,128), random_center = True, random_size = False),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0),
        RandGaussianNoised(
            keys=["image"],
            prob=0.40),
        RandAffined(
            keys=["image", "label"],
            rotate_range=[(degree_min,degree_max),(degree_min,degree_max),(degree_min,degree_max)],
            mode=("bilinear", "nearest"),
            padding_mode=("zeros"),
            prob=0.40),
        # RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 4.5)
        # ),
        # RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.7, 1.7) ),
        RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(gam_min, gam_max)),
        ToTensord(keys=["image", "label"])
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),
        ToTensord(keys=["image", "label"]),
    ]
)


run_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),
        ToTensord(keys=["image"]),
    ]
)

############## DEFINE TRAIN & TEST DATASETS  ############## 

image_paths_train = sorted(glob.glob(os.path.join(train_dataset, "*.nii.gz"))) 
label_paths_train = sorted(glob.glob(os.path.join(train_labels, "*.nii.gz"))) 

image_paths_test = sorted(glob.glob(os.path.join(test_dataset, "*.nii.gz"))) 

dict_data_train_val = [{"image": image_scan, "label": labels_seg}
         for image_scan, labels_seg in zip(image_paths_train, label_paths_train)]

dict_test = [{"image": image_scan}
        for image_scan in zip(image_paths_test)]

dic_train, dic_val = train_test_split(dict_data_train_val, train_size = 0.9, random_state = 1, shuffle= False)

#############################################################################################################
# LOADING DATASETS
#############################################################################################################
print("Loading data ...")

if status_train_proc > 0:

    train_ds = CacheDataset(
        data = dic_train,
        transform = train_transforms,
        cache_num = 100,
        cache_rate = 1.0,
        num_workers = 8)
    
    train_loader = DataLoader(
    train_ds, batch_size= 2, shuffle=True, num_workers=8, pin_memory=True)
    
    val_ds = CacheDataset(
        data = dic_val, 
        transform = val_transforms, 
        cache_num = 6, 
        cache_rate = 1.0, 
        num_workers = 4)
    
    val_loader = DataLoader(
    val_ds, batch_size= 1, shuffle=False, num_workers=4, pin_memory=True)

else:

    run_ds = CacheDataset(
        data = dict_test,
        transform= run_transforms,
        cache_num = 6,
        cache_rate = 1.0, 
        num_workers = 4)

    run_loader = DataLoader(
        run_ds, batch_size= 1, shuffle=False, num_workers=4, pin_memory=True)


#############################################################################################################
# DEFINING MODEL 
#############################################################################################################
print("Defining the model ...")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionUnet(spatial_dims=3,
                     in_channels=1,
                     out_channels=cl_num+1,               
                     channels=(16, 32, 64, 128, 256),
                     strides=(2,2,2,2),
                     kernel_size=3,
                     up_kernel_size=3,
                     dropout=0.5).to(device)

#############################################################################################################
# USE CHECKPOINT TO CONTINUE TRAINING
#############################################################################################################

if status_load_check > 0 :

    print("Loading BOUNTI checkpoints into the model for training ...")
    # to continue running the model 
    model.load_state_dict(torch.load(os.path.join(check_path, ("best_metric_model.pth"))), strict=False)

#############################################################################################################
# TRAINING 
#############################################################################################################

if status_train_proc > 0:

    print("Training ...")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def validation(epoch_iterator_val):
        model.eval()

        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (res, res, res), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
                )
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            

            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(check_path, ("best_metric_model.pth"))
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    torch.save(
                        model.state_dict(), os.path.join(check_path, ("latest_metric_model.pth"))
                    )
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )

            global_step += 1

        return global_step, dice_val_best, global_step_best


    #############################################################################################################
    # DEFINE PARAMETERS FOR TRAINING, etc. 
    #############################################################################################################
    eval_num = 20000
    post_label = AsDiscrete(to_onehot=cl_num+1)
    post_pred = AsDiscrete(argmax=True, to_onehot=cl_num+1)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    #############################################################################################################
    # START TRAINING/TESTING PROCESS
    #############################################################################################################

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )

else: 

    #############################################################################################################
    # FOR TESTING MODEL
    #############################################################################################################

    print("Running ...")
    model.load_state_dict(torch.load(os.path.join(check_path, ("best_metric_model.pth"))), strict=False) # load checkpoint
    mean_dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)  

    for x in range(len(dict_test)): 

        case_num = x
        img = run_ds[case_num]["image"]
        name = dict_test[case_num].get("image")[0]
        print("Case number: ", case_num, " - ", name)
        out_name_sub = name.replace(test_dataset + '/', '')
        out_name_sub = out_name_sub.replace('.nii.gz', '') 

        out_name = results_path + '/' + out_name_sub + "-label.nii.gz"

        print(case_num, out_name)
        img_tmp_info = nib.load(os.path.join(name))
        
        with torch.no_grad():
            run_inputs = torch.unsqueeze(img, 1).cuda()
            run_outputs = sliding_window_inference(
                run_inputs, (res, res, res), 4, model, overlap=0.8
            )

            out_label = torch.argmax(run_outputs, dim=1).detach().cpu()[0, :, :, :]
            out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
            nib.save(out_lab_nii, out_name)

#############################################################################################################