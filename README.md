# Fetal-brain-transient-segmentation

An automatic model - BOUNTI-TR - to segment T2w MRI of the fetal brain into 31 ROIs (of left and right hemispheres) from 21-36 weeks of gestational age, including transient regions: Subplate, Ventricular Zone and Ganglionic Eminence (see [^4] for segmentation details) and Periventricular White Matter (subsegmented in 3 crossroad portions: C1, C2+C5 and C4 - see [^5] for segmentation details). 

This model was trained with a large and heteregeneous cohort with different aquisition parameters (FOV=1.5-3T, TE=80-250ms), including the developing human connectome project (dHCP) dataset [^1] and clinical cohorts from St. Thomas Hospital, London, UK.

Please refer to [^2] and [^3] for the repository and paper regarding the reconstruction/regional segmentation pipeline. Further details on methods for training and validating the model can be found on our paper [^4]

<img width="1908" height="969" alt="Segmentations-model 001" src="https://github.com/user-attachments/assets/9eb668ef-b203-42bf-a476-440bccb9f3f7" />

## Instructions to run BOUNTI-TR:

Preprocessing of the T2w images (in ".nii.gz" format) needs to be done following the tools on this docker [^6] 
**Please read instructions for using the docker to preprocess images in "instructions.txt" in "preprocessing.zip" - preprocessing folder with files for it!!!**

1) T2w MRI images for input to BOUNTI-TR should be brain extracted - in case you need to do skull stripping you can use the 3D CNN tool in [^6] (STEP 2 in "instructions.txt")

2) Images need to be preprocessed using the file "preprocess.sh" inside the preprocessing folder "preprocessing.zip" (STEP 4 in "instructions.txt")

3) Once preprocess is done use these outputs to either train from scratch (you'll need T2w images + corresponding training labels of the tissues) or test the trained model on your machine (only T2w images needed) - you can also use the environement shared here "bounti-tr_env.yml" to run it
   
4) To visualise the segmentations in your desired software you can import the label descriptions from the file "BOUNTI-TR-labels.txt"

# Train from scratch example: 
python ./run_bounti_fetal_seg.py ./train-imgs-folder ./train-labels-folder ./test-imgs-folder ./checkpoint-folder ./results-folder 128 31 1 0 200000 

# Test trained model example: 
python ./run_bounti_fetal_seg.py ./train-imgs-folder ./train-labels-folder ./test-imgs-folder ./checkpoint-folder ./results-folder 128 31 0 1 1 

- make sure the ./checkpoint-folder has the "best-metric-model.pth" file in case of doing inference.

[^1]: https://www.developingconnectome.org/ 
[^2]: https://github.com/SVRTK/auto-proc-svrtk
[^3]: https://www.biorxiv.org/content/10.1101/2023.04.18.537347v2 
[^4]: https://link.springer.com/chapter/10.1007/978-3-031-45544-5_2 
[^5]: https://dl.acm.org/doi/abs/10.1007/978-3-032-05997-0_10
[^6]: https://hub.docker.com/r/fetalsvrtk/segmentation
