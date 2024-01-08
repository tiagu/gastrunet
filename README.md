

# GastrUnet
Gastrunet is a pipeline for fast semantic segmentation of microscopy images of 3D cell cultures for the purpose of analysing their overall shape and growth. 


## Overview:
This repository uses pytorch, fastai and albumentations to train a neural network with a small number of manually-segmented images as ground-truth. I also include a notebook with several tools that can be used to analyse the masks obtained. These were originally performed on notoroids (see Rito et al. 2023).


## Example:
* RGB images of organoid expressing GFP. 
The pipeline extracts (attempts) the maximum length of the 3D object from the 2D images and estimates the amount of (bright) GFP signal within. 

![alt-text](https://github.com/tiagu/gastrunet/blob/main/example_GFP/demo.gif)