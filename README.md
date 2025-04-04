

# GastrUnet
Gastrunet is a pipeline for fast semantic segmentation of microscopy images of 3D cell cultures for the purpose of analysing their overall shape and growth. 

If you use this repository in your research, please cite:
Rito, T. *et al.* Timely TGFβ signalling inhibition induces notochord. Nature 637, 673–682 (2025). DOI: [10.1038/s41586-024-08332-w](https://doi.org/10.1038/s41586-024-08332-w).

## Overview:
This repository uses pytorch, fastai and albumentations to train a neural network with a small number of manually-segmented images as ground-truth. I also include a notebook with several tools that can be used to analyse the masks obtained. These were originally performed on notoroids (see Rito et al. 2023).


## Example:
* RGB images of organoid expressing GFP. 
The pipeline extracts (attempts) the maximum length of the 3D object from the 2D images and estimates the amount of (bright) GFP signal within. 

<p align="center">
  <img src="https://github.com/tiagu/gastrunet/blob/main/example_GFP/demo.gif" alt="alt-text">
</p>

Legend:

${\color{magenta}Magenta \space outline}$ - main organoid mask selected from the image. 
${\color{red}Red \space line}$ - main filament path used to calculate maximum length.
${\color{green}Green \space outlines}$ - detected bright GFP signals.
${\color{magenta}Magenta \space dots}$ - end-points extended to give a more realistic max length.


## Usage:

If your images look similar, feel free to use our trained models at:
https://zenodo.org/records/12684780
