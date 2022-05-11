# Supervised and Self-Supervised Learning for Inverse Rendering of Human Faces

In this work, we aim to improve upon a state of the art inverse rendering method called SfSNet. We constrain SfSNet such that it always predicts unit normals and also introduce specularity in its rendering function with an aim to make it robust even with faces that contains specular highlights. Although its normal estimation gets improved than the original SfSNet work, specular highlights still persist on predicted albedo when tested on real faces with specular highlights. In spite of the fact that the rendering function with added specularity have not worked perfectly, it opens up new possibility of research on similar direction to solve this problem of specularity. 

## SfSNet-Constraints
Implmentation of the SfSNet with constraints can be found in `/sfsnet-constraints`. We explain below the role of each scripts present in that directory.

```
├─── datasets.py: Data loading and pre-processing methods 
├─── log.py: Log the messages and images using telegram bot
├─── utils.py: Helper functions 
├─── model.py: Definition of all the models used. SkipNet and SfSNet (Use during SfSNet Baseline)
├─── model_one.py: Definition of all the models used. SkipNet and SfSNet with 1st constraint (normalized normals)
├─── model_two.py: Definition of all the models used. SkipNet and SfSNet with 2nd constraint (calculated z-component)
├─── sfs.py: Train SfSNet on synthetic data
├─── skip.py: Train Skipnet on synthetic data
├─── gen_face_mask_celeba.py: Generate face mask for each image in celeba dataset
├─── gen_pseudo_skipnet.py: Generate pseudo dataset using pretrained Skipnet on synthetic dataset
├─── mix_sfs.py: Train SfSNet on mix data
├─── photoface.py: Pre-process the photoface dataset
├─── eval.py: Perform evaluation using photoface dataset
├─── *.sh: Slurm job scripts
```

## SfSNet-Specular
Implmentation of the SfSNet with specularity using our constrained model (calculated z-component constraint) can be found in `/sfsnet-specular`. We explain below the role of each scripts present in that directory.

```
├─── config_sfsnet.json: Configuration file to generate using BFM-2017 model
├─── add-specular.py: Add specularity to the synthetic dataset generated using BFM-2017 model
├─── datasets.py: Data loading and pre-processing methods 
├─── log.py: Log the messages and images using telegram bot
├─── utils.py: Helper functions 
├─── model_two.py: Definition of all the models used. SkipNet and SfSNet with 2nd constraint (calculated z-component)
├─── sfs.py: Train SfSNet on synthetic data
├─── skip.py: Train Skipnet on synthetic data
├─── gen_pseudo_skipnet.py: Generate pseudo dataset using pretrained Skipnet on synthetic dataset
├─── mix_sfs.py: Train SfSNet on mix data
├─── eval.py: Perform evaluation using photoface dataset
├─── *.sh: Slurm job scripts
```
### How to generate BFM synthetic dataset?
Follow the instructions from [parametric-face-image-generator](https://github.com/unibas-gravis/parametric-face-image-generator) and use the [configuration file](https://github.com/SRatna/inverse-rendering/blob/main/sfsnet-specular/config_sfsnet.json) to generate the synthetic dataset using BFM-2017 model. The configuration file can be further tweaked to generate different datasets.

## Requirements
**This implementation is only tested under Ubuntu environment with Nvidia GPUs and CUDA installed.**

## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
conda create --name <env> --file requirements.txt
source activate <env>
```

##

Part of the code in this implementation takes [SfSNet-PyTorch](https://github.com/bhushan23/SfSNet-PyTorch) as a reference.
