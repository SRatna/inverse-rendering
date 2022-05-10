# Supervised and Self-Supervised Learning for Inverse Rendering of Human Faces

In this work, we aim to improve upon a state of the art inverse rendering method called SfSNet. We constrain SfSNet such that it always predicts unit normals and also introduce specularity in its rendering function with an aim to make it robust even with faces that contains specular highlights. Although its normal estimation gets improved than the original SfSNet work, specular highlights still persist on predicted albedo when tested on real faces with specular highlights. In spite of the fact that the rendering function with added specularity have not worked perfectly, it opens up new possibility of research on similar direction to solve this problem of specularity. 

## SfSNet-Constraints
Implmentation of the SfSNet with constraints can be found in `\sfsnet-constraints`. We explain below the role of each scripts present in that directory.

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
├─── gen_pseudo_sfsnet.py: Generate pseudo dataset using pretrained SfSNet on synthetic dataset
├─── gen_pseudo_skipnet.py: Generate pseudo dataset using pretrained Skipnet on synthetic dataset
├─── mix_sfs.py: Train SfSNet on mix data
├─── photoface.py: Pre-process the photoface dataset
├─── eval.py: Perform evaluation using photoface dataset
├─── *.sh: Slurm job scripts
```

## SfSNet-Specular
Implmentation of the SfSNet with specularity using our constrained model (calculated z-component constraint) can be found in `\sfsnet-specular`. We explain below the role of each scripts present in that directory.

```
├─── datasets.py: Data loading and pre-processing methods 
├─── log.py: Log the messages and images using telegram bot
├─── utils.py: Helper functions 
├─── model_two.py: Definition of all the models used. SkipNet and SfSNet with 2nd constraint (calculated z-component)
├─── add-specular.py: Add specularity to the synthetic dataset generated using BFM-2017 model
├─── sfs.py: Train SfSNet on synthetic data
├─── skip.py: Train Skipnet on synthetic data
├─── gen_pseudo_sfsnet.py: Generate pseudo dataset using pretrained SfSNet on synthetic dataset
├─── gen_pseudo_skipnet.py: Generate pseudo dataset using pretrained Skipnet on synthetic dataset
├─── mix_sfs.py: Train SfSNet on mix data
├─── eval.py: Perform evaluation using photoface dataset
├─── *.sh: Slurm job scripts
```