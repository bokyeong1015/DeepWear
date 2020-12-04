# DeepWear

Torch7 implementation for paper

* "[Style-Controlled Synthesis of Clothing Segments for Fashion Image Manipulation](https://ieeexplore.ieee.org/document/8770290/references#references)," IEEE Transactions on Multimedia (vol. 22, no. 2, Feb 2020)

Given a clothing segment of a person image and a desired style, our method synthesizes a new clothing item. This item is superimposed on the person image for virtual try-on.

![fig_overview](fig_overview.png)


## Important Notice
Because of copyright issues, we cannot provide our experimental data (i.e., preprocessed version of the datasets in [1–5]).
* Please contact me via e-mail (bokyeong1015 at gmail dot com) if you want to know data download & preprocessing steps.

## Setup
* The following environments were tested (other better GPU/cuda may work, but untested)
  - Ubuntu 16.04 or 14.04
  - GTX Titan X or GTX 1080
  - CUDA 8.0 or 7.5
  - cuDNN 5.1

* Install [Torch7](http://torch.ch/docs/getting-started.html#installing-torch) and the following torch packages:
```
bash ./download_packages.sh
```

## Quick Guide
~~(1) Download FsSyn+LkBk dataset with the following script~~

(2) Download resol2_vggL1 model via [this GoogleDrive link](https://drive.google.com/file/d/185-FzcNrwcjDSROfVrBpPMrPK3fCEePO/view?usp=sharing). Set the path to be `models_trained/resol2_vggL1/model_minValidLoss.t7`

(3) Test the model:
 ```
bash ./test_LkBk.sh
```



## Dataset Download
We had planned to provide full scripts to download our experimental data but recognized some copyright issues on distributing the datasets in [1–5]. Please check the above notice and feel free to contact me.
#### 1. "FsSyn+LkBk Dataset" for Testing (~340MB)
  - We selected person images from FsSyn [1] and top-clothing product images from LkBk [2].
    
#### 2. "Unified Segment Dataset" for Training & Testing (~5.3GB)
  - We extracted clothing segments from CCP [3], CFPD [4], and FS [5] datasets, and unified them.



## Trained Models
* Download the pre-trained models using the following GoogleDrive links. 
  - [resol2_vggL1](https://drive.google.com/file/d/185-FzcNrwcjDSROfVrBpPMrPK3fCEePO/view?usp=sharing) (~1.2GB): **FINAL** model, 384×128 input size, {L1-pixel + VGG-feature} training loss
  - [resol2_onlyL1](https://drive.google.com/file/d/1zuS0V1JrzL1NytXTVzoRDteKEtdiYJ-v/view?usp=sharing) (~1.2GB): 384×128 input size, only L1-pixel loss
  - [resol1_vggL1](https://drive.google.com/file/d/1KllKR-z55OWL9p7XDoFUxARnesCyof_s/view?usp=sharing) (~840MB): 192×64 input size, {L1-pixel + VGG-feature} loss
  - [resol1_onlyL1](https://drive.google.com/file/d/1F-sRp7emZkehxcFJNS7ub4I0xrpHD2eZ/view?usp=sharing) (~840MB): 192×64 input size, only L1-pixel loss
* Put the downloaded models in `models_trained/MODEL_NAME`. MODEL_NAME is specified above (e.g., resol2_vggL1). The example paths become `models_trained/resol2_vggL1/model_minValidLoss.t7`, `models_trained/resol1_vggL1/model_minValidLoss.t7`, etc.

## Test Code
1. (Optional) Modify lines 17 and 18 in `test.lua` depending on the models you want to test (default: resol2_vggL1).
2. Test the model on clothing product images (LkBk) OR segments (UnifSegm) with the following script. Here, person images from FsSyn are used.
 ```
bash ./test_LkBk.sh
bash ./test_UnifSegm.sh
```

## Training Code
1. Download the VGG-19 network (for computing training loss) from [this GoogleDrive link](https://drive.google.com/file/d/1YhmRw_FrKKopCxP8goRR20syUEDmVRNy/view?usp=sharing) (~2.2GB). Then, put the downloaded `VGG_ILSVRC_19_layers_nn.t7` in `src_train/src_percepLoss`.
2. Train the model from scratch for 384×128 (resol2) OR 192×64 (resol1) input with the following script. The learning rate schedule was manually set. 
```
bash ./train_resol2.sh
bash ./train_resol1.sh
```

## References
* [1. [Fashion Synthesis (FsSyn)](https://github.com/zhusz/ICCV17-fashionGAN)] S. Zhu et al., “Be Your Own Prada: Fashion Synthesis with Structural Coherence,” in ICCV’17
* [2. [LookBook (LkBk)](https://dgyoo.github.io/)] D. Yoo et al., “Pixel-level domain transfer,” in ECCV’16
* [3. [Clothing Co-Parsing (CCP)](https://github.com/bearpaw/clothing-co-parsing)] W. Yang et al., “Clothing co-parsing by joint image segmentation and labeling,” in CVPR’14
* [4. [Colorful Fashion Parsing (CFPD)](https://github.com/hrsma2i/dataset-CFPD)] S. Liu et al., “Fashion parsing with weak color-category labels,” IEEE Transactions on Multimedia, 2014
* [5. [Fashionista (FS)](http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/)] K. Yamaguchi et al., “Parsing clothing in fashion photographs,” in CVPR’12

## Acknowledgment
* We thank the authors of [1–5] for providing their datasets.
* The codes for VGG-feature loss borrow heavily from [fast-neural-style](https://github.com/jcjohnson/fast-neural-style/tree/master/fast_neural_style). The network architectures were modified from [pix2pix](https://github.com/phillipi/pix2pix/blob/master/models.lua). The VITON-stage1 masks were computed using [VITON](https://github.com/xthan/VITON). We thank them for open-sourcing their projects.


## Citation
If you plan to use our codes and datasets, please consider citing our paper:
```
@ARTICLE{8770290,
  author={B. {Kim} and G. {Kim} and S. {Lee}},
  journal={IEEE Transactions on Multimedia}, 
  title={Style-Controlled Synthesis of Clothing Segments for Fashion Image Manipulation}, 
  year={2020},
  volume={22},
  number={2},
  pages={298-310},
  doi={10.1109/TMM.2019.2929000}}
```  