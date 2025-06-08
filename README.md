# AlignFormer: Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.3.0](https://img.shields.io/badge/pytorch-1.3.0-green.svg?style=plastic)
![CUDA 10.1](https://camo.githubusercontent.com/5e1f2e59c9910aa4426791d95a714f1c90679f5a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f637564612d31302e312d677265656e2e7376673f7374796c653d706c6173746963)

This repository contains the implementation of the following paper:
> **Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera**<br>
> Ruicheng Feng, Chongyi Li, Huaijin Chen, Shuai Li, Jinwei Gu, Chen Change Loy<br>
> Computer Vision and Pattern Recognition (**CVPR**), 2023<br>

[[Paper](https://arxiv.org/abs/2304.06019)]
[[Project Page](https://jnjaby.github.io/projects/AlignFormer/)]


:star: Come and check our poster at `West Building Exhibit Halls ABC 083` on TUE-PM (20/06/2023)!

:star: If you found this project helpful to your projects, please help star this repo. Thanks! :hugs: 


## Update
- **2025.05**: Release training and inference code of PPMUNet.
- **2023.07**: Release training code of AlignFormer.
- **2023.06**: Release inference code of AlignFormer.
- **2023.03**: This repo is created!


## Dependencies and Installation
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Pytorch >= 1.7.1
- [CUDA](https://developer.nvidia.com/cuda-downloads) >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/jnjaby/AlignFormer.git
cd AlignFormer

# (Optional) create new anaconda env
conda create -n alignformer python=3.8 -y
conda activate AlignFormer

# install python dependencies
pip install -r requirements.txt
python setup.py develop
```

## Quick Inference

We provide quick test code with the pretrained model. The testing command assumes using single GPU testing. Please see **[TrainTest.md](docs/TrainTest.md)** if you prefer using `slurm`.

### Download Pre-trained Models:
Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1_2DR2BNp5rdCf-hm60K8fkV0t6KAlDOR?usp=sharing) to the `experiments/pretrained_models` folder.


### Dataset Preparation:
You can also grab the data directly from [GoogleDrive](https://drive.google.com/file/d/1Llvsy9T_yRKM9fYXvA_J2zfFUjYTEUJQ/view?usp=drive_link), unzip and put them into `./datasets`. Note that iamges in `AlignFormer` are the results of our pre-trained model.

#### Dataset structure
```
├── AlignFormer
│   ├── test_sub
│   └── train
├── lq
│   ├── test_sub
│   └── train
├── mask
│   ├── test_sub
│   └── train
└── ref
    ├── test_sub
    └── train
```


### Testing:
1. Modify the paths to dataset and pretrained model in the following yaml files for configuration.

    ```bash
    ./options/test/AlignFormer_test.yml
    ```

1. Run test code for data (AlignFormer).

    ```bash
    python -u basicsr/test.py -opt "options/test/AlignFormer_test.yml" --launcher="none"
    ```

   Check out the results in `./results`.

1. Run test code for PPMUNet.

    ```bash
    python -u basicsr/test.py -opt "options/test/PPMUNet_test.yml" --launcher="none"
    ```

   Check out the results in `./results`.

## Training models:
### Training AlignFormer:

To train an AlignFormer, you will need to train a DAM module first. Then you can merge the pre-trained DAM into AlignFormer and train the whole model.

1. Prepare the datasets. Please refer to [`Dataset Preparation`](#dataset-preparation).

1. Modify config files `./options/train/AlignFormer/DAM_train.yml`.

1. Run training code (*Slurm Training*). Kindly checkout **[TrainTest.md](docs/TrainTest.md)** and use single GPU training, distributed training, or slurm training as per your preference.

   ```bash
   srun -p [partition] --mpi=pmi2 --job-name=DAM --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=2 --kill-on-bad-exit=1 \
   python -u basicsr/train.py -opt "options/train/AlignFormer/DAM_train.yml" --launcher="slurm"
   ```

1. After training the DAM, modify config file of AlignFormer `./options/train/AlignFormer/AlignFormer_train.yml`.

1. Run training code (*Slurm Training*).

   ```bash
   srun -p [partition] --mpi=pmi2 --job-name=DAM --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=2 --kill-on-bad-exit=1 \
   python -u basicsr/train.py -opt "options/train/AlignFormer_train.yml" --launcher="slurm"
   ```

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./experiments` directory.

### Training PPMUNet:

1. Modify config files `./options/train/PPMUNet/PPMUNet_train.yml`.

1. Run training code (*Slurm Training*). Kindly checkout **[TrainTest.md](docs/TrainTest.md)** and use single GPU training, distributed training, or slurm training as per your preference.

   ```bash
   srun -p [partition] --mpi=pmi2 --job-name=PPMUNet --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=2 --kill-on-bad-exit=1 \
   python -u basicsr/train.py -opt "options/train/PPMUNet/PPMUNet_train.yml" --launcher="slurm"
   ```

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./experiments` directory.



### Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @InProceedings{Feng_2023_Generating,
      author    = {Feng, Ruicheng and Li, Chongyi and Chen, Huaijin and Li, Shuai and Gu, Jinwei and Loy, Chen Change},
      title     = {Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2023},
   }
   ```
   ```bibtex
   @InProceedings{Feng_2021_Removing,
      author    = {Feng, Ruicheng and Li, Chongyi and Chen, Huaijin and Li, Shuai and Loy, Chen Change and Gu, Jinwei},
      title     = {Removing Diffraction Image Artifacts in Under-Display Camera via Dynamic Skip Connection Network},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2021},
      pages     = {662-671}
   }
   ```


### License and Acknowledgement

This project is open sourced under [NTU S-Lab License 1.0](https://github.com/jnjaby/AlignFormer/blob/main/LICENSE). Redistribution and use should follow this license.
The code framework is mainly modified from [BasicSR](https://github.com/xinntao/BasicSR). Please refer to the original repo for more usage and documents.


### Contact

If you have any question, please feel free to contact us via `ruicheng002@ntu.edu.sg`.
