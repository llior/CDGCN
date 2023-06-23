# Community Detection Graph Convolutional Network (CDGCN)

## Paper

[Community Detection Graph Convolutional Network for Overlap-Aware Speaker Diarization](https://ieeexplore.ieee.org/abstract/document/10095143/), ICASSP 2023

## Requirements

Python

[PyTorch](https://pytorch.org/)

[mmcv](https://github.com/open-mmlab/mmcv)

[faiss](https://github.com/facebookresearch/faiss)

## Build environment

Install dependencies (example)
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge faiss
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install tqdm igraph leidenalg mmdet==2.25.0
```

## Datasets

The datasets including OSD label and embeddings are mentioned in [DATASET.md](https://github.com/llior/xv_dataset/DATASET.md)

## Pretrained Model

[CDGCN (Voxceleb2 Train)](https://drive.google.com/file/d/11O4oqoqU5jRmKKCueEaglIOYlK5xS3On/view?usp=sharing)

## Run

Step1: Clone code

```shell
git clone https://github.com/llior/CDGCN.git 
cd CDGCN/
```

### With pretrained model

Step2.1: Prepare pretrained model

```shell
gdown https://drive.google.com/uc?id=1yBMhbx-UD82W3Izo5-uqxTUCZH3onmXs
unzip xv_data.zip 
```

Step2.2: Perform CDGCN

```shell
./cdgcn_run.sh --stage "Test_1st"
./cdgcn_run.sh --stage "Test_2nd"
```

Step2.3: Evaluate the result

```shell
./cdgcn_run.sh --stage "Evaluation" --ref [Reference File Path]
```

### Train CDGCN (Optional)

Step 3.1: Download training datasets

```shell
gdown https://drive.google.com/uc?id=11O4oqoqU5jRmKKCueEaglIOYlK5xS3On
uzip -d xv_data/resnet34se training_data.zip 
```

Step 3.2: Training model with x-vectors of VoxCeleb2 dataset

```shell
./cdgcn_run.sh --stage "Train"
```

## Result on DIHARDIII (DER %)

|                     | DEV(Core) | DEV(Full) | EVAL(Core) | EVAL(Full) |
| :------------------ | :-------: | :-------: | :--------: | :--------: |
| AHC                 |   19.31   |   19.94   |   19.27    |   18.90    |
| K-means             |   25.34   |   23.05   |   23.71    |   21.24    |
| NME-SC              |   18.56   |   17.89   |   17.98    |   16.81    |
| CDGCN w/o Graph-OSD |   17.10   |   16.43   |   16.50    |   15.38    |
| CDGCN               |   15.40   |   13.67   |   15.97    |   13.72    |

## Citation

Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{wang2023community,
  title={Community Detection Graph Convolutional Network for Overlap-Aware Speaker Diarization},
  author={Wang, Jie and Chen, Zhicong and Zhou, Haodong and Li, Lin and Hong, Qingyang},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```
