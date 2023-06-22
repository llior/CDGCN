# Dataset Preparation

## Data Directory

The data directory is constucted as follows:

```
.
xv_data
├── dihard_2020_dev.list
├── dihard_2020_eval.list
├── osd
│   └── model
├── ref_dev.rttm
├── ref_eval.rttm
└── resnet34se
    ├── features
    ├── knns
    ├── labels
    ├── subsegments
    └── work_dir
```

You can download the x-vectors of dataset.

Training set VoxCeleb2: [GoogleDrive](https://drive.google.com/uc?id=11O4oqoqU5jRmKKCueEaglIOYlK5xS3On)

Development set and Evaluation set DIHARD III: [GoogleDrive](https://drive.google.com/uc?id=1yBMhbx-UD82W3Izo5-uqxTUCZH3onmXs)

## Supported datasets


### Training set

[VoxCeleb2 ](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)contains over 1 million utterances for 6,112 celebrities, extracted from videos uploaded to YouTube.

### Development set and Evaluation set

The relevant data releases from [LDC](https://www.ldc.upenn.edu/):
```
- DIHARD III development set (LDC2020E12)
- DIHARD III evaluation set (LDC2021E02)
```