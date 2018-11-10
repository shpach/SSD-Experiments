# COMS4995-Final-Project
An SSD network to apply on video, with KCF tracking as smoothing between frames.

The default config works with the following directory structure:

```bash

.
├── README.md
├── config.ini
├── data
│   └── VOC2007
│       ├── test
│       │   ├── Annotations
│       │   ├── ImageSets
│       │   ├── JPEGImages
│       │   ├── SegmentationClass
│       │   └── SegmentationObject
│       └── train
│           ├── Annotations
│           ├── ImageSets
│           ├── JPEGImages
│           ├── SegmentationClass
│           └── SegmentationObject
├── datareader.py
├── detect_track_merge_new.py
├── main.py
├── model.py
└── utils.py

```
