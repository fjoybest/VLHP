# VLHP: Learning Discriminative Vision-Language Hybrid Prototypes for Weakly Supervised Semantic Segmentation

🎉 **VLHP has been accepted by ACM Multimedia 2025 (ACMMM25)!**

📄 **Paper:** [https://dl.acm.org/doi/abs/10.1145/3746027.37548931](https://dl.acm.org/doi/abs/10.1145/3746027.37548931)
---

## 🛠️ Environment Setup

Please configure the environment using requirements.txt`

### 1. Create and Activate Conda Environment

```bash
conda create -n vlhp python=3.8 -y
conda activate vlhp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

This repository supports both **VOC** and **COCO** datasets.

### 1. Download VOC / COCO Datasets

For VOC preparation, please follow the same procedure used in **ToCo**:

🔗 [https://github.com/rulixiang/ToCo](https://github.com/rulixiang/ToCo)

### 2. Place VOC Dataset in your datapath. 

The VOC dataset should be organized as:

```
yourdatapath/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        ├── ImageSets/
        └── ...
```

---

## 🚀 Inference on PASCAL VOC 2012

```bash
python infer_seg_voc.py
```

---

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{fang2025vlhp,
  title={VLHP: Learning Discriminative Vision-Language Hybrid Prototypes for Weakly Supervised Semantic Segmentation},
  author={Fang, Jingyuan and Ning, Yang and Nie, Xiushan and Liu, Xinfeng and Cheng, Zhiyong},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={2939--2948},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This repository is built upon and inspired by ToCo and WeCLIP, which are both excellent works in weakly supervised semantic segmentation.


