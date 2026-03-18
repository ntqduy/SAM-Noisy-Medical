---
license: apache-2.0
tags:
- medical
- segmentation
- sam3
- lora
- vision
pipeline_tag: image-segmentation
---

# MedSAM3 v1: Delving into Segment Anything with Medical Concepts (LoRA Weights)

This repository contains the **v1 LoRA weights** for **MedSAM3**.

## 📊 Model & Dataset Statistics

We constructed a large-scale dataset uniformly sampled to ensure diversity and robustness. The model covers **diverse medical modalities**:

* **Radiology:** CT, MRI, PET, X-ray
* **Optical/Microscopic:** Microscopy, Histopathology, Dermoscopy, OCT, Cell
* **Video/Procedure:** Ultrasound, Endoscopy, Surgery video

**Dataset Scale:**
* **658,094** Images
* **2,863,974** Instance Annotations
* **330** Unique Medical Text IDs (Concepts)

## ⚠️ Usage Instructions

**These are not standalone weights.** To use this model, you must load these LoRA weights in combination with the base **SAM3** model. Please refer to our official GitHub repository for detailed instructions on environment setup, weight loading, and inference.

* **GitHub Repository:** [MedSAM3 on GitHub](https://github.com/Joey-S-Liu/MedSAM3)
* **Paper:** [ArXiv](https://arxiv.org/abs/2511.19046)

## 🖊️ Citation

```bibtex
@misc{liu2025medsam3delvingsegmentmedical,
      title={MedSAM3: Delving into Segment Anything with Medical Concepts}, 
      author={Anglin Liu and Rundong Xue and Xu R. Cao and Yifan Shen and Yi Lu and Xiang Li and Qianqian Chen and Jintai Chen},
      year={2025},
      eprint={2511.19046},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2511.19046](https://arxiv.org/abs/2511.19046)}, 
}