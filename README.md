# Cats and Dogs Detection

This repository contains the implementation of a custom similarity metric for bounding box evaluation, used in a YOLO-based object detection model for classifying cats and dogs. The custom metric improves upon IoU, GIoU, DIoU, and CIoU by incorporating multiple geometric factors such as center distance, aspect ratio, and size similarity.

## 🚀 YOLO Setup & Architecture

We use **YOLOv5** for object detection. The setup involves:
- **Pretrained YOLOv5 model**: Fine-tuned on a custom dataset containing [images of cats and dogs](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection)
- **Custom metric**: Introduced for better bounding box evaluation.
- **Training framework**: PyTorch-based implementation using Kaggle notebooks.

## 📏 Custom Metric Definition

Our custom metric evaluates the similarity between two bounding boxes using:

```python
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def customized_similarity(box1, box2):
    def center_distance(b1, b2):
        c1 = np.array([b1[0] + b1[2] / 2, b1[1] + b1[3] / 2])
        c2 = np.array([b2[0] + b2[2] / 2, b2[1] + b2[3] / 2])
        dist = np.linalg.norm(c1 - c2)
        max_dim = max(b1[2], b1[3], b2[2], b2[3])  # Normalize by max dimension
        return dist / max_dim  

    def aspect_ratio_similarity(b1, b2):
        ar1 = b1[2] / b1[3]  # width/height
        ar2 = b2[2] / b2[3]
        return min(ar1, ar2) / max(ar1, ar2)

    def size_similarity(b1, b2):
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        return min(area1, area2) / max(area1, area2)

    iou = compute_iou(box1, box2)
    d = center_distance(box1, box2)
    ar_sim = aspect_ratio_similarity(box1, box2)
    size_sim = size_similarity(box1, box2)

    similarity = (0.25*iou) + (0.25*(1 - d)) + (0.25*ar_sim) + (0.25*size_sim)
    return similarity
```

### ✅ Why This Metric?
- 📌 Combines IoU, center distance, aspect ratio, and size similarity.
- 📌 Works well even when IoU is zero.

## 📚 Training & Evaluation Instructions

### 🔧 **1. Training the Model**

Clone the repository and install dependencies:
```sh
git clone https://github.com/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Train YOLOv5 on the dataset:
```sh
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --patience 5
```

### 🧐 **2. Evaluating the Model**

Use the validation dataset for evaluation:
```sh
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4
```

## 📌 Kaggle Notebook
You can find the full implementation in the Kaggle notebook:
[📌 Rad Cats & Dogs Object Detection](https://www.kaggle.com/code/charukabandara/cats-and-dogs-object-detection)



