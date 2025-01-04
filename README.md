# PCB_defect_detection
## Project Overview

This repository contains the implementation for a PCB (Printed Circuit Board) defect detection project developed as part of the **Quantitative Engineering and Analysis 1** course. The project employs linear algebra techniques and machine learning to detect defects on PCBs after fabrication. Our goal is to establish a robust framework capable of identifying six classes of defects effectively.

### Key Features

- **Defect Detection Classes**: Missing Hole, Mouse Bite, Open Circuit, Short, Spur, Spurious Copper.
- **Dataset**: Uses the [PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) from Kaggle.
- **Machine Learning Model**: Built using YOLOv8 (You Only Look Once).
- **Accuracy**: Achieved 92% accuracy on the test dataset.

### A visualization of the project's achievements
[Download the QEA project PDF](QEA%20project.pdf)
---

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python (3.8 or later)
- PyTorch
- Kaggle API
- YOLOX framework
- Additional utilities (listed in `requirements.txt`)

### Installation

1. Clone this repository:
    
    ```
    git clone https://github.com/Bugates/PCB_defect_detection.git
    cd PCB_defect_detection
    ```
    
2. Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    

### Dataset Setup

1. Download the dataset using Kaggle API:
    
    ```
    kaggle datasets download -d akhatova/pcb-defects
    ```
    
2. Extract the dataset:
    
    ```
    unzip pcb-defects.zip -d PCB_DATASET
    ```
    
3. Update the dataset path in the script if needed.

---

## Project Workflow

### 1. Data Preparation

- Images and annotations are parsed and organized into training, validation, and testing splits.
- Annotations are converted to YOLO-compatible `.txt` format with bounding boxes and labels.

### 2. Model Training

- The YOLOv8n model is trained on the processed dataset using 1000 epochs.
- The training script computes metrics such as Precision, Recall, and mAP.

### 3. Evaluation

- The trained model is evaluated on the validation and test datasets to compute key metrics and overall accuracy.

---

## Results

- **Precision**: 0.91
- **Recall**: 0.90
- **mAP@50**: 0.918
- **mAP@50-95**: 0.88
---

## How to Use

### Training the Model

Run the training script to train the YOLOv8 model:

```
python train.py
```

### Evaluating the Model

Run the evaluation script to assess model performance:

```
python evaluate.py
```

### Making Predictions

Use the trained model to predict defects on new images:

```
python predict.py --image <image_path>
```

---

## Repository Structure

```
.
├── data/                 # Dataset and annotations
├── models/               # Pretrained and trained model weights
├── scripts/              # Python scripts for training and evaluation
├── poster/               # Visualizations of our work
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Co-author

- **[Navya Tiwari] (https://github.com/Bugates)**

- **[Anika Mahesh] (https://github.com/AnikaMahesh)** 

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- The [Kaggle PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects).
- [You Only Look Once: Unified, Real-Time Object Detection (John Redmond et al., 2015 )](https://arxiv.org/pdf/1506.02640)
- YOLOv8 framework for object detection.
- **Quantitative Engineering and Analysis 1** course for inspiring this project.
