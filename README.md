# Document Segmentation

This project provides a **document segmentation model** that recognizes key elements in scientific documents, including **titles, text, figures, tables, and lists**. The model was trained using a subset of the **PubLayNet** dataset.

---

## Dataset

The model was trained on a subset of **PubLayNet**, which is a very large dataset (~100 GB) of scientific document images. For this project, we used the subset provided by Kaggle:

> “The original dataset is very huge (~100 GB) and is divided into 7 parts by the original publisher. You can find the whole dataset over [here](https://www.kaggle.com/datasets/). In this dataset, you will find part 0/7 of the original dataset which is 13 GB in size. You will also find the annotations of the dataset in COCO format.”

The dataset contains:

- Images of scientific documents.
- Bounding box annotations for the following 5 classes:
  - **title**
  - **text**
  - **figure**
  - **table**
  - **list**

For this project, we used:

- ~47,000 images
- ~3 million objects

Due to computational limits, **only a subset of 5,000 images** was used for training.

---

## Model

We used **YOLO (You Only Look Once)** for object detection, specifically the **nano model**, which is the smallest and fastest version.  

The model predicts bounding boxes around the different document elements (titles, text, figures, tables, lists), making it suitable for automated document layout analysis.

---

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Running the model

```bash
python -m src.inference --image "data/images/test/PMC1215499_00001.jpg" --output "results"
```

## Example

<img width="610" height="794" alt="image" src="https://github.com/user-attachments/assets/29c1a2bd-1de8-4851-aea6-2e9494a7290e" />


