# Nepal Script Text Recognition From Ancient Artifacts: Challenges and Opportunities

### Authors:

#### Swornim Nakarmi, Sarin Sthapit, Sahil Ratna Tuladhar, Arya Shakya, Bal Krishna Bal, Rajani Chulyadyo

---

## Overview

This repository contains the source code and pretrained models for a Nepal Script OCR pipeline. The system detects individual text lines from document images using a YOLO-based line detection model, then recognizes the text in each line using a CRNN model trained with CTC loss.

---

## Project Structure

```
assets/
    charset.csv                 # Character encoding mapping for Nepal and Devanagari script characters
images/                         # Sample input images
models/
    line_detection_model.onnx   # YOLO-based line segmentation model
    ocr_model.onnx              # CRNN-CTC OCR model
utils/
    image_loader.py             # Image loading and line segmentation utilities
    ocr_pipeline.py             # OCR inference and text recognition utilities
model_development_and_training.ipynb    # Model architecture, training, and evaluation
ocr_pipeline.ipynb                      # End-to-end OCR pipeline demo
requirements.txt
README.md
LICENSE
.gitignore
```

---

## Setup

### Prerequisites

- Python 3.9 or higher
- pip
- [Noto Sans Newa Font](https://fonts.google.com/noto/specimen/Noto+Sans+Newa)

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SvarnimN/nepal-script-text-recognition-from-ancient-artifacts.git
cd nepal-script-text-recognition-from-ancient-artifacts
pip install -r requirements.txt
```

---

## Running the OCR Pipeline

The recommended way to run the pipeline is through the provided notebook:

```
ocr_pipeline.ipynb
```

Open it with Jupyter and run the cells in order. The notebook walks through:

1. Loading and resizing an input image
2. Detecting and segmenting individual text lines
3. Running OCR inference on each line

---

## Notebooks

| Notebook                               | Description                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| `model_development_and_training.ipynb` | Model architecture design, dataset preparation, training loop, and evaluation metrics |
| `ocr_pipeline.ipynb`                   | End-to-end inference pipeline: load an image, detect lines, and recognize text        |

---

## Models

Both line detection and OCR models are exported to ONNX format and located in `models/`.

| Model            | File                        | Description                                                             |
| ---------------- | --------------------------- | ----------------------------------------------------------------------- |
| Line Detection   | `line_detection_model.onnx` | Detects and localizes text line bounding boxes in an input image        |
| Text Recognition | `ocr_model.onnx`            | CRNN model that transcribes a cropped line image into Nepal Script text |

---

## Dataset

The complete dataset used in this work consists of various artifacts including handwritten texts, manuscripts, stone, copper, and wood inscriptions in Nepal Script. A subset of this dataset consisting of stone inscriptions has been made publicly available as the [Nepal Script Stone Inscription Dataset](https://www.kaggle.com/dsv/15121632).

---

## Citation
If you use any part of this work in your research, please cite:
```bibtex
@inproceedings{nakarmi-etal-2026-nepal,
  title = {Nepal Script Text Recognition from Ancient Artifacts: Challenges and Opportunities},
  author = {Nakarmi, Swornim and Sthapit, Sarin and Tuladhar, Sahil Ratna and Shakya, Arya and Bal, Bal Krishna and Chulyadyo, Rajani},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {3161--3170},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/427yvm8biop7},
  abstract = {Nepal Script, a script of significant linguistic, historical, and cultural importance, can be found in ancient artifacts in Nepal. As this script has faced a decline in use, it is considered among endangered scripts at present. For its revival and preservation, it is important to digitize ancient artifacts written in Nepal Script and create an accessible digital dataset. Among such artifacts are stone inscriptions, and manuscripts, from which we attempt to recognize texts using Artificial Intelligence techniques. This paper presents our approach of preparing a dataset through an extensive data acquisition method, and developing a system that recognizes Nepal Script texts from images. Our system combines the YOLOv8 algorithm with Convolutional Recurrent Neural Network architecture and Connectionist Temporal Classification loss. Our dataset consists of 5,219 text line images from ancient stone inscriptions, manuscripts, and modern handwritten and typed documents. Utilizing an augmented dataset of 41,752 samples, our system achieved 12.61% Character Error Rate. Despite the small training dataset, our model successfully predicted texts in not only new stone inscriptions and manuscripts but also wooden and copper plate inscriptions. We expect our contributions will encourage further research on Nepal Script and other Nepalese scripts.}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
