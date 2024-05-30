# Mountain Lion Classification Challenge

<p align="center">
  <img src="/pics/MLProj.png" width="600">
</p>


This repository contains a machine learning project aimed at classifying images into two categories: containing a mountain lion or not. The core of this project is a Jupyter notebook that outlines the process of training a model for this binary classification task.

## Project Overview

The goal of this project is to accurately classify images based on the presence of mountain lions. This can have applications in wildlife monitoring, research, and conservation efforts. The model is trained on a labeled dataset and validated using a separate set of images.

## Getting Started

To run the project in [Google Colab](https://colab.research.google.com/), click:


<a target="_blank" href="https://colab.research.google.com/github/je-santos/lions/blob/main/classifying_lions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

<br>
<br>
<br>



Alternately, to run the project locally, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook installed, or use [Google Colab](https://colab.research.google.com/) for an online environment.
3. Install required Python packages:

```bash
pip install -r requirements.txt
```

4. Open the `classifiying_lions.ipynb` notebook and follow the instructions within.

## Model Training

The notebook walks you through the data preparation, model training, and validation steps. It utilizes a pre-defined neural network architecture optimized for image classification tasks. The training process includes data augmentation techniques to improve model generalization.


## Leaderboard

Below is the leaderboard showing the performance of various model iterations based on the validation dataset. The models are ranked by validation accuracy.

| Name      | Validation Data Percentage | Random Number Seed | Validation Accuracy | Notes |
|-----------|----------------------------|--------------------|---------------------|-------|
| Javier    | 20%                        | 123                | 70%                 | Provided notebook      |
| Javier    | 20%                        | 123                | 74%                 | Followed Hari's suggestion and trained with 256x256      |
     

## References
[1] The images are located in Hari's [Google Drive](https://drive.google.com/drive/folders/1g8MZmL-rDcTXpUrVBf-UEgudM2NHgKQJ).
