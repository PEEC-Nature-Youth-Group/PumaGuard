PumaGuard
=========

This repository contains a machine learning project aimed at classifying images
into two categories: containing a mountain lion or not. The core of this project
is a Jupyter notebook, `Mountain_Lions.ipynb
<https://github.com/nicolasbock/extreme-lion-challenge/blob/main/notebooks/Mountain_Lions.ipynb>`__,
that outlines the process of training a model for this binary classification
task.

Project Overview
----------------

The goal of this project is to accurately classify images based on the presence
of mountain lions. This can have applications in wildlife monitoring, research,
and conservation efforts. The model is trained on a labeled dataset and
validated using a separate set of images.

Getting Started
---------------

The easiest place to start is to run the project in

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/nicolasbock/extreme-lion-challenge/blob/main/Mountain_Lions.ipynb

This approach does not require any local resources.

Alternately, to run the project locally, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook installed.
3. Install required Python packages:

.. code:: bash

   pip install -r requirements.txt

4. Open the `Mountain_Lions.ipynb
   <https://github.com/nicolasbock/extreme-lion-challenge/blob/main/notebooks/Mountain_Lions.ipynb>`__
   notebook and follow the instructions therein.

Model Training
--------------

The notebook walks you through the data preparation, model training, and
validation steps. It utilizes a pre-defined neural network architecture
optimized for image classification tasks. The training process includes data
augmentation techniques to improve model generalization.

Leaderboard
-----------

Below is the leaderboard showing the performance of various model iterations
based on the validation dataset. The models are ranked by validation accuracy.

+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
| Name   | Validation Data Percentage | Random Number Seed | Validation Accuracy | Notes                                               |
+========+============================+====================+=====================+=====================================================+
| PEECYG | 20%                        | 123                | 69%                 | Provided notebook                                   |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
| PEECYG | 20%                        | 123                | 75%                 | Increased resolution to 256x256                     |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
| PEECYG | 20%                        | 123                | 67%                 | With augmentation                                   |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
|PEECYG  | 20%                        | 123                | 80%                 | Pretrained                                          |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
|PEECYG  | 20%                        | 123                | 88%                 | Increased # of images                               |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+
|PEECYG  | 20%                        | 123                | 91%                 | Increased # of images, increased res to 512x512     |
+--------+----------------------------+--------------------+---------------------+-----------------------------------------------------+

References
----------

[1] The images are located in the `data folder
<https://github.com/nicolasbock/extreme-lion-challenge/tree/main/data>`__ in the
repository.
