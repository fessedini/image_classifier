# Image Classification for Refund Department (Batch Automation)

## Project Overview

This project automates the classification of returned clothing items using a deep learning model. The system processes product images, predicts both **main** and **subcategories**, and runs automatically in **nightly batch jobs** using a **Flask API** and **cron scheduler**. Additionally this project aims to reduce the workforce and costs for manually sorting items.

---

## Table of Contents

- [Use Case](#use-case)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Usage](#usage)
- [Cronjob for Automation](#cronjob-for-automation)

---

## Use Case

As the online platform for sustainable products grows, so does the number of returns. Manual sorting is no longer scalable. This project solves that problem by:

- Classifying images into categories automatically
- Running the prediction pipeline daily overnight
- Reducing manual workload and improving efficiency

---

## Dataset

This project uses a dataset of clothing item images structured into folders by category and subcategory. The images are required to train and evaluate the model.

Due to file size limitations, the images in the **`data/train`**, **`data/test`**, and **`data/raw/outfit_items_dataset`** folders are not included in this repository.

To reproduce the project, please download a similar or identical dataset and organize it as follows:

- **data/raw/outfit_items_dataset** - Original data organized into main categories (*bottomwear*, *footwear*, and *upperwear*) with the corresponding sub categories (*pants* and *shorts*, *heels* and *sneakers*, *jacket* and *shirt*)
- **data/train** - Training data structured by subcategory (e.g. *0_pants*, *1_shorts*, ...)
- **data/test** - Testing data structured by subcategory

This project used the data from an open platform named **Kaggle**. You can download the dataset [**here**](https://www.kaggle.com/datasets/kritanjalijain/outfititems)

**Note:** Make sure to maintain the exact folder structure to ensure correct label mapping via `ImageFolder`.

---

## Project Structure

The projects structure uses a clear and modular folder structure to separate raw data from test and training data, processed and incoming images, scripts, model files, logs, output files and a bash script.

This structure did not come about by chance, but was consciously chosen. In addition to the above mentioned separation of concerns, the structure is also important when it comes to other aspects:

- Enhanced **reproducibility** as the model can be retrained from raw data or any prdecition errors can be debuged from logs
- Scripts can rely on this structure to find files or save the model without hardcoding
- Enhanced **scalability** as new categories or subcategories can be added by just manually updating the folder structure and retrain
- Enabling consistent and reproducible mapping between folder names and class indices (**ImageFolder**)

```
image_classifier/
├── data/
│   ├── incoming_images/              # New images to classify
│   ├── processed/                    # Processed images after prediction
│   ├── raw/
│   │   └── outfit_items_dataset/
│   │       ├── bottomwear/
│   │       │   ├── pants/
│   │       │   └── shorts/
│   │       ├── footwear/
│   │       │   ├── heels/
│   │       │   └── sneakers/
│   │       └── upperwear/
│   │           ├── jacket/
│   │           └── shirt/
│   ├── test/
│   │   ├── 0_pants/
│   │   ├── 1_shorts/
│   │   ├── 2_heels/
│   │   ├── 3_sneakers/
│   │   ├── 4_jacket/
│   │   └── 5_shirt/
│   └── train/
│       ├── 0_pants/
│       ├── 1_shorts/
│       ├── 2_heels/
│       ├── 3_sneakers/
│       ├── 4_jacket/
│       └── 5_shirt/
├── logs/
│   └── cronlog.log                  # Cron job logs
├── model/
│   └── model.pth                    # Saved trained model
├── scripts/
│   ├── app.py                       # Flask API for predictions
│   ├── batch_processing.py          # Script for batch processing images
│   └── train_model.py               # Model architecture, training and evaluation
├── full_batch.sh                    # Bash script for nightly batch runs
├── predictions.csv                  # Output file with predictions
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── image_classifier.code-workspace  # VS Code workspace file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fessedini/image_classifier.git
cd image_classifier
```

### 2. Create and Activate the Conda Environment

```bash
conda create -n image_classifier python=3.11
conda activate image_classifier
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages, including:
- **torch and torchvision** – for building and running the deep learning model
- **flask** – for serving the prediction API
- **pandas** – for saving batch results as CSV
- **requests** – for sending image data to the API
- **pillow** – for image loading and preprocessing (used by app.py)

---

## How it works

### Model Architecture

The classification model is based on a custom ResNet18 model with two output heads. This allows the model to simultaneously predict two labels for each image, the **main** and the **sub category**.

- **main categories**: bottomwear, footwear, upperwear
- **sub categories**: pants, shorts, heels, sneakers, jacket, shirt

The model uses a pretrained ResNet18 backbone and includes two parallel output heads to handle both classification tasks. All images are resized to 224x224 pixels and normalized using standard ImageNet parameters.

---

### Training & Testing

The model can be trained from scratch or retrained using the existing dataset by running the following script:

```bash
python scripts/train_model.py
```

This will:

- Apply data augmentation and normalization to all images.
- Use the folder names/numbers to generate labels via ImageFolder.
- Train the model for 15 epochs on the training set
- Evaluate performance on the test set
- Save the trained model as model/model.pth.

After training, you will see the accuracy for both:

- **Main category prediction**
- **Subcategory prediction**

---

## Usage

### Start Flask API (manually)

Start the Flask API manually in the terminal and launch the Flask server locally on http://127.0.0.1:5000 by running:

```bash
python scripts/app.py
```

### Run batch processing (manually)

Before the batch processing is started manually, it should be pointed out that it is advisable to open a second terminal beforehand and run the batch_processing script there.

```bash
python scripts/batch_processing.py
```

This script will:

- Scan for new images in the incoming_images folder
- Send the images to the running Flask API (if pictures are present)
- Save predictions to predictions.csv
- Move processed images to the processed folder

### Cronjob for Automation

To classify images automatically and running the prediction pipeline daily overnight this project relies on cronjobs.

Before the cron job is executed, it is essential to make default settings as cronjob can be executed in two ways (**nano** and **vim**).

The following command will set the nano editor (as it is simpler and beginner-friendly):

```bash
export VISUAL=nano
export EDITOR=nano
```
To automate daily batch predictions at **2:00 AM**, open the crontab generator using:

```bash
crontab -e
```
and add or create the following line:

```bash
0 2 * * * /home/youruser/project/full_batch.sh >> /home/youruser/project/logs/cronlog.log 2>&1
```
As you can see in the code, the cronjob runs the bash-script **full_batch.sh**. It automates the entire workflow:

- Activating the conda environment
- Starting the Flask API in the background
- Runs the batch_processing.py file
- Stops the Flask API
- Logs everything to a log file

**Important:** Please note that the automated cron job will only run if the machine is turned on and not in sleep or hibernation mode at the scheduled time (e.g., 2:00 AM).