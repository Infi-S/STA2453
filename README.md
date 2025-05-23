# Fish Classification with Sound

This repository works on the Fish Classification with Sound project which is based on the dataset at [WidebandPingFest/FishTetherExperiment](https://github.com/WidebandPingFest/FishTetherExperiment).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)
6. [License](#license)

---

## Introduction

This project leverages sound data to perform fish classification using both traditional machine learning and deep learning approaches. It includes data cleaning, exploratory data analysis (EDA), and model training scripts along with reproducible notebooks.

---

## Project Structure

- **Data**:  
  Due to file size limitations, processed data are not uploaded to GitHub. Instead, they can be found at:  
  [Processed Data](https://mega.nz/folder/8JYnwQKA#ODm-ycelxWaDrcM35pjqog) <br>
  Or, the sample_data.csv can provide an idea on how this dataset looks like.

- **Output**:  
  This folder includes both text and image outputs that are finally used in each report. Complete outputs can be found and reproduced by the notebooks in the **Script** folder.

- **Script**:  
  This folder contains Python scripts and Jupyter notebooks used for data cleaning, exploratory data analysis, progress reporting and model constructions.

- **Model**:  
  This folder includes scikit-learn and PyTorch models trained and saved during research.

---

## Installation

Before running the project, ensure you have Python 3.9+ installed. Then follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Infi-S/STA2453.git
cd STA2453
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
For the complete list of required packages, please refer to the requirements.txt file, or directly install all required packages using:
```bash
pip install -r requirements.txt
```

---

## Usage
Download processed data from this [Mega link](https://mega.nz/folder/8JYnwQKA#ODm-ycelxWaDrcM35pjqog) and place in the Data folder.

Open notebooks in the Sciprt folder using Jupyter Lab or Notebook:
Run the notebooks to reproduce data cleaning, EDA, and modeling steps.
> For 0. Data Rephrasing.ipynb, 1. EDA.ipynb and 2. Progress Report.ipynb, they need to be placed in the root folder to run properly

---

## Citation
If you use this project, please cite this repository and the original dataset:
[WidebandPingFest/FishTetherExperiment](https://github.com/WidebandPingFest/FishTetherExperiment).

## License
This project is for academic research and educational use under the MIT License. See [LICENSE](./LICENSE) for details.



