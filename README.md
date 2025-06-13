# Energy Prediction Model for research
This repository contains the Energy Prediction Model program in KICT R&D project call ERC, designed to predict energy consumption based on a pre-trained deep learning model. You can download [the trained model](https://drive.google.com/drive/folders/1-wKivitvpK1PJbNFvKjvy77hqg7R1siX?usp=sharing) for energy prediction.
It supports both a command-line interface (CLI) and a web-based interface for predictions.

<img src="https://github.com/mac999/building_energy_prediction_model/blob/main/UI.png" height="500" />

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Files Description](#files-description)
- [config.json Structure](#configjson-structure)
- [Input CSV File Requirements](#input-csv-file-requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [CLI Mode](#cli-mode)
  - [Web UI Mode (Gradio)](#web-ui-mode-gradio)
- [Prediction Output Format](#prediction-output-format)
- [Prediction Process Summary](#prediction-process-summary)
- [Error Handling and Notes](#error-handling-and-notes)
- [Directory Structure](#directory-structure)
- [Author](#author)
- [Created Date](#created-date)

## Overview

This program acts as an energy prediction module. It utilizes a `config.json` file (which defines the model's architecture, input variables, and normalization standards) and a trained model parameter file (`.pth`) to predict energy consumption from input CSV data. The prediction results are output as CSV files and can be used via either a command-line interface (CLI) or a web interface. The `config` file for the generated model's deep learning parameters is designed to be variable, supporting various training model types.

## Features

* **Flexible Model Configuration**: Model architecture and parameters are defined in a `config.json` file, allowing for easy modification and support for various model types.
* **CLI Support**: Run predictions directly from the command line with specified input and output folders.
* **Web Interface (Gradio)**: Provides a user-friendly browser-based UI for uploading model files and input CSVs, and downloading prediction results as a ZIP file.
* **Data Preprocessing**: Handles missing values and offers an optional energy consumption filter based on a minimum threshold.
* **MinMax Normalization**: Applies MinMax scaling for consistent model input.
* **Output**: Adds a `Predicted_EnergyConsumption` column to the input CSVs, saving them to a specified output directory.

## Files Description

| Filename           | Description                                                 |
| :----------------- | :---------------------------------------------------------- |
| `dce_prediction_mode.py` | The main Python script for energy prediction.               |
| `config.json`      | Model configuration, input variables, and normalization settings. |
| [`model.pth`](https://drive.google.com/drive/folders/1-wKivitvpK1PJbNFvKjvy77hqg7R1siX?usp=sharing)        | The PyTorch file storing the trained model's parameters. |
| `input/*.csv`      | Input data files for prediction.                  |
| `output/*.csv`     | Output files where prediction results will be saved (automatically generated). |

## config.json Structure

Below is an example structure of the `config.json` file:

```json
{
  "model": "mlp",
  "input_cols": ["Temperature", "Humidity", "Insolation", "week", "TempDiff"],
  "filter_energy": 120.0,
  "dataset_ratio": 0.5,
  "hidden_size": [24, 32, 64, 32, 24, 18],
  "dropout_ratio": 0.0,
  "batch_size": 32,
  "epochs": 5000,
  "model_fname": "./models/model_20250519-1341_8_train_best.pth"
}
```

| Item              | Description                                                                       |
| :---------------- | :-------------------------------------------------------------------------------- |
| `"model"`         | Type of model to use (currently only "mlp" is supported)[cite: 9].               |
| `"input_cols"`    | List of columns to use from the input CSV[cite: 9].                              |
| `"filter_energy"` | Minimum energy consumption threshold for filtering (EnergyConsumption > 120)[cite: 9]. |
| `"dataset_ratio"` | Ratio used during data training (not used during prediction)[cite: 9].           |
| `"hidden_size"`   | Number of neurons in each hidden layer of the MLP[cite: 9].                      |
| `"dropout_ratio"` | Dropout ratio (for preventing overfitting; 0.0 means no dropout)[cite: 9].       |
| `"batch_size"`    | Batch size used during training (irrelevant for prediction)[cite: 9].            |
| `"epochs"`        | Number of training epochs (irrelevant for prediction)[cite: 9].                  |
| `"model_fname"`   | File path of the trained model (`.pth` format)[cite: 9].                         |

## Input CSV File Requirements

* All columns specified in `input_cols` must exist[cite: 10].
* Each row represents a prediction unit (e.g., hour, day)[cite: 10].
* There should be no missing values (NaN, NaT)[cite: 10].
* The `EnergyConsumption` column can optionally exist; values below `filter_energy` will be excluded if configured[cite: 10].

**Example Input CSV:**

```csv
Temperature,Humidity,Insolation,week,TempDiff,EnergyConsumption
22.5,45.0,300.0,2,3.5,140.0
21.0,50.0,290.0,3,2.8,132.0
...
```

## Installation

To install the necessary packages, run the following command:

```bash
pip install torch pandas numpy scikit-learn gradio
```

## How to Run

The program can be executed in two modes: CLI or Web UI.

### CLI Mode

To run in CLI mode, specify the paths for the model configuration, input folder, and output folder.

```bash
python dce_prediction_mode.py --model_file ./config/config.json --input_folder ./input --output_folder ./output --webapp_server False
```

| Argument          | Description                                    |
| :---------------- | :--------------------------------------------- |
| `--model_file`    | Path to the configuration file (`config.json`)[cite: 11]. |
| `--input_folder`  | Folder containing input CSV files[cite: 11].  |
| `--output_folder` | Folder to save prediction results[cite: 11].  |
| `--webapp_server` | Whether to run in web UI mode (`False` for CLI mode)[cite: 11]. |

Upon completion, predicted CSV files with the same filenames will be generated in the `output` folder, including a `Predicted_EnergyConsumption` column[cite: 12].

### Web UI Mode (Gradio)

To run the web interface, set the `--webapp_server` argument to `True`.

```bash
python dce_prediction_mode.py --webapp_server True
```

After execution, open your web browser and navigate to the address provided by Gradio (usually `http://127.0.0.1:7860` or a similar local address).

You will be able to upload the following files:
* `config.json`
* `model.pth`
* Input CSV files for prediction

Click the "Predict" button to download the results as a ZIP file.

## Prediction Output Format

The output CSV files will include a new column `Predicted_EnergyConsumption`:

```csv
Temperature,Humidity,Insolation,week,TempDiff,Predicted_EnergyConsumption
22.5,45.0,300.0,2,3.5,138.2
21.0,50.0,290.0,3,2.8,129.7
...
```

## Prediction Process Summary

1.  Load model structure and input column information from `config.json`[cite: 13].
2.  Load the trained model (`.pth`) into memory[cite: 13].
3.  Iterate through each input CSV file, performing the following steps:
    * Remove missing values[cite: 13].
    * Apply energy filtering if necessary[cite: 13].
    * Generate model input values after MinMax normalization[cite: 13].
    * Perform model prediction[cite: 13].
    * Generate the result file after inverse normalization[cite: 13].

## Error Handling and Notes

| Situation                                | Action / Message                                       |
| :--------------------------------------- | :----------------------------------------------------- |
| No input CSV files                       | Displays "`no input file.`" message[cite: 14].     |
| Missing input columns                    | Prediction failure or empty results[cite: 14].      |
| Model type is not `"mlp"`                | `NotImplementedError` occurs[cite: 14].             |
| Corrupted model file                     | PyTorch `load_state_dict` failure[cite: 14].        |

## Directory Structure

```
project/
├── dce_prediction_mode.py
├── config/
│   └── config.json
├── models/
│   └── model_20250519-1341_8_train_best.pth
├── input/
│   ├── day1.csv
│   └── day2.csv
└── output/
    ├── day1.csv
    └── day2.csv
```

## Author
Kang Taewook

## Created Date
2025.05.19
