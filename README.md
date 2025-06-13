# Energy Prediction Model for research
This repository contains the Energy Prediction Model program in KICT R&D project call ERC, designed to predict energy consumption based on a pre-trained deep learning model. You can download [the trained model](https://drive.google.com/drive/folders/1-wKivitvpK1PJbNFvKjvy77hqg7R1siX?usp=sharing) for energy prediction.
It supports both a command-line interface (CLI) and a web-based interface for predictions.

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

This program acts as an energy prediction module. [cite_start]It utilizes a `config.json` file (which defines the model's architecture, input variables, and normalization standards) and a trained model parameter file (`.pth`) to predict energy consumption from input CSV data. [cite_start]The prediction results are output as CSV files and can be used via either a command-line interface (CLI) or a web interface. [cite_start]The `config` file for the generated model's deep learning parameters is designed to be variable, supporting various training model types.

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
| `config.json`      | [cite_start]Model configuration, input variables, and normalization settings. |
| `model.pth`        | [cite_start]The PyTorch file storing the trained model's parameters. |
| `input/*.csv`      | [cite_start]Input data files for prediction.                  |
| `output/*.csv`     | [cite_start]Output files where prediction results will be saved (automatically generated). |

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
