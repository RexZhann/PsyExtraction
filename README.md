# PsyExtraction
This  project aim to develop an NLP framework that extracts descriptive phrases for the entered entity.

This repository provides an implementation of a keyword feature extraction framework using Large Language Models (LLMs). The goal of this project is to identify and extract key features of target keywords or phrases from a large corpus of psychological literature. This is especially useful for understanding complex terms and concepts across various psychology domains.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Framework Overview](#framework-overview)
- [Modules](#modules)
- [Evaluation](#evaluation)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The primary objective of this project is to extract critical features related to psychological terms such as "depression" or "bipolar disorder" from relevant literature. By leveraging LLMs, this framework enables the automated extraction of nuanced information, which can be used for further analysis or research purposes.

### Example:
- **Input:** `depression`
- **Output:** `loss of pleasure, reduced sleeping time, low self-esteem`

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

After setting up the environment, you can start extracting keyword features by running the main script. Ensure that your corpus is preprocessed and stored in the appropriate format.

```bash
python main.py
```
