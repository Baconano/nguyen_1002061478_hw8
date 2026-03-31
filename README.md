## Overview
This project implements the **Expectation Maximization (EM)** algorithm for Part-of-Speech (POS) tagging using the Wall Street Journal (WSJ) dataset[cite: 9, 12]. [cite_start]The implementation includes data preprocessing, parameter initialization, the Forward-Backward algorithm (E-step), and parameter updates (M-step)[cite: 16, 17, 18, 19].

## Deliverables 
* `nguyen_1002061478_hw8.py`: Python source code implementing the EM algorithm.
* `report.pdf`: Documentation of the approach, implementation, and results.
* `README.md`: Instructions for running the code.

## Prerequisites
* Python 3.x
* NumPy library (`pip install numpy`)

## Data Files
The following datasets must be present in the same directory as the script:
* `WSJ_02-21.pos` (Training data)
* `WSJ_24.pos` (Test data)

## How to Run
To execute the training and evaluation, run the following command in your terminal:

```bash
python nguyen_1002061478_hw8.py
```
Implementation Details

Preprocessing: Maps words and tags to unique integer indices and handles basic normalization (lowercasing).


Initialization: Uses a semi-supervised seeding approach for emission probabilities to improve convergence speed and accuracy.


EM Algorithm: Iteratively updates transition and emission matrices using the Forward-Backward algorithm until convergence (monitored via Log-Likelihood).


Evaluation: Predicts POS tags for the test set and calculates accuracy against ground-truth labels.
