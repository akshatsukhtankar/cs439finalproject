# Parameter-Efficient Fine-Tuning for Financial Sentiment Analysis

This project explores the performance and efficiency tradeoffs between traditional sparse bag-of-words models and modern transformer-based architectures in the real of domain-specific sentiment analysis. Specifically, this project compares a **RoBERTa-base** model adapted via **Low-Rank Adaptation (LoRA)** against a **TF-IDF Logistic Regression** baseline.

## Project Overview

Financial text often contains unique semantic structures where general-purpose models fail (e.g., misclassifying "debt obligations" as purely negative). This study utilizes the **Financial PhraseBank** dataset to demonstrate how parameter-efficient fine-tuning (PEFT) can achieve state-of-the-art accuracy with minimal computational overhead.

### Key Components
- **Baseline:** TF-IDF Vectorization (5,000 features) + Logistic Regression.
- **PEFT Model:** RoBERTa-base with LoRA ($r=16$, $\alpha=32$) targeting the query and value projection matrices.
- **Interpretability:** Model decision-making is analyzed using **SHAP (SHapley Additive exPlanations)** waterfall plots to contrast keyword reliance vs. contextual comprehension. After `baseline.py` and `train_peft.py` finish running each will generate their respective SHAP waterfall plot.  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akshatsukhtankar/cs439finalproject.git
   cd cs439finalproject
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is structured into three main modules:

1. Data Preparation
   `phrasebank_data.py` contains the logic for loading and splitting the Financial PhraseBank dataset. By default, it uses the `sentences_75agree` configuration to ensure high-quality ground truth labels.

2. Baseline Model
   To train and evaluate the traditional machine learning baseline:
   ```bash
   python baseline.py
   ```
   This script will output performance metrics (Accuracy, F1-Score, Runtime) and save a SHAP waterfall plot to the `./results` directory.

3. PEFT (LoRA) Model
   To execute the fine-tuning pipeline for the RoBERTa model:
   ```bash
   python train_peft.py
   ```
   This script performs training for 8 epochs using the Hugging Face `Trainer` API. It saves the evaluation metrics, a SHAP interpretation HTML file, and a waterfall visualization to the `./results` directory.

## Results

Evaluation metrics and visualizations are automatically saved to the `./results` folder:

- `baseline_metrics.json` / `peft_metrics.json`: Detailed performance logs.
- `baseline_shap_waterfall.png` / `peft_shap_waterfall.png`: Feature importance visualizations for qualitative analysis.
- `peft_shap.html`: SHAP text explanation for an example prediction.

## Reproducibility

To ensure identical results, all scripts utilize a fixed random seed of 42. The all-agree or 75-agree subsets of the Financial PhraseBank are used to minimize label noise during the 80/20 train-test split. 

## Summary of Repository Structure
- `phrasebank_data.py`: Handles dataset loading from the Hugging Face Hub.
- `baseline.py`: Implements the scikit-learn TF-IDF pipeline.
- `train_peft.py`: Implements the LoRA fine-tuning and SHAP analysis for the transformer.
- `results/`: Output directory for models and metrics.
