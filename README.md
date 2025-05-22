# ADE Extraction with ModernBERT and DSPy

## Dataset Requirement

**You must download the ADE dataset from HuggingFace:**
[SetFit/ade_corpus_v2_classification](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification)

- Download the dataset and place the `train.jsonl` and `test.jsonl` files in a `data/` directory at the root of this project.

## Project Overview

This project implements a pipeline for Adverse Drug Event (ADE) extraction from medical text using:
- ModernBERT (transformer-based NER model)
- DSPy (for LLM-based extraction and optimization)
- OpenAI GPT-4o-mini (for LLM extraction)

The pipeline includes:
- Preprocessing and tokenization of medical notes
- Extraction of drugs and adverse events (ADEs) using both direct LLM and DSPy-optimized LLM
- Preparation of NER data with robust offset mapping
- Fine-tuning ModernBERT for NER
- Evaluation and comparison of extraction performance (F1, Precision, Recall)

## Setup Instructions

1. **Clone this repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (You may need to install `dspy`, `transformers`, `torch`, `scikit-learn`, `matplotlib`, `pandas`, `python-dotenv`, and `openai`.)

3. **Set your OpenAI API key:**
   - Create a `.env` file in the project root with:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Download the ADE dataset:**
   - Go to [https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification)
   - Download the `train.jsonl` and `test.jsonl` files
   - Place them in a `data/` directory: `data/train.jsonl`, `data/test.jsonl`

## Running the Pipeline

Run the main script:
```bash
python main.py
```

The script will:
- Use the first N records from the ADE dataset for training (configurable in `main.py`)
- Run the extraction pipeline twice:
  1. With DSPy optimization (LLM extraction improved)
  2. With direct LLM extraction (no optimization)
- Fine-tune ModernBERT on the extracted NER data
- Print and compare F1, Precision, and Recall for both approaches

## Output

- The console output will show a comparison of extraction metrics:
  - F1 Score, Precision, Recall for both DSPy-optimized and direct LLM extraction
  - Example entity extraction results
- Model checkpoints and analysis plots will be saved in the project directory

## Features
- Robust NER data preparation using offset mapping for correct token/entity alignment
- Batch processing for large datasets
- Data augmentation for improved training
- Early stopping and learning rate finder for efficient fine-tuning
- Class weight support for imbalanced data
- Visualization and analysis of results

## References
- [ADE Corpus V2 on HuggingFace](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification)
- [DSPy Framework](https://github.com/stanfordnlp/dspy)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)

---

For questions or issues, please open an issue in this repository. 
