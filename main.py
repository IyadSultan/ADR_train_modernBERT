#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adverse Drug Event (ADE) Extraction and ModernBERT Fine-Tuning

This script implements a complete pipeline for:
1. Processing medical notes
2. Extracting adverse drug events (drug names and adverse reactions)
3. Fine-tuning ModernBERT using the DSPy framework

The pipeline follows these steps:
- Load and preprocess medical notes
- Extract drug mentions and adverse events using DSPy modules
- Prepare extracted data for named entity recognition (NER) training
- Fine-tune ModernBERT on the extracted data
- Optimize the extraction process using DSPy's optimization capabilities
- Visualize and evaluate the results

Author: [Your Name]
Date: [Current Date]
"""

#############################################################
# Import necessary libraries
#############################################################
import os
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# DSPy and Transformer imports
import dspy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


#############################################################
# Constants and Configuration
#############################################################
# Label mappings for NER task
NER_LABELS = {
    "O": 0,       # Outside any entity
    "B-DRUG": 1,  # Beginning of drug mention
    "I-DRUG": 2,  # Inside of drug mention
    "B-ADE": 3,   # Beginning of adverse event
    "I-ADE": 4    # Inside of adverse event
}

# Reverse mapping
ID_TO_LABEL = {v: k for k, v in NER_LABELS.items()}

# ModernBERT model name
MODEL_NAME = "answerdotai/ModernBERT-base"

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

#############################################################
# Medical Note Processing
#############################################################
class MedicalNoteProcessor:
    """
    Class to handle loading and preprocessing of medical notes.
    
    This processor handles the initial text cleaning and preparation
    before any entity extraction takes place.
    """
    
    def __init__(self):
        """Initialize the medical note processor with empty notes list."""
        self.notes = []
        self.processed_notes = []
        
    def load_notes(self, notes_list):
        """
        Load medical notes from a list of strings.
        
        Args:
            notes_list (list): List of medical note strings
            
        Returns:
            int: Number of notes loaded
        """
        self.notes = notes_list
        print(f"Loaded {len(self.notes)} medical notes.")
        return len(self.notes)
        
    def load_notes_from_file(self, file_path):
        """
        Load medical notes from a text file (one note per line).
        
        Args:
            file_path (str): Path to the file containing notes
            
        Returns:
            int: Number of notes loaded
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.notes = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.notes)} medical notes from {file_path}.")
            return len(self.notes)
        except Exception as e:
            print(f"Error loading notes from file: {e}")
            return 0
    
    def preprocess_notes(self):
        """
        Preprocess the medical notes to clean and standardize the text.
        
        This includes:
        - Removing extra whitespace
        - Standardizing punctuation
        - Removing unwanted special characters
        
        Returns:
            list: The processed notes
        """
        processed_notes = []
        for note in self.notes:
            # Remove multiple spaces and replace with single space
            note = re.sub(r'\s+', ' ', note)
            
            # Remove special characters except for allowed punctuation
            note = re.sub(r'[^\w\s.,;:?!()-]', '', note)
            
            # Standardize common abbreviations (expand as needed)
            note = re.sub(r'\bpt\b', 'patient', note, flags=re.IGNORECASE)
            note = re.sub(r'\brx\b', 'prescription', note, flags=re.IGNORECASE)
            
            # Add to processed notes
            processed_notes.append(note)
        
        self.processed_notes = processed_notes
        return processed_notes
    
    def tokenize_notes(self, tokenizer):
        """
        Tokenize the medical notes using the provided tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            
        Returns:
            list: List of tokenized note dictionaries
        """
        tokenized_notes = []
        for note in self.processed_notes:
            # Tokenize with padding and truncation
            tokens = tokenizer(
                note, 
                padding="max_length", 
                truncation=True, 
                max_length=512,  # ModernBERT can handle up to 8192, but this is more efficient
                return_tensors="pt"
            )
            tokenized_notes.append(tokens)
        
        return tokenized_notes


#############################################################
# DSPy Modules for ADE Extraction
#############################################################
class ADEExtractor(dspy.Module):
    """
    DSPy module for extracting adverse drug events from medical notes.
    """
    
    def __init__(self):
        """Initialize the ADE extractor with appropriate signature."""
        super().__init__()
        
        # Define a proper signature class first
        class ExtractADESignature(dspy.Signature):
            """Extract adverse drug events from clinical text."""
            
            # Input field
            text = dspy.InputField(description="Clinical text from a medical note")
            
            # Output fields
            drugs = dspy.OutputField(description="List of drug mentions found in the text")
            adverse_events = dspy.OutputField(description="List of adverse drug events found in the text")
            drug_ade_pairs = dspy.OutputField(description="List of pairs mapping drugs to adverse events in format 'drug: adverse_event'")
        
        # Use the signature class
        self.extract_ade = dspy.ChainOfThought(ExtractADESignature)
    
    def forward(self, text):
        """
        Process a medical note to extract drug-ADE pairs.
        """
        # Use DSPy's chain-of-thought reasoning to extract entities and relationships
        result = self.extract_ade(text=text)
        return result


#############################################################
# Data Processing for ADE Extraction
#############################################################
class ADEDatasetProcessor:
    """
    Processor to extract and structure ADE data from medical notes.
    
    This class takes processed notes and extracts drug-ADE pairs,
    then formats them for NER and relation extraction training.
    """
    
    def __init__(self, notes_processor, ade_extractor):
        """
        Initialize the ADE dataset processor.
        
        Args:
            notes_processor (MedicalNoteProcessor): Processor with processed notes
            ade_extractor (ADEExtractor): DSPy module for ADE extraction
        """
        self.notes_processor = notes_processor
        self.ade_extractor = ade_extractor
        self.extracted_data = []
        self.ner_data = []
        self.relation_data = []
        
    def extract_ades_from_notes(self):
        """
        Extract ADEs from all processed notes.
        
        Uses the DSPy ADE extractor to identify drugs, adverse events,
        and the relationships between them.
        
        Returns:
            list: List of dictionaries with extracted data
        """
        extracted_data = []
        
        # Process each note to extract information
        for note in self.notes_processor.processed_notes:
            # Use the ADE extractor to get drug-ADE pairs
            try:
                extraction_result = self.ade_extractor(note)
                
                # Create a structured record
                record = {
                    'text': note,
                    'drugs': extraction_result.drugs,
                    'adverse_events': extraction_result.adverse_events,
                    'drug_ade_pairs': extraction_result.drug_ade_pairs
                }
                
                extracted_data.append(record)
            except Exception as e:
                print(f"Error extracting ADEs from note: {e}")
                # Add empty record to maintain alignment
                record = {
                    'text': note,
                    'drugs': [],
                    'adverse_events': [],
                    'drug_ade_pairs': []
                }
                extracted_data.append(record)
            
        self.extracted_data = extracted_data
        return extracted_data
        
    def prepare_ner_data(self):
        """
        Prepare data for Named Entity Recognition (NER) training.
        
        Converts extracted drug and ADE mentions into BIO-tagged
        data suitable for NER fine-tuning.
        
        Returns:
            list: List of dictionaries with text and entity annotations
        """
        ner_data = []
        
        for record in self.extracted_data:
            text = record['text']
            entities = []
            
            # Add drug entities
            for drug in record['drugs']:
                # Find all occurrences of the drug name in the text
                for match in re.finditer(re.escape(drug), text):
                    entities.append({
                        'start': match.start(),
                        'end': match.end(),
                        'label': 'DRUG'
                    })
            
            # Add adverse event entities
            for event in record['adverse_events']:
                # Find all occurrences of the adverse event in the text
                for match in re.finditer(re.escape(event), text):
                    entities.append({
                        'start': match.start(),
                        'end': match.end(),
                        'label': 'ADE'
                    })
            
            # Add the annotated example to the NER dataset
            ner_data.append({
                'text': text,
                'entities': entities
            })
            
        self.ner_data = ner_data
        return ner_data
        
    def prepare_relation_data(self):
        """
        Prepare data for Relation Extraction training.
        
        Identifies pairs of drug-ADE entities that have a causal relationship.
        
        Returns:
            list: List of dictionaries with relation annotations
        """
        relation_data = []
        
        for record in self.extracted_data:
            text = record['text']
            
            # Process each drug-ADE pair from the extracted data
            for pair in record['drug_ade_pairs']:
                # Pairs should be in format "drug: adverse_event"
                parts = pair.split(":")
                if len(parts) == 2:
                    drug, ade = parts[0].strip(), parts[1].strip()
                    
                    # Find positions of all occurrences in text
                    drug_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(drug), text)]
                    ade_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(ade), text)]
                    
                    # Add relation data for each drug-ADE occurrence
                    for drug_pos in drug_positions:
                        for ade_pos in ade_positions:
                            relation_data.append({
                                'text': text,
                                'drug': drug,
                                'drug_start': drug_pos[0],
                                'drug_end': drug_pos[1],
                                'ade': ade,
                                'ade_start': ade_pos[0],
                                'ade_end': ade_pos[1],
                                'relation': 'CAUSES'  # Relationship type
                            })
        
        self.relation_data = relation_data
        return relation_data


#############################################################
# Dataset Class for ModernBERT Fine-tuning
#############################################################
class ADEDataset(Dataset):
    """
    PyTorch Dataset for ADE extraction fine-tuning.
    
    This dataset prepares tokenized inputs with NER labels
    for training ModernBERT.
    """
    
    def __init__(self, texts, tags, tokenizer, max_len=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text strings
            tags (list): List of tag sequences (numeric IDs)
            tokenizer: HuggingFace tokenizer
            max_len (int): Maximum sequence length
        """
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): Example index
            
        Returns:
            dict: Tokenized inputs with labels
        """
        text = self.texts[idx]
        tags = self.tags[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Convert tags to tensor and pad to match input length
        # Tags that don't align with tokens get label "O" (0)
        labels = torch.tensor(tags + [0] * (self.max_len - len(tags)), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


#############################################################
# ModernBERT Fine-tuning Class
#############################################################
class ModernBERTFineTuner:
    """
    Fine-tunes ModernBERT for ADE extraction using NER approach.
    
    This class handles the preparation of data, training loop, and
    evaluation of ModernBERT for the ADE extraction task.
    """
    
    def __init__(self, model_name=MODEL_NAME):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name (str): HuggingFace model name or path
        """
        self.model_name = model_name
        
        # Load tokenizer and model
        print(f"Loading ModernBERT model and tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model for token classification (NER)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(NER_LABELS),  # O, B-DRUG, I-DRUG, B-ADE, I-ADE
            id2label=ID_TO_LABEL,
            label2id=NER_LABELS
        )
        print("Model and tokenizer loaded successfully.")
        
    def prepare_ner_data(self, ner_data):
        """
        Convert NER data to format suitable for training.
        
        Args:
            ner_data (list): List of dictionaries with text and entities
            
        Returns:
            tuple: (texts, tags_list) - Lists of texts and their token tags
        """
        texts = []
        tags_list = []
        
        for item in ner_data:
            text = item['text']
            entities = sorted(item['entities'], key=lambda x: x['start'])
            
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            
            # Create tags for each token (using BIO scheme)
            # Initially all tokens are "outside" any entity
            tags = ['O'] * len(tokens)
            
            # Label entities using BIO scheme
            for entity in entities:
                start, end = entity['start'], entity['end']
                entity_text = text[start:end]
                
                # Tokenize just the entity text
                entity_tokens = self.tokenizer.tokenize(entity_text)
                
                # Find where these tokens appear in the full tokenized text
                # This handles subword tokenization
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        # Mark as B-LABEL for first token, I-LABEL for rest
                        if entity['label'] == 'DRUG':
                            tags[i] = 'B-DRUG'
                            for j in range(1, len(entity_tokens)):
                                tags[i+j] = 'I-DRUG'
                        elif entity['label'] == 'ADE':
                            tags[i] = 'B-ADE'
                            for j in range(1, len(entity_tokens)):
                                tags[i+j] = 'I-ADE'
            
            # Convert string tags to IDs
            numeric_tags = [self.tag_to_id(tag) for tag in tags]
            
            texts.append(text)
            tags_list.append(numeric_tags)
        
        return texts, tags_list
    
    def tag_to_id(self, tag):
        """
        Convert tag string to numeric ID.
        
        Args:
            tag (str): Tag string (e.g., 'O', 'B-DRUG', etc.)
            
        Returns:
            int: Numeric ID for the tag
        """
        return NER_LABELS.get(tag, 0)  # Default to "O" (outside) if tag not found
    
    def evaluate_base_model(self, test_texts, test_tags):
        """
        Evaluate the base ModernBERT model (before fine-tuning) on test data.
        
        Args:
            test_texts (list): Test text examples
            test_tags (list): Test tag sequences
            
        Returns:
            dict: Metrics including F1, precision, and recall
        """
        print("Loading base ModernBERT model...")
        # Load the base model without fine-tuning
        base_model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(NER_LABELS),
            id2label=ID_TO_LABEL,
            label2id=NER_LABELS
        )
        
        # Create test dataset and dataloader
        test_dataset = ADEDataset(test_texts, test_tags, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=8)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        base_model.eval()
        
        # Evaluation loop
        test_loss = 0
        y_true = []
        y_pred = []
        
        print("Evaluating base model...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                test_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                # Only consider tokens that are part of the input (not padding)
                active_mask = attention_mask == 1
                active_labels = torch.where(
                    active_mask, labels, torch.tensor(0).type_as(labels)
                )
                active_preds = torch.where(
                    active_mask, predictions, torch.tensor(0).type_as(predictions)
                )
                
                # Collect for metrics calculation
                y_true.extend(active_labels.cpu().numpy().flatten())
                y_pred.extend(active_preds.cpu().numpy().flatten())
        
        # Calculate metrics (exclude padding tokens with label=0)
        metrics = {'f1': 0, 'precision': 0, 'recall': 0}
        mask = np.array(y_true) != 0
        
        if mask.sum() > 0:  # Ensure we have non-padding tokens
            y_true_masked = np.array(y_true)[mask]
            y_pred_masked = np.array(y_pred)[mask]
            
            # Calculate metrics
            metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
            metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
            
            print(f"Base Model Test Loss: {test_loss/len(test_loader):.4f}")
            print(f"Base Model F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        else:
            print("No non-padding tokens found in test set")
        
        return metrics


#############################################################
# DSPy Optimization Class
#############################################################

class ADEOptimizer:
    """
    Uses DSPy optimization techniques to improve ADE extraction.
    
    This class leverages DSPy's optimization capabilities to enhance
    the performance of the ADE extractor.
    """
    
    def __init__(self, ade_extractor):
        """
        Initialize the optimizer.
        
        Args:
            ade_extractor (ADEExtractor): The DSPy ADE extractor module
        """
        self.ade_extractor = ade_extractor
        
    def prepare_examples(self, extracted_data):
        """
        Prepare examples for DSPy optimization.
        
        Args:
            extracted_data (list): Extracted data records
            
        Returns:
            list: List of DSPy examples
        """
        examples = []
        print(f"Preparing {len(extracted_data)} examples for DSPy optimization")
        
        for record in extracted_data:
            # Create example with expected input/output for DSPy
            example = dspy.Example(
                text=record['text'],
                drugs=record['drugs'],
                adverse_events=record['adverse_events'],
                drug_ade_pairs=record['drug_ade_pairs']
            ).with_inputs('text')  # Specify 'text' as the input field
            
            examples.append(example)
        
        return examples
    
    def evaluate_extraction(self, gold, pred):
        """
        Evaluate extraction quality for DSPy optimization.
        
        Args:
            gold: Gold standard (expected) output
            pred: Predicted output
            
        Returns:
            float: F1 score measuring extraction quality
        """
        # Calculate precision and recall for drug extraction
        # Handle edge cases (empty lists) with max() to avoid division by zero
        drug_precision = len(set(pred.drugs) & set(gold.drugs)) / max(len(pred.drugs), 1)
        drug_recall = len(set(pred.drugs) & set(gold.drugs)) / max(len(gold.drugs), 1)
        drug_f1 = 2 * drug_precision * drug_recall / max(drug_precision + drug_recall, 1e-6)
        
        # Calculate precision and recall for ADE extraction
        ade_precision = len(set(pred.adverse_events) & set(gold.adverse_events)) / max(len(pred.adverse_events), 1)
        ade_recall = len(set(pred.adverse_events) & set(gold.adverse_events)) / max(len(gold.adverse_events), 1)
        ade_f1 = 2 * ade_precision * ade_recall / max(ade_precision + ade_recall, 1e-6)
        
        # Return average F1 score
        avg_f1 = (drug_f1 + ade_f1) / 2
        return avg_f1
    
    def optimize(self, examples):
        """
        Optimize ADE extraction using DSPy or return the original extractor if optimization fails.
        """
        print(f"\n{'='*20} OPTIMIZING ADE EXTRACTION {'='*20}")
        
        # Split examples into training and development sets
        train_examples, dev_examples = train_test_split(examples, test_size=0.2, random_state=42)
        print(f"Training set: {len(train_examples)} examples")
        print(f"Development set: {len(dev_examples)} examples")
        
        try:
            # Simplest possible approach - try to create the optimizer 
            # with just the required arguments
            optimizer = dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=3
            )
            
            print("Running DSPy optimization...")
            
            # Try the simplest compile call with just the module
            optimized_extractor = optimizer.compile(self.ade_extractor)
            print("DSPy optimization complete!")
            return optimized_extractor
            
        except Exception as e:
            # If any error occurs, log it and return the original extractor
            print(f"Error during DSPy optimization: {e}")
            print("Falling back to the original non-optimized extractor")
            return self.ade_extractor

#############################################################
# Complete Pipeline Function
#############################################################
def run_ade_extraction_pipeline(medical_notes):
    """
    Run the complete ADE extraction and ModernBERT fine-tuning pipeline.
    
    This function orchestrates the entire process from data processing
    to model fine-tuning and evaluation.
    
    Args:
        medical_notes (list): List of medical note strings
        
    Returns:
        dict: Pipeline results including models and extracted data
    """
    print(f"\n{'='*20} STARTING ADE EXTRACTION PIPELINE {'='*20}")
    
    # Configure DSPy with OpenAI
    print("Configuring DSPy with language model...")
    try:
        # DSPy uses this format for OpenAI models
        model_name = "openai/gpt-4o-mini"  # or "openai/gpt-4" for better results
        
        # Initialize with the correct format
        lm = dspy.LM(model_name)
        dspy.settings.configure(lm=lm)
        print(f"Successfully configured DSPy with model: {model_name}")
    except Exception as e:
        print(f"Error configuring OpenAI: {e}")
        raise
    
    # Step 1: Process the medical notes
    print("\nStep 1: Processing medical notes...")
    notes_processor = MedicalNoteProcessor()
    notes_processor.load_notes(medical_notes)
    processed_notes = notes_processor.preprocess_notes()
    print(f"Processed {len(processed_notes)} medical notes")
    
    # Step 2: Initialize and use ADE extractor
    print("\nStep 2: Extracting ADEs from notes...")
    ade_extractor = ADEExtractor()
    
    # Step 3: Process notes to extract ADEs
    print("\nStep 3: Processing notes with ADE extractor...")
    dataset_processor = ADEDatasetProcessor(notes_processor, ade_extractor)
    extracted_data = dataset_processor.extract_ades_from_notes()
    
    # Step 4: Prepare data for training
    print("\nStep 4: Preparing data for NER and relation extraction...")
    ner_data = dataset_processor.prepare_ner_data()
    relation_data = dataset_processor.prepare_relation_data()
    
    # Step 5: Optimize the ADE extractor using DSPy
    print("\nStep 5: Optimizing ADE extraction with DSPy...")
    optimizer = ADEOptimizer(ade_extractor)
    examples = optimizer.prepare_examples(extracted_data)
    optimized_extractor = optimizer.optimize(examples)
    
    # Step 6: Prepare for ModernBERT fine-tuning
    print("\nStep 6: Preparing data for ModernBERT fine-tuning...")
    finetuner = ModernBERTFineTuner()
    train_texts, train_tags = finetuner.prepare_ner_data(ner_data)
    
    # Step 7: Split data for training/validation/test
    print("\nStep 7: Splitting data for training, validation, and testing...")
    # First split into train and temp (80/20)
    train_texts, temp_texts, train_tags, temp_tags = train_test_split(
        train_texts, train_tags, test_size=0.2, random_state=42
    )
    # Then split temp into validation and test (50/50, which is 10/10 of original)
    val_texts, test_texts, val_tags, test_tags = train_test_split(
        temp_texts, temp_tags, test_size=0.5, random_state=42
    )
    print(f"Training set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_texts)} examples")
    print(f"Test set: {len(test_texts)} examples")
    
    # NEW STEP: Evaluate base ModernBERT model before fine-tuning
    print("\nStep 8: Evaluating base ModernBERT model...")
    base_metrics = finetuner.evaluate_base_model(test_texts, test_tags)
    
    # Step 9: Fine-tune ModernBERT
    print("\nStep 9: Fine-tuning ModernBERT for ADE extraction...")
    model, tokenizer, training_metrics = finetuner.train(
        train_texts, train_tags, val_texts, val_tags, epochs=3
    )
    
    # Step 10: Define extraction function using fine-tuned model
    print("\nStep 10: Creating inference function with fine-tuned model...")
    
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def extract_ades_with_finetuned_model(text):
        """
        Extract ADEs from text using the fine-tuned ModernBERT model.
        
        Args:
            text (str): Medical note text
            
        Returns:
            dict: Extracted drugs and adverse events
        """
        # Tokenize the input text
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Get predictions
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert predictions to tags
        predicted_tags = [ID_TO_LABEL[p.item()] for p in predictions[0]]
        
        # Extract entities
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        drugs = []
        ades = []
        current_drug = []
        current_ade = []
        
        # Process each token and its predicted tag
        for token, tag in zip(tokens, predicted_tags):
            if tag == "B-DRUG":
                # Start of a new drug entity
                if current_drug:
                    # Add previous drug entity if exists
                    drugs.append("".join(current_drug).replace("##", ""))
                    current_drug = []
                current_drug.append(token)
            elif tag == "I-DRUG" and current_drug:
                # Continuation of drug entity
                current_drug.append(token)
            elif tag == "B-ADE":
                # Start of a new adverse event entity
                if current_ade:
                    # Add previous ADE entity if exists
                    ades.append("".join(current_ade).replace("##", ""))
                    current_ade = []
                current_ade.append(token)
            elif tag == "I-ADE" and current_ade:
                # Continuation of adverse event entity
                current_ade.append(token)
            else:
                # Outside any entity - finalize current entities if they exist
                if current_drug:
                    drugs.append("".join(current_drug).replace("##", ""))
                    current_drug = []
                if current_ade:
                    ades.append("".join(current_ade).replace("##", ""))
                    current_ade = []
        
        # Add any remaining entities
        if current_drug:
            drugs.append("".join(current_drug).replace("##", ""))
        if current_ade:
            ades.append("".join(current_ade).replace("##", ""))
        
        # Clean up special tokens and whitespace
        drugs = [d.replace("[CLS]", "").replace("[SEP]", "").strip() for d in drugs if d]
        ades = [a.replace("[CLS]", "").replace("[SEP]", "").strip() for a in ades if a]
        
        return {
            "drugs": drugs,
            "adverse_events": ades
        }
    
    # Step 11: Evaluate the fine-tuned model on the test set
    print("\nStep 11: Evaluating fine-tuned ModernBERT model...")
    finetuned_metrics = {}
    
    # Create test dataset and dataloader
    test_dataset = ADEDataset(test_texts, test_tags, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Evaluation loop
    test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            test_loss += outputs.loss.item()
            
            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Only consider tokens that are part of the input (not padding)
            active_mask = attention_mask == 1
            active_labels = torch.where(
                active_mask, labels, torch.tensor(0).type_as(labels)
            )
            active_preds = torch.where(
                active_mask, predictions, torch.tensor(0).type_as(predictions)
            )
            
            # Collect for metrics calculation
            y_true.extend(active_labels.cpu().numpy().flatten())
            y_pred.extend(active_preds.cpu().numpy().flatten())
    
    # Calculate metrics (exclude padding tokens with label=0)
    mask = np.array(y_true) != 0
    if mask.sum() > 0:  # Ensure we have non-padding tokens
        y_true_masked = np.array(y_true)[mask]
        y_pred_masked = np.array(y_pred)[mask]
        
        # Calculate metrics
        finetuned_metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
        finetuned_metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        finetuned_metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        print(f"Fine-tuned Model Test Loss: {test_loss/len(test_loader):.4f}")
        print(f"Fine-tuned Model F1: {finetuned_metrics['f1']:.4f}, Precision: {finetuned_metrics['precision']:.4f}, Recall: {finetuned_metrics['recall']:.4f}")
        
        # Compare with base model metrics
        f1_improvement = finetuned_metrics['f1'] - base_metrics['f1']
        precision_improvement = finetuned_metrics['precision'] - base_metrics['precision']
        recall_improvement = finetuned_metrics['recall'] - base_metrics['recall']
        
        print("\nModernBERT Performance Comparison (Before vs. After Fine-tuning):")
        print(f"F1 Score:   {base_metrics['f1']:.4f} → {finetuned_metrics['f1']:.4f} (Change: {f1_improvement:+.4f})")
        print(f"Precision:  {base_metrics['precision']:.4f} → {finetuned_metrics['precision']:.4f} (Change: {precision_improvement:+.4f})")
        print(f"Recall:     {base_metrics['recall']:.4f} → {finetuned_metrics['recall']:.4f} (Change: {recall_improvement:+.4f})")
    else:
        print("No non-padding tokens found in test set")
        finetuned_metrics = {'f1': 0, 'precision': 0, 'recall': 0}
    
    print("\nPipeline complete!")
    
    # Return all components for further use
    return {
        "optimized_extractor": optimized_extractor,
        "finetuned_model": model,
        "tokenizer": tokenizer,
        "extraction_function": extract_ades_with_finetuned_model,
        "metrics": {
            "base_model": base_metrics,
            "finetuned_model": finetuned_metrics,
            "training": training_metrics
        },
        "processed_data": {
            "extracted_data": extracted_data,
            "ner_data": ner_data,
            "relation_data": relation_data
        }
    }

#############################################################
# Visualization Functions
#############################################################
def visualize_extractions(extraction_results):
    """
    Visualize ADE extraction results across all notes.
    
    Args:
        extraction_results (list): List of extraction result dictionaries
        
    Returns:
        dict: Visualization statistics
    """
    print(f"\n{'='*20} VISUALIZING EXTRACTION RESULTS {'='*20}")
    
    # Count drug and ADE frequencies
    drug_counts = {}
    ade_counts = {}
    drug_ade_pairs = []
    
    # Process all extraction results
    for result in extraction_results:
        # Count drug mentions
        for drug in result['drugs']:
            drug_counts[drug] = drug_counts.get(drug, 0) + 1
        
        # Count adverse event mentions
        for ade in result['adverse_events']:
            ade_counts[ade] = ade_counts.get(ade, 0) + 1
        
        # Record drug-ADE pairs
        for pair in result['drug_ade_pairs']:
            parts = pair.split(":")
            if len(parts) == 2:
                drug_ade_pairs.append((parts[0].strip(), parts[1].strip()))
    
    # Plot drug frequencies
    plt.figure(figsize=(12, 6))
    plt.bar(drug_counts.keys(), drug_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Drug Mention Frequencies')
    plt.xlabel('Drug')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('drug_frequencies.png')
    print("Drug frequency chart saved to 'drug_frequencies.png'")
    
    # Plot ADE frequencies
    plt.figure(figsize=(12, 6))
    plt.bar(ade_counts.keys(), ade_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Adverse Event Frequencies')
    plt.xlabel('Adverse Event')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('ade_frequencies.png')
    print("Adverse event frequency chart saved to 'ade_frequencies.png'")
    
    # Analyze drug-ADE pairs
    print("\nTop Drug-ADE Pairs:")
    pair_counts = {}
    for drug, ade in drug_ade_pairs:
        pair = f"{drug} → {ade}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Sort by frequency and display top pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs[:10]:  # Top 10 pairs
        print(f"{pair}: {count}")
    
    # Save results to a file
    with open("ade_extraction_results.json", "w") as f:
        json.dump({
            'drug_counts': drug_counts,
            'ade_counts': ade_counts,
            'drug_ade_pairs': [f"{drug} → {ade}" for drug, ade in drug_ade_pairs]
        }, f, indent=2)
    print("\nDetailed results saved to 'ade_extraction_results.json'")
    
    return {
        'drug_counts': drug_counts,
        'ade_counts': ade_counts,
        'drug_ade_pairs': drug_ade_pairs
    }


#############################################################
# Main Execution Function
#############################################################
def clean_note(note):
    """
    Clean and format a medical note for processing.
    
    Args:
        note (str): Raw medical note text
        
    Returns:
        str: Cleaned note text
    """
    # Remove extra whitespace
    note = ' '.join(note.split())
    
    # Remove any special characters that might interfere with processing
    note = re.sub(r'[^\w\s.,;:?!()-]', '', note)
    
    return note
def visualize_model_comparison(metrics):
    """
    Visualize performance comparison between base and fine-tuned models.
    
    Args:
        metrics (dict): Dictionary containing base and fine-tuned model metrics
        
    Returns:
        None
    """
    print(f"\n{'='*20} VISUALIZING MODEL COMPARISON {'='*20}")
    
    # Extract metrics
    base_metrics = metrics['base_model']
    finetuned_metrics = metrics['finetuned_model']
    training_metrics = metrics['training']
    
    # Plot F1, Precision, and Recall comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar positions
    models = ['Base ModernBERT', 'Fine-tuned ModernBERT']
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars for each metric
    plt.bar(x - width, [base_metrics['f1'], finetuned_metrics['f1']], width, label='F1 Score')
    plt.bar(x, [base_metrics['precision'], finetuned_metrics['precision']], width, label='Precision')
    plt.bar(x + width, [base_metrics['recall'], finetuned_metrics['recall']], width, label='Recall')
    
    # Customize plot
    plt.ylabel('Score')
    plt.title('Performance Comparison: Base vs. Fine-tuned ModernBERT')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add value labels on the bars
    for i, model in enumerate(models):
        plt.text(i - width, base_metrics['f1'] + 0.02, f"{base_metrics['f1']:.3f}", ha='center')
        plt.text(i, base_metrics['precision'] + 0.02, f"{base_metrics['precision']:.3f}", ha='center')
        plt.text(i + width, base_metrics['recall'] + 0.02, f"{base_metrics['recall']:.3f}", ha='center')
        
        if i == 1:  # Fine-tuned model
            plt.text(i - width, finetuned_metrics['f1'] + 0.02, f"{finetuned_metrics['f1']:.3f}", ha='center')
            plt.text(i, finetuned_metrics['precision'] + 0.02, f"{finetuned_metrics['precision']:.3f}", ha='center')
            plt.text(i + width, finetuned_metrics['recall'] + 0.02, f"{finetuned_metrics['recall']:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Model comparison chart saved to 'model_comparison.png'")
    
    # If we have training metrics, plot the learning curve
    if 'epoch_losses' in training_metrics and len(training_metrics['epoch_losses']) > 0:
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(training_metrics['epoch_losses']) + 1)
        
        plt.plot(epochs, training_metrics['epoch_losses'], 'b-', label='Training Loss')
        
        if 'val_losses' in training_metrics and len(training_metrics['val_losses']) > 0:
            plt.plot(epochs, training_metrics['val_losses'], 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('learning_curve.png')
        print("Learning curve saved to 'learning_curve.png'")
        
        # If we have F1 scores, plot them too
        if 'f1_scores' in training_metrics and len(training_metrics['f1_scores']) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, training_metrics['f1_scores'], 'g-', label='F1 Score')
            plt.plot(epochs, training_metrics['precision_scores'], 'b-', label='Precision')
            plt.plot(epochs, training_metrics['recall_scores'], 'r-', label='Recall')
            plt.title('Training Metrics by Epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.ylim(0, 1.0)
            plt.savefig('training_metrics.png')
            print("Training metrics curve saved to 'training_metrics.png'")
    
    # Save the detailed metrics to a JSON file
    with open("model_evaluation_metrics.json", "w") as f:
        json.dump({
            'base_model': base_metrics,
            'finetuned_model': finetuned_metrics,
            'improvements': {
                'f1': finetuned_metrics['f1'] - base_metrics['f1'],
                'precision': finetuned_metrics['precision'] - base_metrics['precision'],
                'recall': finetuned_metrics['recall'] - base_metrics['recall']
            },
            'training_metrics': training_metrics
        }, f, indent=2)
    print("Detailed evaluation metrics saved to 'model_evaluation_metrics.json'")


def main():
    """
    Main execution function to run the ADE extraction pipeline.
    
    This function demonstrates the full pipeline with example medical notes.
    In a real application, replace these with your actual notes.
    """
    print(f"\n{'='*50}")
    print(f"{'ADVERSE DRUG EVENT EXTRACTION WITH ModernBERT':^50}")
    print(f"{'='*50}")
    
    # Load notes from the formatted file
    notes_processor = MedicalNoteProcessor()
    notes_processor.load_notes_from_file("formatted_notes.txt")
    
    # Clean and preprocess the notes
    medical_notes = [clean_note(note) for note in notes_processor.notes]
    print(f"Loaded and cleaned {len(medical_notes)} medical notes")
    
    # Run the pipeline
    pipeline_results = run_ade_extraction_pipeline(medical_notes)
    
    # Extract ADEs from a new note using the fine-tuned model
    print("\nTesting extraction on a new note...")
    new_note = "Patient was prescribed amoxicillin 500mg TID for bacterial infection. After 2 days, patient reported skin rash and itching."
    extraction_result = pipeline_results["extraction_function"](new_note)
    
    print("Extracted Drugs:", extraction_result["drugs"])
    print("Extracted Adverse Events:", extraction_result["adverse_events"])
    
    # Save the fine-tuned model for future use
    model_dir = "./ade_extraction_model"
    tokenizer_dir = "./ade_extraction_tokenizer"
    print(f"\nSaving fine-tuned model to {model_dir} and tokenizer to {tokenizer_dir}")
    pipeline_results["finetuned_model"].save_pretrained(model_dir)
    pipeline_results["tokenizer"].save_pretrained(tokenizer_dir)
    
    # Example of using the optimized DSPy extractor
    print("\nTesting extraction with optimized DSPy extractor...")
    optimized_extractor = pipeline_results["optimized_extractor"]
    dspy_result = optimized_extractor(text=new_note)
    
    print("DSPy Extraction Results:")
    print("Drugs:", dspy_result.drugs)
    print("Adverse Events:", dspy_result.adverse_events)
    print("Drug-ADE Pairs:", dspy_result.drug_ade_pairs)
    
    # Visualize the extraction results
    visualize_extractions(pipeline_results["processed_data"]["extracted_data"])
    
    # Visualize the model comparison (new)
    visualize_model_comparison(pipeline_results["metrics"])
    
    print(f"\n{'='*50}")
    print(f"{'PIPELINE EXECUTION COMPLETE':^50}")
    print(f"{'='*50}")

#############################################################
# Program Entry Point
#############################################################
if __name__ == "__main__":
    """
    Entry point when running the script directly.
    
    This ensures the main() function is only called when
    the script is executed, not when imported.
    """
    # You can add argument parsing here if needed
    main()