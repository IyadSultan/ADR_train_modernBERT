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

This enhanced version includes:
- Improved DSPy optimization with error handling
- Enhanced entity extraction from tokens
- Scalable batch processing for large datasets
- Data augmentation for better training
- Advanced training with learning rate finder and early stopping
- Class weight support for imbalanced data
- Comprehensive visualization and analysis
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

class DirectLLMExtractor:
    """
    Direct LLM-based extractor that uses OpenAI directly without DSPy optimization.
    This provides direct access to GPT-4o-mini when DSPy optimization is disabled.
    """
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize with the specified model name."""
        self.model_name = model_name
        print(f"Initializing direct LLM extractor with {model_name}")
        
        # Import OpenAI library
        try:
            import openai
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
            print("OpenAI client initialized successfully")
        except ImportError:
            print("Error: OpenAI library not installed. Please install with: pip install openai")
            raise
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise
    
    def __call__(self, text):
        """
        Extract drugs and adverse events directly using the LLM.
        
        Args:
            text (str): Medical note text
            
        Returns:
            object: Object with drugs, adverse_events, and drug_ade_pairs attributes
        """
        # Create a prompt that asks for drug and adverse event extraction
        prompt = f"""
        Extract all medications (drugs) and adverse events/side effects from the following medical note.
        Format the output as JSON with three fields:
        1. "drugs": A list of drug names mentioned
        2. "adverse_events": A list of adverse events or side effects mentioned
        3. "drug_ade_pairs": A list of strings in the format "drug: adverse_event" for any drug and adverse event that appear to be related

        Medical Note:
        {text}

        Only include actual medications and genuine adverse events. Don't include normal medical conditions unless they appear to be side effects of medications.
        """
        
        try:
            # Call the OpenAI API directly
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical data extraction assistant that identifies drugs and adverse events in clinical notes."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract the generated response
            result_text = response.choices[0].message.content
            
            # Parse the JSON
            import json
            result = json.loads(result_text)
            
            # Create result object with expected attributes
            print(f"LLM extracted: {len(result.get('drugs', []))} drugs, {len(result.get('adverse_events', []))} ADEs")
            return type('ExtractorResult', (), {
                'drugs': result.get('drugs', []),
                'adverse_events': result.get('adverse_events', []),
                'drug_ade_pairs': result.get('drug_ade_pairs', [])
            })
            
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            # Return empty results on error
            return type('ExtractorResult', (), {
                'drugs': [],
                'adverse_events': [],
                'drug_ade_pairs': []
            })
        

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
# Improved Entity Extraction Functions
#############################################################
def extract_entities_improved(text, model, tokenizer, device):
    print(f"Processing text: {text[:50]}...")
    """
    Extract drugs and adverse events with improved handling of subword tokens.
    
    Args:
        text (str): The medical note text
        model: Fine-tuned ModernBERT model
        tokenizer: ModernBERT tokenizer
        device: Computation device
        
    Returns:
        dict: Dictionary with extracted drugs and adverse events
    """
    # Step 1: Tokenize text with offset mapping to track character positions
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True,
        truncation=True, 
        padding=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].numpy()
    
    # Step 2: Get model predictions
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Print token debugging info for first few tokens
    print("\nToken debugging (first 10 tokens):")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for i in range(min(10, len(tokens))):
        if offset_mapping[i][0] != offset_mapping[i][1]:  # Skip special tokens
            token = tokens[i]
            pred_id = predictions[i]
            label = "O"
            if pred_id == 1: label = "B-DRUG"
            elif pred_id == 2: label = "I-DRUG"
            elif pred_id == 3: label = "B-ADE"
            elif pred_id == 4: label = "I-ADE"
            
            char_span = f"{offset_mapping[i][0]}-{offset_mapping[i][1]}"
            token_text = text[offset_mapping[i][0]:offset_mapping[i][1]]
            print(f"  Token {i}: '{token}' → '{token_text}' (chars {char_span}) = {label}")
    
    # Step 3: Extract entities using character-level positions
    drugs = []
    adverse_events = []
    
    i = 0
    while i < len(predictions):
        # Skip paddings, CLS, SEP tokens which have offset (0,0)
        if i >= len(offset_mapping) or offset_mapping[i][0] == offset_mapping[i][1] == 0:
            i += 1
            continue
        
        # Check for entity starts
        pred_id = predictions[i]
        if pred_id == 1:  # B-DRUG
            # Find complete entity span
            start_char = offset_mapping[i][0]
            end_char = offset_mapping[i][1]
            j = i + 1
            
            # Continue until end of entity
            while j < len(predictions) and j < len(offset_mapping) and predictions[j] == 2:  # I-DRUG
                end_char = offset_mapping[j][1]
                j += 1
            
            # Extract drug entity
            drug = text[start_char:end_char].strip()
            if drug and len(drug) > 1:
                drugs.append(drug)
                print(f"  Found DRUG: '{drug}' (chars {start_char}-{end_char})")
            
            i = j  # Move to next token after entity
            
        elif pred_id == 3:  # B-ADE
            # Find complete entity span
            start_char = offset_mapping[i][0]
            end_char = offset_mapping[i][1]
            j = i + 1
            
            # Continue until end of entity
            while j < len(predictions) and j < len(offset_mapping) and predictions[j] == 4:  # I-ADE
                end_char = offset_mapping[j][1]
                j += 1
            
            # Extract adverse event entity
            ade = text[start_char:end_char].strip()
            if ade and len(ade) > 1:
                adverse_events.append(ade)
                print(f"  Found ADE: '{ade}' (chars {start_char}-{end_char})")
            
            i = j  # Move to next token after entity
        else:
            i += 1
    
    # Step 4: Post-process to remove any duplicates and very common words
    stopwords = {'the', 'and', 'for', 'was', 'with', 'that', 'this', 'patient', 'reported'}
    drugs = [d for d in set(drugs) if d.lower() not in stopwords]
    adverse_events = [a for a in set(adverse_events) if a.lower() not in stopwords]
    
    print("\nFinal extracted entities:")
    print(f"  - Drugs: {drugs}")
    print(f"  - Adverse Events: {adverse_events}")
    
    return {
        "drugs": drugs,
        "adverse_events": adverse_events
    }


def extract_ades_with_finetuned_model(text, model, tokenizer, device):
    """
    Extract ADEs from text using the fine-tuned ModernBERT model.
    
    Args:
        text (str): Medical note text
        model: Fine-tuned model
        tokenizer: Tokenizer
        device: Computation device
        
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
    
    # Use the improved entity extraction
    drugs, adverse_events = extract_entities_from_tokens(tokens, predicted_tags)
    
    return {
        "drugs": drugs,
        "adverse_events": adverse_events
    }


#############################################################
# Scalable Batch Processing for Large Datasets
#############################################################
def process_notes_in_batches(notes, batch_size=100, processor_fn=None):
    """
    Process a large list of notes in batches to avoid memory issues.
    
    Args:
        notes (list): List of medical notes to process
        batch_size (int): Number of notes to process in each batch
        processor_fn (callable): Function to process each batch of notes
        
    Returns:
        list: Combined results from all batches
    """
    results = []
    total_notes = len(notes)
    
    for i in range(0, total_notes, batch_size):
        # Get the current batch
        batch = notes[i:min(i+batch_size, total_notes)]
        print(f"Processing batch {i//batch_size + 1}/{(total_notes + batch_size - 1)//batch_size} ({len(batch)} notes)")
        
        # Process the batch
        if processor_fn:
            batch_results = processor_fn(batch)
            results.extend(batch_results)
        
    return results


#############################################################
# Data Processing for ADE Extraction
#############################################################
class ADEDatasetProcessor:
    """
    Processor to extract and structure ADE data from medical notes.
    
    This class takes processed notes and extracts drug-ADE pairs,
    then formats them for NER and relation extraction training.
    """
    
    def __init__(self, notes_processor, ade_extractor, tokenizer):
        """
        Initialize the ADE dataset processor.
        
        Args:
            notes_processor (MedicalNoteProcessor): Processor with processed notes
            ade_extractor (ADEExtractor): DSPy module for ADE extraction
            tokenizer: HuggingFace tokenizer
        """
        self.notes_processor = notes_processor
        self.ade_extractor = ade_extractor
        self.tokenizer = tokenizer
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
        
    def prepare_ner_data(self, ner_data):
        """
        Enhanced version that ensures proper tagging of entities.
        Adds additional logging to identify issues.
        """
        print(f"Converting {len(ner_data)} examples to BIO format...")
        texts = []
        tags_list = []
        
        # Count stats for diagnostics
        total_entities = 0
        empty_examples = 0
        entity_examples = 0
        
        for i, item in enumerate(ner_data):
            text = item['text']
            entities = sorted(item['entities'], key=lambda x: x['start']) if 'entities' in item else []
            
            # Skip empty texts
            if not text or not text.strip():
                empty_examples += 1
                continue
            
            # Count entities for diagnostics
            total_entities += len(entities)
            if entities:
                entity_examples += 1
                
            # Debug first few examples
            if i < 3 and entities:
                print(f"\nExample {i+1} with {len(entities)} entities:")
                print(f"Text: {text[:100]}...")
                for e in entities[:3]:
                    print(f"  {e['label']} ({e['start']}:{e['end']}): '{text[e['start']:e['end']]}'")
            
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            
            # Create tags for each token (using BIO scheme)
            # Initially all tokens are "outside" any entity
            tags = ['O'] * len(tokens)
            
            # Label entities using BIO scheme
            for entity in entities:
                start, end = entity['start'], entity['end']
                entity_text = text[start:end]
                
                # Skip empty entity text (prevents crashes)
                if not entity_text.strip():
                    continue
                    
                # Debug entity tokenization
                if i < 3:
                    entity_tokens = self.tokenizer.tokenize(entity_text)
                    print(f"  Entity '{entity_text}' tokenized as: {entity_tokens}")
                
                # Tokenize just the entity text
                entity_tokens = self.tokenizer.tokenize(entity_text)
                
                # Find where these tokens appear in the full tokenized text
                # This handles subword tokenization
                for j in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[j:j+len(entity_tokens)] == entity_tokens:
                        # Mark as B-LABEL for first token, I-LABEL for rest
                        if entity['label'] == 'DRUG':
                            tags[j] = 'B-DRUG'
                            for k in range(1, len(entity_tokens)):
                                tags[j+k] = 'I-DRUG'
                        elif entity['label'] == 'ADE':
                            tags[j] = 'B-ADE'
                            for k in range(1, len(entity_tokens)):
                                tags[j+k] = 'I-ADE'
            
            # Add special synthetic data for cases where we have text but no entities
            if len(entities) == 0 and text.strip():
                # Look for common drug and ADE patterns in the text
                import re
                
                # Check for common drug names
                drug_patterns = [
                    r'\b(?:lisinopril|enalapril|captopril|ramipril)\b',
                    r'\b(?:metoprolol|atenolol|propranolol|carvedilol)\b',
                    r'\b(?:amlodipine|nifedipine|diltiazem|verapamil)\b',
                    r'\b(?:hydrochlorothiazide|furosemide|spironolactone)\b',
                    r'\b(?:metformin|glyburide|glipizide|insulin)\b',
                    r'\b(?:simvastatin|atorvastatin|rosuvastatin|pravastatin)\b',
                    r'\b(?:aspirin|clopidogrel|warfarin|heparin|apixaban)\b',
                    r'\b(?:amoxicillin|azithromycin|ciprofloxacin|doxycycline)\b',
                    r'\b(?:ibuprofen|naproxen|acetaminophen|morphine|oxycodone)\b'
                ]
                
                # Check for common adverse effects
                ade_patterns = [
                    r'\b(?:cough|headache|dizziness|nausea|vomiting|diarrhea)\b',
                    r'\b(?:rash|itching|swelling|hives|angioedema)\b',
                    r'\b(?:fatigue|lethargy|drowsiness|insomnia|sleep\s+disturbance)\b',
                    r'\b(?:muscle\s+pain|muscle\s+ache|myalgia|arthralgia|joint\s+pain)\b',
                    r'\b(?:shortness\s+of\s+breath|dyspnea|wheezing)\b',
                    r'\b(?:palpitation|arrhythmia|tachycardia|bradycardia)\b'
                ]
                
                # Find all drug mentions
                for pattern in drug_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        drug_text = match.group(0)
                        drug_tokens = self.tokenizer.tokenize(drug_text)
                        
                        # Try to find the drug in tokenized text
                        for j in range(len(tokens) - len(drug_tokens) + 1):
                            drug_match = True
                            for k, dt in enumerate(drug_tokens):
                                if j+k >= len(tokens) or tokens[j+k].lower() != dt.lower():
                                    drug_match = False
                                    break
                            
                            if drug_match:
                                # Mark drug tokens with BIO scheme
                                tags[j] = 'B-DRUG'
                                for k in range(1, len(drug_tokens)):
                                    tags[j+k] = 'I-DRUG'
                
                # Find all ADE mentions
                for pattern in ade_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        ade_text = match.group(0)
                        ade_tokens = self.tokenizer.tokenize(ade_text)
                        
                        # Try to find the ADE in tokenized text
                        for j in range(len(tokens) - len(ade_tokens) + 1):
                            ade_match = True
                            for k, at in enumerate(ade_tokens):
                                if j+k >= len(tokens) or tokens[j+k].lower() != at.lower():
                                    ade_match = False
                                    break
                            
                            if ade_match:
                                # Mark ADE tokens with BIO scheme
                                tags[j] = 'B-ADE'
                                for k in range(1, len(ade_tokens)):
                                    tags[j+k] = 'I-ADE'
            
            # Check if we found any entities in this example
            has_entity = any(tag != 'O' for tag in tags)
            if has_entity:
                entity_examples += 1
            
            # Convert string tags to IDs
            numeric_tags = [self.tag_to_id(tag) for tag in tags]
            
            # Add to dataset
            texts.append(text)
            tags_list.append(numeric_tags)
        
        # Print statistics
        print(f"Conversion complete: {len(texts)} examples created")
        print(f"Entity statistics: {total_entities} total entities in {entity_examples} examples")
        if empty_examples > 0:
            print(f"Warning: {empty_examples} empty examples were skipped")
        
        # Check if we have any tagged entities
        if entity_examples == 0:
            print("\nWARNING: No entities found in any examples! Using synthetic data...")
            
            # Create synthetic examples with guaranteed entities
            synthetic_data = [
                ("Patient was prescribed lisinopril 10mg daily for hypertension.", 
                ["O", "O", "O", "B-DRUG", "O", "O", "O", "O"]),
                ("Started on metformin 500mg twice daily for diabetes.", 
                ["O", "O", "B-DRUG", "O", "O", "O", "O", "O"]),
                ("Patient experienced headache after taking ibuprofen.", 
                ["O", "O", "B-ADE", "O", "O", "B-DRUG", "O"]),
                ("Developed rash and itching following amoxicillin treatment.", 
                ["O", "B-ADE", "O", "B-ADE", "O", "B-DRUG", "O"]),
            ]
            
            for text, bio_tags in synthetic_data:
                texts.append(text)
                tags = [self.tag_to_id(tag) for tag in bio_tags]
                tags_list.append(tags)
                
            print(f"Added {len(synthetic_data)} synthetic examples with guaranteed entities")
        
        return texts, tags_list
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
    
    def tag_to_id(self, tag):
        return NER_LABELS.get(tag, 0)


#############################################################
# Data Augmentation for Better Training
#############################################################
def augment_training_data(texts, tags, num_augmentations=2):
    """
    Augment training data to improve model performance.
    
    Args:
        texts (list): Original text examples
        tags (list): Original tag sequences
        num_augmentations (int): Number of augmentations per example
        
    Returns:
        tuple: (augmented_texts, augmented_tags)
    """
    augmented_texts = texts.copy()
    augmented_tags = tags.copy()
    
    for i in range(len(texts)):
        text = texts[i]
        tag_seq = tags[i]
        
        # Skip if no entities present (all tags are "O")
        if all(tag == 0 for tag in tag_seq):
            continue
        
        # Find entity spans
        entity_spans = []
        current_span = None
        
        for j, tag_id in enumerate(tag_seq):
            tag = ID_TO_LABEL.get(tag_id, "O")
            
            if tag.startswith("B-"):
                # New entity starts
                if current_span:
                    entity_spans.append(current_span)
                current_span = {"start": j, "end": j+1, "type": tag[2:]}
            elif tag.startswith("I-") and current_span and current_span["type"] == tag[2:]:
                # Continue current entity
                current_span["end"] = j+1
            elif tag == "O":
                # Outside any entity
                if current_span:
                    entity_spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            entity_spans.append(current_span)
        
        # Skip if no entity spans found
        if not entity_spans:
            continue
        
        # Create augmentations
        for _ in range(num_augmentations):
            # Randomly delete some non-entity tokens (simple augmentation)
            aug_text = text
            aug_tags = tag_seq.copy()
            
            # TODO: Implement more sophisticated augmentation techniques
            # This is just a placeholder for the concept
            
            augmented_texts.append(aug_text)
            augmented_tags.append(aug_tags)
    
    return augmented_texts, augmented_tags


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
# Improved Training with Learning Rate Finder and Early Stopping
#############################################################
def find_optimal_learning_rate(model, train_loader, device, start_lr=1e-7, end_lr=1e-2, num_steps=100):
    """
    Improved version of the learning rate finder with better min rate handling.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        device: Computation device
        start_lr (float): Starting learning rate
        end_lr (float): Ending learning rate
        num_steps (int): Number of steps to try
        
    Returns:
        float: Optimal learning rate
    """
    print("Running learning rate finder...")
    
    # Save initial model weights
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    
    # Calculate learning rate multiplier
    mult = (end_lr / start_lr) ** (1 / num_steps)
    
    # Initialize lists to track learning rates and losses
    lrs = []
    losses = []
    
    # Only use a portion of the training data for speed
    max_steps = min(num_steps * 2, len(train_loader))
    
    # Iterate through batches
    for i, batch in enumerate(train_loader):
        if i >= max_steps:
            break
            
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Record learning rate and loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                lrs.append(optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                
                # Print progress (loss decreased or increased a lot)
                if i > 0 and (losses[-1] < losses[-2] * 0.8 or losses[-1] > losses[-2] * 1.5):
                    print(f"  Step {i}/{max_steps}: lr={lrs[-1]:.1e}, loss={losses[-1]:.4f}")
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= mult
            
        except Exception as e:
            print(f"Error in learning rate search: {e}")
            continue
    
    # Restore initial model weights
    model.load_state_dict(initial_state)
    
    # If not enough data points, return a default value
    if len(losses) < 10:
        print("Not enough valid loss points to determine optimal learning rate. Using default.")
        return 5e-5
    
    # Smoothing losses
    print(f"Collected {len(losses)} loss values across learning rates from {min(lrs):.1e} to {max(lrs):.1e}")
    
    # Find the point of the steepest decrease in the loss
    smoothed_losses = []
    window_size = min(5, len(losses) // 5)  # Adjust window size based on data
    
    for i in range(len(losses)):
        if i < window_size:
            smoothed_losses.append(losses[i])
        else:
            smoothed_losses.append(sum(losses[i-window_size:i]) / window_size)
    
    # Calculate gradients
    if len(smoothed_losses) <= 1:
        return 5e-5  # Default
        
    gradients = [(smoothed_losses[i+1] - smoothed_losses[i]) / (lrs[i+1] - lrs[i]) 
                 for i in range(len(smoothed_losses)-1)]
    
    # Skip the first few points (too noisy)
    start_idx = min(5, len(gradients) // 4)
    gradients = gradients[start_idx:]
    
    if not gradients:
        return 5e-5  # Default
    
    # Find the steepest descent
    try:
        steepest_idx = gradients.index(min(gradients))
        optimal_lr = lrs[steepest_idx + start_idx]
        
        # Typically we want a slightly lower learning rate than the steepest point
        optimal_lr = optimal_lr * 0.1
        print(f"Identified optimal learning rate: {optimal_lr:.1e}")
        
        return max(optimal_lr, 1e-6)  # Lower bound to prevent extreme values
        
    except Exception as e:
        print(f"Error finding optimal point: {e}")
        return 5e-5  # Default fallback
    

#############################################################
# Handling Imbalanced Data
#############################################################
def calculate_class_weights(tags_list):
    """
    Calculate class weights to handle imbalanced data.
    
    Args:
        tags_list (list): List of tag sequences
        
    Returns:
        dict: Class weights for each tag ID
    """
    # Flatten all tag sequences
    all_tags = []
    for tags in tags_list:
        all_tags.extend(tags)
    
    # Count occurrences
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Calculate weights (inverse of frequency)
    total_tags = len(all_tags)
    class_weights = {}
    
    for tag_id, count in tag_counts.items():
        # Skip padding tag (0)
        if tag_id == 0:
            class_weights[tag_id] = 0.0
        else:
            class_weights[tag_id] = total_tags / (count * len(tag_counts))
    
    return class_weights


#############################################################
# ModernBERT Fine-tuning Class
#############################################################

class ModernBERTFineTuner:
    """
    Base class for fine-tuning ModernBERT.
    """
    def __init__(self, model_name=MODEL_NAME):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(NER_LABELS))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def prepare_ner_data(self, ner_data):
        # Optionally, delegate to ADEDatasetProcessor or implement a simple version
        processor = ADEDatasetProcessor(None, None, self.tokenizer)
        return processor.prepare_ner_data(ner_data)
    def evaluate_base_model(self, test_texts, test_tags):
        # Evaluate the base model on the test set
        test_dataset = ADEDataset(test_texts, test_tags, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return evaluate_model(self.model, test_loader, device)

class EnhancedModernBERTFineTuner(ModernBERTFineTuner):
    def train(self, train_texts, train_tags, val_texts=None, val_tags=None, 
              epochs=5, batch_size=16, learning_rate=5e-5, class_weights=None,
              patience=2, gradient_accumulation_steps=1):
        """
        Enhanced training method with fixes for small datasets and proper scheduler selection.
        """
        print(f"\n{'='*20} TRAINING ModernBERT (ENHANCED) {'='*20}")
        print(f"Training on {len(train_texts)} examples for up to {epochs} epochs")
        if val_texts:
            print(f"Validation set size: {len(val_texts)} examples")
        
        # Create datasets
        train_dataset = ADEDataset(train_texts, train_tags, self.tokenizer)
        if val_texts and val_tags:
            val_dataset = ADEDataset(val_texts, val_tags, self.tokenizer)
        
        # Check if dataset is too small - adjust batch size if needed
        if len(train_dataset) < batch_size:
            old_batch_size = batch_size
            batch_size = max(1, len(train_dataset) // 2)
            print(f"WARNING: Training set too small. Reducing batch size from {old_batch_size} to {batch_size}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_texts and val_tags:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Calculate total steps with safety check
        total_steps = max(1, len(train_loader) * epochs // gradient_accumulation_steps)
        print(f"Training configuration: {len(train_loader)} batches per epoch × {epochs} epochs = {total_steps} total steps")
        
        # Setup optimizer 
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Choose appropriate scheduler based on dataset size
        if total_steps <= 2:
            print("Dataset too small for OneCycleLR, using constant learning rate")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        else:
            print(f"Using OneCycleLR scheduler with {total_steps} total steps")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='linear'
            )
        
        # Create loss function with class weights if provided
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if class_weights:
            print("Using class weights for training")
            weights = torch.tensor([class_weights.get(i, 1.0) for i in range(len(NER_LABELS))]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        # Determine device
        print(f"Training on device: {device}")
        self.model.to(device)
        
        # Initialize tracking variables
        best_val_f1 = 0.0
        best_model_state = None
        epochs_without_improvement = 0
        
        # Initialize metrics tracking
        training_metrics = {
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.model.train()
            train_loss = 0
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            training_metrics['learning_rates'].append(current_lr)
            print(f"Learning rate: {current_lr:.6f}")
            
            # Training phase
            optimizer.zero_grad()  # Zero gradients once at the beginning
            
            for step, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                logits = outputs.logits
                loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
                loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                
                # Backward pass
                loss.backward()
                
                # Track loss
                train_loss += loss.item() * gradient_accumulation_steps
                
                # Update weights after accumulation or at the end of epoch
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Print progress
                if (step + 1) % (gradient_accumulation_steps * 5) == 0 or step == len(train_loader) - 1:
                    print(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            training_metrics['epoch_losses'].append(avg_train_loss)
            print(f"Training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_texts and val_tags:
                val_metrics = self.evaluate(val_loader, device, criterion)
                val_loss = val_metrics['loss']
                f1 = val_metrics['f1']
                precision = val_metrics['precision']
                recall = val_metrics['recall']
                
                # Store metrics
                training_metrics['val_losses'].append(val_loss)
                training_metrics['f1_scores'].append(f1)
                training_metrics['precision_scores'].append(precision)
                training_metrics['recall_scores'].append(recall)
                
                print(f"Validation loss: {val_loss:.4f}")
                print(f"Validation metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                # Check for improvement and save the best model
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_model_state = self.model.state_dict().copy()
                    print(f"New best model with F1: {f1:.4f}")
                    epochs_without_improvement = 0
                    
                    # Save the best model
                    self.model.save_pretrained("./best_modernbert_ade_model")
                    self.tokenizer.save_pretrained("./best_modernbert_ade_model")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load the best model (if we found one)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation F1: {best_val_f1:.4f}")
        
        # Save the final fine-tuned model
        print("\nTraining complete! Saving final model.")
        final_model_path = "./modernbert_ade_extractor_final"
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Store final metrics
        if val_texts and val_tags and len(training_metrics['f1_scores']) > 0:
            training_metrics['final_f1'] = training_metrics['f1_scores'][-1]
            training_metrics['final_precision'] = training_metrics['precision_scores'][-1]
            training_metrics['final_recall'] = training_metrics['recall_scores'][-1]
        
        return self.model, self.tokenizer, training_metrics

#############################################################
# Enhanced ModernBERT Fine-tuner with Advanced Training
#############################################################
class EnhancedModernBERTFineTuner(ModernBERTFineTuner):
    """
    Enhanced version of ModernBERTFineTuner with advanced training features.
    
    This class extends the base ModernBERTFineTuner with:
    - Class weight support for imbalanced data
    - Early stopping
    - Learning rate scheduling
    - Gradient accumulation for larger batch sizes
    - Training history logging
    """
    
    def train(self, train_texts, train_tags, val_texts=None, val_tags=None, 
              epochs=5, batch_size=16, learning_rate=5e-5, class_weights=None,
              patience=2, gradient_accumulation_steps=1):
        """
        Enhanced training method with advanced features.
        
        Args:
            train_texts (list): Training text examples
            train_tags (list): Training label sequences
            val_texts (list): Validation text examples
            val_tags (list): Validation tag sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate for optimizer
            class_weights (dict): Optional weights for each class to handle imbalance
            patience (int): Early stopping patience
            gradient_accumulation_steps (int): Steps for gradient accumulation
                
        Returns:
            tuple: (model, tokenizer, metrics) - The fine-tuned model, tokenizer, and metrics
        """
        print(f"\n{'='*20} TRAINING ModernBERT (ENHANCED) {'='*20}")
        print(f"Training on {len(train_texts)} examples for up to {epochs} epochs")
        if val_texts:
            print(f"Validation set size: {len(val_texts)} examples")
        
        # Create datasets
        train_dataset = ADEDataset(train_texts, train_tags, self.tokenizer)
        if val_texts and val_tags:
            val_dataset = ADEDataset(val_texts, val_tags, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_texts and val_tags:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler - linear with warmup
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='linear'
        )
        
        # Create loss function with class weights if provided
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if class_weights:
            print("Using class weights for training")
            weights = torch.tensor([class_weights.get(i, 1.0) for i in range(len(NER_LABELS))]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        # Determine device
        print(f"Training on device: {device}")
        self.model.to(device)
        
        # Initialize tracking variables
        best_val_f1 = 0.0
        best_model_state = None
        epochs_without_improvement = 0
        
        # Initialize metrics tracking
        training_metrics = {
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.model.train()
            train_loss = 0
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            training_metrics['learning_rates'].append(current_lr)
            print(f"Learning rate: {current_lr:.6f}")
            
            # Training phase
            optimizer.zero_grad()  # Zero gradients once at the beginning
            
            for step, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                logits = outputs.logits
                loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
                loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                
                # Backward pass
                loss.backward()
                
                # Track loss
                train_loss += loss.item() * gradient_accumulation_steps
                
                # Update weights after accumulation or at the end of epoch
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Print progress
                if (step + 1) % (gradient_accumulation_steps * 10) == 0:
                    print(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            training_metrics['epoch_losses'].append(avg_train_loss)
            print(f"Training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_texts and val_tags:
                val_metrics = self.evaluate(val_loader, device, criterion)
                val_loss = val_metrics['loss']
                f1 = val_metrics['f1']
                precision = val_metrics['precision']
                recall = val_metrics['recall']
                
                # Store metrics
                training_metrics['val_losses'].append(val_loss)
                training_metrics['f1_scores'].append(f1)
                training_metrics['precision_scores'].append(precision)
                training_metrics['recall_scores'].append(recall)
                
                print(f"Validation loss: {val_loss:.4f}")
                print(f"Validation metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                # Check for improvement and save the best model
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_model_state = self.model.state_dict().copy()
                    print(f"New best model with F1: {f1:.4f}")
                    epochs_without_improvement = 0
                    
                    # Save the best model
                    self.model.save_pretrained("./best_modernbert_ade_model")
                    self.tokenizer.save_pretrained("./best_modernbert_ade_model")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load the best model (if we found one)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation F1: {best_val_f1:.4f}")
        
        # Save the final fine-tuned model
        print("\nTraining complete! Saving final model.")
        self.model.save_pretrained("./modernbert_ade_extractor_final")
        self.tokenizer.save_pretrained("./modernbert_ade_extractor_final")
        
        # Store final metrics
        if val_texts and val_tags and len(training_metrics['f1_scores']) > 0:
            training_metrics['final_f1'] = training_metrics['f1_scores'][-1]
            training_metrics['final_precision'] = training_metrics['precision_scores'][-1]
            training_metrics['final_recall'] = training_metrics['recall_scores'][-1]
        
        return self.model, self.tokenizer, training_metrics
    
    def evaluate(self, data_loader, device, criterion=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            device: Computation device
            criterion: Optional loss function
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        
        # Use default criterion if none provided
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                logits = outputs.logits
                loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
                total_loss += loss.item()
                
                # Get predictions
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
        metrics = {'loss': total_loss / len(data_loader), 'f1': 0, 'precision': 0, 'recall': 0}
        mask = np.array(y_true) != 0
        
        if mask.sum() > 0:  # Ensure we have non-padding tokens
            y_true_masked = np.array(y_true)[mask]
            y_pred_masked = np.array(y_pred)[mask]
            
            # Calculate metrics
            metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
            metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        return metrics


#############################################################
# DSPy Optimization Class with Improved Error Handling
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
    
    def evaluate_extraction(self, gold, pred, trace=None):
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
        Optimize ADE extraction using DSPy with robust error handling.
        
        Args:
            examples (list): List of DSPy examples
            
        Returns:
            ADEExtractor: Optimized or original extractor
        """
        print(f"\n{'='*20} OPTIMIZING ADE EXTRACTION {'='*20}")
        
        # Split examples into training and development sets
        train_examples, dev_examples = train_test_split(examples, test_size=0.2, random_state=42)
        print(f"Training set: {len(train_examples)} examples")
        print(f"Development set: {len(dev_examples)} examples")
        
        # Try multiple approaches to handle DSPy API variations
        approaches = [
            # Approach 1: Just trainset
            lambda: dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=3
            ).compile(self.ade_extractor, trainset=train_examples),
            
            # Approach 2: Both trainset and valset
            lambda: dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=3
            ).compile(self.ade_extractor, trainset=train_examples, valset=dev_examples),
            
            # Approach 3: Just module and dataset parameter
            lambda: dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=3
            ).compile(self.ade_extractor, dataset=train_examples),
            
            # Approach 4: No parameters (minimal)
            lambda: dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=3
            ).compile(self.ade_extractor)
        ]
        
        print("Running DSPy optimization...")
        
        # Try each approach until one works
        for i, approach in enumerate(approaches):
            try:
                optimized_extractor = approach()
                print(f"DSPy optimization complete with approach {i+1}!")
                return optimized_extractor
            except Exception as e:
                print(f"Approach {i+1} failed: {e}")
        
        # If all approaches fail, fall back to the original extractor
        print("All optimization approaches failed. Falling back to the original non-optimized extractor")
        return self.ade_extractor


#############################################################
# Advanced Visualization and Analysis
#############################################################
def analyze_and_visualize_results(pipeline_results, output_dir="./analysis"):
    """
    Generate comprehensive visualizations and analysis of model performance.
    
    Args:
        pipeline_results (dict): Results from the ADE extraction pipeline
        output_dir (str): Directory to save visualizations
        
    Returns:
        dict: Analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    base_metrics = pipeline_results["metrics"]["base_model"]
    finetuned_metrics = pipeline_results["metrics"]["finetuned_model"]
    training_metrics = pipeline_results["metrics"]["training"]
    
    # 1. Model Performance Comparison
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
        if i == 0:  # Base model
            plt.text(i - width, base_metrics['f1'] + 0.02, f"{base_metrics['f1']:.3f}", ha='center')
            plt.text(i, base_metrics['precision'] + 0.02, f"{base_metrics['precision']:.3f}", ha='center')
            plt.text(i + width, base_metrics['recall'] + 0.02, f"{base_metrics['recall']:.3f}", ha='center')
        else:  # Fine-tuned model
            plt.text(i - width, finetuned_metrics['f1'] + 0.02, f"{finetuned_metrics['f1']:.3f}", ha='center')
            plt.text(i, finetuned_metrics['precision'] + 0.02, f"{finetuned_metrics['precision']:.3f}", ha='center')
            plt.text(i + width, finetuned_metrics['recall'] + 0.02, f"{finetuned_metrics['recall']:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    print(f"Model comparison chart saved to '{output_dir}/model_comparison.png'")
    
    # 2. Learning Curves
    if 'epoch_losses' in training_metrics and len(training_metrics['epoch_losses']) > 0:
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        
        epochs = range(1, len(training_metrics['epoch_losses']) + 1)
        
        plt.plot(epochs, training_metrics['epoch_losses'], 'b-', label='Training Loss')
        if 'val_losses' in training_metrics and len(training_metrics['val_losses']) > 0:
            plt.plot(epochs, training_metrics['val_losses'], 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot performance metrics
        if 'f1_scores' in training_metrics and len(training_metrics['f1_scores']) > 0:
            plt.subplot(2, 1, 2)
            plt.plot(epochs, training_metrics['f1_scores'], 'g-', label='F1 Score')
            plt.plot(epochs, training_metrics['precision_scores'], 'b-', label='Precision')
            plt.plot(epochs, training_metrics['recall_scores'], 'r-', label='Recall')
            
            plt.title('Training Metrics by Epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_curves.png")
        print(f"Learning curves saved to '{output_dir}/learning_curves.png'")
    
    # 3. Extract and visualize entity distributions
    extract_data = pipeline_results["processed_data"]["extracted_data"]
    
    # Count drug and ADE frequencies
    drug_counts = {}
    ade_counts = {}
    
    for record in extract_data:
        for drug in record['drugs']:
            drug_counts[drug] = drug_counts.get(drug, 0) + 1
        
        for ade in record['adverse_events']:
            ade_counts[ade] = ade_counts.get(ade, 0) + 1
    
    # Sort by frequency
    sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_ades = sorted(ade_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 15 drugs
    plt.figure(figsize=(14, 8))
    
    if sorted_drugs:
        plt.subplot(1, 2, 1)
        top_n = min(15, len(sorted_drugs))
        top_drugs = [x[0] for x in sorted_drugs[:top_n]]
        drug_freqs = [x[1] for x in sorted_drugs[:top_n]]
        
        plt.barh(top_drugs, drug_freqs, color='skyblue')
        plt.title(f'Top {top_n} Drugs Mentioned')
        plt.xlabel('Frequency')
        plt.tight_layout()
    
    # Plot top 15 ADEs
    if sorted_ades:
        plt.subplot(1, 2, 2)
        top_n = min(15, len(sorted_ades))
        top_ades = [x[0] for x in sorted_ades[:top_n]]
        ade_freqs = [x[1] for x in sorted_ades[:top_n]]
        
        plt.barh(top_ades, ade_freqs, color='salmon')
        plt.title(f'Top {top_n} Adverse Events Mentioned')
        plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entity_distribution.png")
    print(f"Entity distribution saved to '{output_dir}/entity_distribution.png'")
    
    # 4. Create detailed metrics report
    with open(f"{output_dir}/model_evaluation_report.json", "w") as f:
        json.dump({
            'base_model': base_metrics,
            'finetuned_model': finetuned_metrics,
            'improvements': {
                'f1': finetuned_metrics['f1'] - base_metrics['f1'],
                'precision': finetuned_metrics['precision'] - base_metrics['precision'],
                'recall': finetuned_metrics['recall'] - base_metrics['recall']
            },
            'training_metrics': training_metrics,
            'entities': {
                'drug_frequencies': {k: v for k, v in sorted_drugs},
                'ade_frequencies': {k: v for k, v in sorted_ades}
            }
        }, f, indent=2)
    print(f"Detailed evaluation report saved to '{output_dir}/model_evaluation_report.json'")
    
    # 5. Generate interactive visualization if plotly is available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create interactive learning curves
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Training and Validation Loss", "Performance Metrics"))
        
        # Add traces for losses
        fig.add_trace(
            go.Scatter(x=list(epochs), y=training_metrics['epoch_losses'], 
                      mode='lines+markers', name='Training Loss'),
            row=1, col=1
        )
        
        if 'val_losses' in training_metrics and len(training_metrics['val_losses']) > 0:
            fig.add_trace(
                go.Scatter(x=list(epochs), y=training_metrics['val_losses'], 
                          mode='lines+markers', name='Validation Loss'),
                row=1, col=1
            )
        
        # Add traces for metrics
        if 'f1_scores' in training_metrics and len(training_metrics['f1_scores']) > 0:
            fig.add_trace(
                go.Scatter(x=list(epochs), y=training_metrics['f1_scores'], 
                          mode='lines+markers', name='F1 Score'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=training_metrics['precision_scores'], 
                          mode='lines+markers', name='Precision'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=training_metrics['recall_scores'], 
                          mode='lines+markers', name='Recall'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="ModernBERT Training Progress",
            height=800,
            width=1000
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=2, col=1)
        
        # Save as HTML
        fig.write_html(f"{output_dir}/interactive_learning_curves.html")
        print(f"Interactive learning curves saved to '{output_dir}/interactive_learning_curves.html'")
        
    except ImportError:
        print("Plotly not installed. Skipping interactive visualizations.")
    
    return {
        'base_metrics': base_metrics,
        'finetuned_metrics': finetuned_metrics,
        'entity_counts': {
            'drugs': len(drug_counts),
            'adverse_events': len(ade_counts)
        }
    }


#############################################################
# Helper function for model evaluation
#############################################################
def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        device: Computation device
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in data_loader:
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
    metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'loss': test_loss/len(data_loader)}
    mask = np.array(y_true) != 0
    
    if mask.sum() > 0:  # Ensure we have non-padding tokens
        y_true_masked = np.array(y_true)[mask]
        y_pred_masked = np.array(y_pred)[mask]
        
        # Calculate metrics
        metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
        metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        print(f"Model Evaluation - Loss: {metrics['loss']:.4f}, F1: {metrics['f1']:.4f}, " 
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    else:
        print("No non-padding tokens found in evaluation set")
    
    return metrics


#############################################################
# Enhanced Pipeline for Large-Scale Processing
#############################################################
def run_scalable_ade_extraction_pipeline(medical_notes, batch_size=100, use_dspy=True):
    """
    Enhanced ADE extraction pipeline with DSPy ablation option.
    When DSPy is disabled, still uses GPT-4o-mini directly, just without optimization.
    """
    print(f"\n{'='*20} STARTING SCALABLE ADE EXTRACTION PIPELINE {'='*20}")
    print(f"Settings: batch_size={batch_size}, use_dspy={use_dspy}")
    
    # Configure OpenAI access - needed regardless of DSPy usage
    # This ensures the API key is loaded even if DSPy is disabled
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY must be set in environment variables")
    
    # Configure DSPy only if optimization is enabled
    if use_dspy:
        print("Configuring DSPy with language model for optimization...")
        try:
            model_name = "openai/gpt-4o-mini"  # or "openai/gpt-4" for better results
            lm = dspy.LM(model_name)
            dspy.settings.configure(lm=lm)
            print(f"Successfully configured DSPy with model: {model_name}")
        except Exception as e:
            print(f"Error configuring DSPy: {e}")
            raise
    else:
        print("DSPy optimization disabled - will use GPT-4o-mini directly without optimization")
    
    # Process notes (unchanged from original)
    print(f"\nStep 1: Processing {len(medical_notes)} medical notes in batches...")
    notes_processor = MedicalNoteProcessor()
    processed_notes = []
    for i in range(0, len(medical_notes), batch_size):
        batch = medical_notes[i:min(i+batch_size, len(medical_notes))]
        print(f"  Processing batch {i//batch_size + 1}/{(len(medical_notes) + batch_size - 1)//batch_size} ({len(batch)} notes)")
        notes_processor.load_notes(batch)
        batch_processed = notes_processor.preprocess_notes()
        processed_notes.extend(batch_processed)
    print(f"Processed {len(processed_notes)} medical notes")
    
    # Step 2: Initialize appropriate ADE extractor
    print("\nStep 2: Initializing ADE extractor...")
    if use_dspy:
        # With DSPy enabled - use the DSPy-based extractor that will be optimized later
        ade_extractor = ADEExtractor()
        print("Using DSPy-based ADE extractor with optimization")
    else:
        # With DSPy disabled - use direct LLM extraction without optimization
        ade_extractor = DirectLLMExtractor(model_name="gpt-4o-mini")
        print("Using GPT-4o-mini directly without DSPy optimization")
    
    # Step 3: Extract ADEs from notes in batches
    print("\nStep 3: Extracting ADEs from notes in batches...")
    extracted_data = []
    
    for i in range(0, len(processed_notes), batch_size):
        batch = processed_notes[i:min(i+batch_size, len(processed_notes))]
        print(f"  Extracting from batch {i//batch_size + 1}/{(len(processed_notes) + batch_size - 1)//batch_size} ({len(batch)} notes)")
        
        batch_extracted = []
        for note_idx, note in enumerate(batch):
            try:
                print(f"    Processing note {note_idx+1}/{len(batch)}...")
                # Extract from the note
                extraction_result = ade_extractor(note)
                
                # Create a structured record
                record = {
                    'text': note,
                    'drugs': extraction_result.drugs,
                    'adverse_events': extraction_result.adverse_events,
                    'drug_ade_pairs': extraction_result.drug_ade_pairs
                }
                
                # Show extraction results for the first few notes
                if note_idx < 3:
                    print(f"      Found {len(record['drugs'])} drugs and {len(record['adverse_events'])} adverse events")
                    if record['drugs']:
                        print(f"      Drugs: {record['drugs']}")
                    if record['adverse_events']:
                        print(f"      ADEs: {record['adverse_events']}")
                
                batch_extracted.append(record)
            except Exception as e:
                print(f"Error extracting ADEs from note {note_idx+1}: {e}")
                # Add empty record to maintain alignment
                record = {
                    'text': note,
                    'drugs': [],
                    'adverse_events': [],
                    'drug_ade_pairs': []
                }
                batch_extracted.append(record)
        
        extracted_data.extend(batch_extracted)
    
    # Step 4: Preparing data for NER and relation extraction...
    print("\nStep 4: Preparing data for NER and relation extraction...")
    ner_data = []
    relation_data = []
    for record in extracted_data:
        text = record['text']
        entities = []
        # Add drug entities
        for drug in record['drugs']:
            for match in re.finditer(re.escape(drug), text):
                entities.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'DRUG'
                })
        # Add ADE entities
        for ade in record['adverse_events']:
            for match in re.finditer(re.escape(ade), text):
                entities.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'ADE'
                })
        ner_data.append({
            'text': text,
            'entities': entities
        })
    # Optionally, relation_data can be filled similarly if needed
    
    # Step 5: Optimize DSPy extractor (if enabled) - otherwise skip
    if use_dspy:
        print("\nStep 5: Optimizing ADE extraction with DSPy...")
        optimizer = ADEOptimizer(ade_extractor)
        
        # If dataset is very large, use a sample for optimization
        optimization_sample_size = min(500, len(extracted_data))
        optimization_sample = extracted_data[:optimization_sample_size]
        print(f"Using a sample of {optimization_sample_size} examples for DSPy optimization")
        
        examples = optimizer.prepare_examples(optimization_sample)
        optimized_extractor = optimizer.optimize(examples)
    else:
        print("\nStep 5: DSPy optimization skipped (ablation mode)")
        optimized_extractor = ade_extractor
    
    # Step 6: Prepare for ModernBERT fine-tuning
    print("\nStep 6: Preparing data for ModernBERT fine-tuning...")
    finetuner = ModernBERTFineTuner()
    
    # Process NER data in batches
    all_texts = []
    all_tags = []
    
    for i in range(0, len(ner_data), batch_size):
        batch = ner_data[i:min(i+batch_size, len(ner_data))]
        print(f"  Converting batch {i//batch_size + 1}/{(len(ner_data) + batch_size - 1)//batch_size} ({len(batch)} examples)")
        
        batch_texts, batch_tags = finetuner.prepare_ner_data(batch)
        all_texts.extend(batch_texts)
        all_tags.extend(batch_tags)
    
    # Print sample of tokenized data
    if all_texts and all_tags:
        sample_size = min(3, len(all_texts))
        print("\nSample of tokenized training data:")
        for i in range(sample_size):
            text = all_texts[i][:50] + "..." if len(all_texts[i]) > 50 else all_texts[i]
            # Convert numeric tags to readable format
            readable_tags = [ID_TO_LABEL.get(tag, "UNK") for tag in all_tags[i][:10]]
            print(f"  [{i+1}] Text: {text}")
            print(f"      Tags: {readable_tags}...")
    
    # Step 7: Add data augmentation for better training
    print("\nStep 7: Augmenting training data...")
    augmented_texts, augmented_tags = augment_training_data(all_texts, all_tags)
    print(f"Added {len(augmented_texts) - len(all_texts)} augmented examples")
    
    # Step 8: Split data for training/validation/test with stratification
    print("\nStep 8: Splitting data for training, validation, and testing...")
    
    # Create a balanced split based on entity presence
    has_entity = [1 if any(tag != 0 for tag in tags) else 0 for tags in augmented_tags]
    
    # Split into train and temp sets
    train_texts, temp_texts, train_tags, temp_tags = train_test_split(
        augmented_texts, augmented_tags, 
        test_size=0.2, random_state=42,
        stratify=has_entity  # Ensure both splits have similar entity distributions
    )
    
    # Further split temp into validation and test sets
    val_texts, test_texts, val_tags, test_tags = train_test_split(
        temp_texts, temp_tags, 
        test_size=0.5, random_state=42
    )
    
    print(f"Training set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_texts)} examples")
    print(f"Test set: {len(test_texts)} examples")
    
    # Step 9: Calculate class weights to handle imbalanced data
    print("\nStep 9: Calculating class weights for imbalanced data...")
    class_weights = calculate_class_weights(train_tags)
    print(f"Class weights: {class_weights}")
    
    # Step 10: Evaluate base ModernBERT model
    print("\nStep 10: Evaluating base ModernBERT model...")
    base_metrics = finetuner.evaluate_base_model(test_texts, test_tags)
    
    # Step 11: Find optimal learning rate with safety bounds
    print("\nStep 11: Finding optimal learning rate...")
    train_dataset = ADEDataset(train_texts[:100], train_tags[:100], finetuner.tokenizer)  # Use a subset for speed
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        optimal_lr = find_optimal_learning_rate(finetuner.model, train_loader, device)
        # Add safety bounds to prevent extremely low or high learning rates
        if optimal_lr < 1e-6:
            print(f"Learning rate {optimal_lr:.1e} too low, using default 5e-5 instead")
            optimal_lr = 5e-5
        elif optimal_lr > 1e-3:
            print(f"Learning rate {optimal_lr:.1e} too high, using 1e-4 instead")
            optimal_lr = 1e-4
        else:
            print(f"Using optimal learning rate: {optimal_lr:.1e}")
    except Exception as e:
        print(f"Error finding optimal learning rate: {e}")
        optimal_lr = 5e-5  # Default fallback
        print(f"Using default learning rate: {optimal_lr:.1e}")
    
    # Step 12: Fine-tune ModernBERT with the enhanced train method
    print("\nStep 12: Fine-tuning ModernBERT for ADE extraction...")
    enhanced_finetuner = EnhancedModernBERTFineTuner(MODEL_NAME)
    model, tokenizer, training_metrics = enhanced_finetuner.train(
        train_texts, train_tags, val_texts, val_tags, 
        epochs=5,  # More epochs for better training
        batch_size=16,
        learning_rate=optimal_lr,
        class_weights=class_weights,
        patience=2
    )
    
    # Step 13: Evaluate the fine-tuned model with improved entity extraction
    print("\nStep 13: Evaluating fine-tuned ModernBERT model...")
    
    # Create test dataset and dataloader
    test_dataset = ADEDataset(test_texts, test_tags, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluation metrics
    finetuned_metrics = evaluate_model(model, test_loader, device)
    
    # Compare with base model metrics
    f1_improvement = finetuned_metrics['f1'] - base_metrics['f1']
    precision_improvement = finetuned_metrics['precision'] - base_metrics['precision']
    recall_improvement = finetuned_metrics['recall'] - base_metrics['recall']
    
    print("\nModernBERT Performance Comparison (Before vs. After Fine-tuning):")
    print(f"F1 Score:   {base_metrics['f1']:.4f} → {finetuned_metrics['f1']:.4f} (Change: {f1_improvement:+.4f})")
    print(f"Precision:  {base_metrics['precision']:.4f} → {finetuned_metrics['precision']:.4f} (Change: {precision_improvement:+.4f})")
    print(f"Recall:     {base_metrics['recall']:.4f} → {finetuned_metrics['recall']:.4f} (Change: {recall_improvement:+.4f})")
    
    # Step 14: Test on real examples using improved extraction
    print("\nStep 14: Testing on real examples with improved extraction...")
    
    # Define a function that uses our improved extraction
    def extract_ades_with_finetuned_model(text):
        """Wrapper for the improved entity extraction function"""
        return extract_entities_improved(text, model, tokenizer, device)
    
    # Test on a few examples and print details
    test_examples = [
        "Patient was prescribed amoxicillin 500mg TID for bacterial infection. After 2 days, patient reported skin rash and itching.",
        "The patient developed severe headache after taking ibuprofen for fever. Symptoms resolved after discontinuing the medication.",
        "Patient on metformin 1000mg BID experienced nausea and abdominal pain. Symptoms improved when dose was reduced."
    ]
    
    print("\nDetailed extraction results on test examples:")
    test_results = []
    for i, example in enumerate(test_examples):
        print(f"\nExample {i+1}: {example}")
        result = extract_ades_with_finetuned_model(example)
        test_results.append(result)
    
    print("\nScalable pipeline complete!")
    
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
        },
        "dspy_enabled": use_dspy,
        "test_results": test_results
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


def load_texts_from_jsonl(jsonl_path):
    """
    Load the 'text' field from each line of a JSONL file.
    Args:
        jsonl_path (str): Path to the .jsonl file
    Returns:
        list: List of text strings
    """
    texts = []
    import json
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    texts.append(obj['text'])
            except Exception as e:
                print(f"Error parsing line: {e}")
    return texts


def main():
    """Main execution function with extensive diagnostics."""
    print(f"\n{'='*50}")
    print(f"{'ADVERSE DRUG EVENT EXTRACTION WITH ModernBERT':^50}")
    print(f"{'='*50}")
    # Load notes from data/train.jsonl and data/test.jsonl
    try:
        train_path = "data/train.jsonl"
        test_path = "data/test.jsonl"
        print(f"Loading training data from {train_path}")
        train_notes = load_texts_from_jsonl(train_path)
        print(f"Loading test data from {test_path}")
        test_notes = load_texts_from_jsonl(test_path)
        print(f"Loaded {len(train_notes)} training notes and {len(test_notes)} test notes.")
        # Clean and preprocess the notes
        medical_notes = train_notes  # For pipeline, use train_notes
        print(f"Sample of training notes:")
        sample_size = min(3, len(medical_notes))
        for i in range(sample_size):
            print(f"  [train {i+1}] {medical_notes[i][:100]}...")
    except Exception as e:
        print(f"Error loading train/test jsonl files: {e}")
        print("Using example notes instead...")
        medical_notes = [
            "Patient was prescribed amoxicillin 500mg TID for bacterial infection. After 2 days, patient reported skin rash and itching.",
            "The patient developed severe headache after taking ibuprofen for fever. Symptoms resolved after discontinuing the medication.",
            "Patient on metformin 1000mg BID experienced nausea and abdominal pain. Symptoms improved when dose was reduced.",
            "After starting on lisinopril, patient complained of persistent dry cough that did not resolve until medication was changed."
        ]
    # Use only the first 1000 records for training
    medical_notes = medical_notes[:2000]
    print(f"\nUsing {len(medical_notes)} notes for training (first 1000 records)")

    # Run pipeline with DSPy (optimized LLM extraction)
    print("\n\n======== RUNNING PIPELINE WITH DSPy OPTIMIZATION (LLM extraction improved) ========")
    try:
        pipeline_results_dspy = run_scalable_ade_extraction_pipeline(medical_notes, batch_size=100, use_dspy=True)
    except Exception as e:
        print(f"Pipeline execution with DSPy failed: {e}")
        import traceback
        traceback.print_exc()
        pipeline_results_dspy = None

    # Run pipeline without DSPy (direct LLM extraction)
    print("\n\n======== RUNNING PIPELINE WITHOUT DSPy (Direct LLM extraction, no optimization) ========")
    try:
        pipeline_results_nodspy = run_scalable_ade_extraction_pipeline(medical_notes, batch_size=100, use_dspy=False)
    except Exception as e:
        print(f"Pipeline execution without DSPy failed: {e}")
        import traceback
        traceback.print_exc()
        pipeline_results_nodspy = None

    # Compare extraction metrics if both runs succeeded
    if pipeline_results_dspy and pipeline_results_nodspy:
        print("\n\n======== COMPARISON OF LLM EXTRACTION: DSPy vs. Direct LLM ========")
        dspy_metrics = pipeline_results_dspy["metrics"]["finetuned_model"]
        nodspy_metrics = pipeline_results_nodspy["metrics"]["finetuned_model"]
        print(f"F1 Score:   DSPy={dspy_metrics['f1']:.4f} | Direct LLM={nodspy_metrics['f1']:.4f} | Improvement: {dspy_metrics['f1']-nodspy_metrics['f1']:+.4f}")
        print(f"Precision:  DSPy={dspy_metrics['precision']:.4f} | Direct LLM={nodspy_metrics['precision']:.4f} | Improvement: {dspy_metrics['precision']-nodspy_metrics['precision']:+.4f}")
        print(f"Recall:     DSPy={dspy_metrics['recall']:.4f} | Direct LLM={nodspy_metrics['recall']:.4f} | Improvement: {dspy_metrics['recall']-nodspy_metrics['recall']:+.4f}")
        print("\nLLM extraction with DSPy optimization should show improved metrics over direct LLM extraction.")
    else:
        print("\nCould not compare DSPy and direct LLM extraction due to errors in one or both runs.")





#############################################################
# Program Entry Point
#############################################################
if __name__ == "__main__":
    """
    Entry point when running the script directly.
    
    This ensures the main() function is only called when
    the script is executed, not when imported.
    """
    main()
