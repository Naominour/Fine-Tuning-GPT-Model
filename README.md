# Fine-Tuning GPT Model for Medical Applications

This project creates a language model that generates medical text. It has three main parts: training the model on medical data, testing its performance, and deploying it through a simple Flask web app with a user-friendly interface.

## Table of Contents
- Project Overview
- Architecture Diagram
- Project Components
- Installation and Setup
- How to Run the Project
- Reproducing Results
- Contributing

## Project Overview
This project **fine-tunes a GPT-based language model on a medical dataset** to generate contextually accurate medical text. The key features include:

◉ Training from scratch or using a pre-trained GPT-2 model.
◉ A validation pipeline to monitor training loss.
◉ A web-based interface for generating medical text completions based on user input.

## Architecture Diagram
#### Data Flow
**Data Loader:** Prepares medical text data for training and validation.
**Training Loop:** Trains the GPT model using distributed training (if enabled).
**Validation Loop:** Evaluates the model periodically and logs metrics.

#### Model Flow
**Input:** Tokenized text input.
**Embedding Layer:** Converts tokens to embeddings.
**Transformer Blocks:** Processes embeddings through multi-head self-attention and feed-forward layers.
**Output:** Generates token probabilities and predictions.

#### Deployment Flow
**Model Loading:** Loads the fine-tuned model for inference.
**Web Interface:** Provides an input box for text prompts.
**Text Generation:** Generates completions using top-k sampling and temperature adjustment.

### Project Components
**1. Model Training (train_gpt2.py)**
- Implements the GPT model, data loader, and distributed training setup.
- Monitors training and validation loss via logging and MLflow.
- Provides periodic model checkpoints for reproducibility.
  
**2. Testing (test.py)**
- Validates the final model loss to ensure training has converged.
- Asserts the final loss is below a threshold for deployment readiness.

**3. Deployment (app.py)**
- Hosts the model as a **REST API** using Flask.
- Includes a simple and intuitive web interface for text prompt input.
- Generates and displays medical text completions in real-time.
