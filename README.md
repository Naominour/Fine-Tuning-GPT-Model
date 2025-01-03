# Fine-Tuning GPT Model for Medical Applications

This project creates a language model that generates medical text. It has three main parts: training the model on medical data, testing its performance, and deploying it through a simple Flask web app with a user-friendly interface.

![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![PyTorch](https://img.shields.io/badge/Skill-PyTorch-blueviolet)
![Transformers](https://img.shields.io/badge/Skill-Deep%20Learning-orange)
![Generative AI](https://img.shields.io/badge/Skill-Generative%20AI-yellow)
![GPT](https://img.shields.io/badge/Skill-GPT-green)
![LLaMA](https://img.shields.io/badge/Skill-GPT-blue)
![Fine Tuning Pretrained Models](https://img.shields.io/badge/Skill-Fine%20Tuning%20Pretrained%20Models-orange)
![Model Deployment](https://img.shields.io/badge/Skill-Model%20Deployment-purpule)
![Tokenization](https://img.shields.io/badge/Skill-Tokenization-blue)
![Experiment Tracking Mlflow](https://img.shields.io/badge/Skill-Experiment%20Tracking%20Mlflow-yellow)
![Data Augmentation](https://img.shields.io/badge/Skill-Data%20Augmentation-red)
![Version Control](https://img.shields.io/badge/Skill-Version%20Control-white)
![High Performance Computing](https://img.shields.io/badge/Skill-High%20Performance%20Computing-orange)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-blue)

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

## Installation and Setup
**Prerequisites**
• Python 3.8+
• GPU with CUDA (optional but recommended)
• Libraries: torch, transformers, flask, mlflow, tiktoken, etc.

**Setup Instructions**
Clone the repository:
```
git clone https://github.com/your-repo/medical-text-generator.git
cd medical-text-generator
```
Install dependencies:
```
pip install -r requirements.txt
```

Download or prepare the medical dataset and place it in the medical_dataset_cache directory.

## How to Run the Project
**1. Train the Model**
```
python train_gpt2.py
```
Adjust hyperparameters in the script as needed.
Monitor training progress via MLflow or logs in the log directory.
**2. Test the Model**
```
python test.py
```
Ensures the final training loss is below the specified threshold.

**3. Start the Application**
```
python app.py
```
Access the web interface at http://localhost:5000 in your browser.

**4. Use the Web Interface**
Enter a medical text prompt in the input box.
Click "Generate" to see the model's completion.
