from flask import Flask, render_template, request, jsonify
import torch
import tiktoken
import torch.nn.functional as F
import os
from medical import GPT, GPTConfig

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and device
model = None
device = None
enc = None

def load_model():
    global model, device, enc
    
    print("Loading model...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = GPT(GPTConfig(vocab_size=50257))
    
    # Load the saved model state
    model_path = os.path.join('medical_dataset_cache', 'saved_model.pth')
    print(f"Loading model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # If checkpoint is a dictionary with 'model' key
            model.load_state_dict(checkpoint['model'])
        else:
            # If checkpoint is the direct state dict
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print("Tokenizer initialized")
    
    return True

def generate_completion(prompt, max_new_tokens=300):
    global model, device, enc
    
    # Tokenize the input prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    
    # Generate text
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            if len(tokens[0]) > 1024:  # Prevent exceeding max context length
                tokens = tokens[:, -1024:]
                
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # Get logits for the last token
            
            # Apply temperature and top-k sampling
            probs = F.softmax(logits / 0.7, dim=-1)  # temperature = 0.7
            top_k_probs, top_k_indices = torch.topk(probs, k=50, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(top_k_probs, num_samples=1)
            idx_next = torch.gather(top_k_indices, -1, idx_next)
            
            # Append to the sequence
            tokens = torch.cat((tokens, idx_next), dim=1)
            next_token = idx_next.item()
            generated_tokens.append(next_token)
            
                        # Only stop on end token if we've generated a minimum length
            if len(generated_tokens) > 100:  # Minimum 100 tokens before considering stopping
                if next_token == enc.eot_token:
                    break
            
            # Prevent infinite loops or repetitive text
            if len(generated_tokens) > 400:  # Hard stop if too long
                break
    
            # Stop if we generate an end of text token
            if idx_next.item() == enc.eot_token:
                break
    
    # Decode the generated tokens
    generated_text = enc.decode(generated_tokens)
    return generated_text

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Text Generator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            textarea {
                width: 100%;
                height: 150px;
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                resize: vertical;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                min-height: 100px;
            }
            .loader {
                display: none;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Medical Text Generator</h1>
            <p>Enter your medical text prompt below and click "Generate" to continue the text.</p>
            
            <textarea id="prompt" placeholder="Enter your medical text here..."></textarea>
            <button onclick="generateText()">Generate</button>
            
            <div class="loader" id="loader"></div>
            <div id="result"></div>
        </div>

        <script>
        function generateText() {
            const prompt = document.getElementById('prompt').value;
            const resultDiv = document.getElementById('result');
            const loader = document.getElementById('loader');
            
            if (!prompt) {
                alert('Please enter some text first!');
                return;
            }
            
            loader.style.display = 'block';
            resultDiv.innerHTML = '';
            
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({prompt: prompt}),
            })
            .then(response => response.json())
            .then(data => {
                loader.style.display = 'none';
                resultDiv.innerHTML = '<strong>Generated Text:</strong><br>' + 
                                    prompt + '<span style="color: #007bff">' + data.generated_text + '</span>';
            })
            .catch((error) => {
                loader.style.display = 'none';
                resultDiv.innerHTML = 'Error: Unable to generate text. Please try again.';
                console.error('Error:', error);
            });
        }
        </script>
    </body>
    </html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    try:
        generated_text = generate_completion(prompt)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting application...")
    print("Current working directory:", os.getcwd())
    print("Checking if model file exists:", os.path.exists('medical_dataset_cache/saved_model.pth'))
    if load_model():
        print("Starting Flask server...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check the model path and try again.")