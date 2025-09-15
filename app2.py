from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, session
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from base_structure import BaseStructure
from utils.misc import get_model
import yaml
import io
import base64
from argparse import Namespace
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
import secrets
from werkzeug.datastructures import FileStorage
from io import BytesIO
import traceback
import mysql.connector
from mysql.connector import Error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key

# MySQL Database Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',  # MySQL root password
    'database': 'selfmask_db'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

class SelfMaskInference:
    def __init__(self, model_path, config_path):
        try:
            # Load configuration
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            if isinstance(self.config, dict):
                self.config = Namespace(**self.config)
            logger.info("Configuration loaded successfully")
            
            # Setup device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize model with exact config
            logger.info("Initializing model...")
            self.model = get_model(
                arch="maskformer",
                configs=self.config
            ).to(self.device)
            
            # Load trained weights
            logger.info(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            logger.info("Model weights loaded successfully")
            
            # Initialize base structure
            logger.info("Initializing base structure...")
            self.base_structure = BaseStructure(
                model=self.model,
                device=self.device
            )
            
            # Setup transforms matching training
            self.transform = T.Compose([
                T.Resize((224, 224)),  # Fixed size from your config
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            # Read the image file
            if isinstance(image, FileStorage):
                image = Image.open(image.stream).convert('RGB')
            else:
                image = Image.fromarray(image).convert('RGB')
            
            # Apply transforms matching training
            image = self.transform(image)
            
            # Add batch dimension
            image = image.unsqueeze(0)
            
            logger.info(f"Preprocessed image shape: {image.shape}")
            logger.info(f"Image value range: [{image.min()}, {image.max()}]")
            
            return image.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, image):
        """Run inference with proper post-processing"""
        try:
            # Preprocess
            logger.info("Starting image preprocessing...")
            img_tensor = self.preprocess_image(image)
            logger.info(f"Preprocessed image tensor shape: {img_tensor.shape}")
            
            # Create input dictionary
            dict_data = {'x': img_tensor}
            
            # Get predictions
            logger.info("Running model inference...")
            with torch.no_grad():
                outputs = self.base_structure._forward(dict_data)
            
            # Process predictions
            mask_pred = outputs['mask_pred']  # [1, n_decoder_layers, n_queries, H, W]
            objectness = outputs.get('objectness', None)  # [1, n_decoder_layers, n_queries, 1]
            
            # Debug shapes
            logger.info(f"mask_pred shape: {mask_pred.shape}")
            if objectness is not None:
                logger.info(f"objectness shape: {objectness.shape}")
            
            # Get best mask based on objectness scores
            if objectness is not None:
                # Get the last decoder layer's predictions
                last_layer_masks = mask_pred[0, -1]  # [n_queries, H, W]
                last_layer_objectness = objectness[0, -1].squeeze(-1)  # [n_queries]
                
                logger.info(f"Last layer masks shape: {last_layer_masks.shape}")
                logger.info(f"Last layer objectness shape: {last_layer_objectness.shape}")
                
                # Get best query index
                best_idx = torch.argmax(last_layer_objectness)
                logger.info(f"Best query index: {best_idx.item()}")
                
                # Get the best mask
                best_mask = last_layer_masks[best_idx]  # [H, W]
            else:
                # If no objectness scores, use the first query from last layer
                best_mask = mask_pred[0, -1, 0]  # [H, W]
            
            # Convert to numpy and ensure values are in [0, 1]
            best_mask = best_mask.cpu().numpy()
            best_mask = np.clip(best_mask, 0, 1)
            
            logger.info(f"Final mask shape: {best_mask.shape}")
            logger.info(f"Mask value range: [{best_mask.min()}, {best_mask.max()}]")
            
            return {
                'mask': best_mask,
                'objectness_scores': objectness[0, -1].squeeze(-1).cpu().numpy() if objectness is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Initialize inference pipeline
MODEL_PATH = "ckpt/nq20_ndl6_bc_sr10100_duts_pm_seed0_contrastive/latest_model.pt"
CONFIG_PATH = "configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"

try:
    inference = SelfMaskInference(MODEL_PATH, CONFIG_PATH)
    logger.info("Inference pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize inference pipeline: {str(e)}")
    raise

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    return jsonify({'success': True})
                
                return jsonify({'error': 'Invalid email or password'}), 401
            except Error as e:
                logger.error(f"Database error: {e}")
                return jsonify({'error': 'Database error'}), 500
            finally:
                cursor.close()
                conn.close()
        
        return jsonify({'error': 'Database connection error'}), 500
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # Check if email already exists
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    return jsonify({'error': 'Email already registered'}), 400
                
                # Insert new user
                hashed_password = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (email, password) VALUES (%s, %s)",
                    (email, hashed_password)
                )
                conn.commit()
                
                # Get the new user's ID
                user_id = cursor.lastrowid
                session['user_id'] = user_id
                return jsonify({'success': True})
            except Error as e:
                logger.error(f"Database error: {e}")
                return jsonify({'error': 'Database error'}), 500
            finally:
                cursor.close()
                conn.close()
        
        return jsonify({'error': 'Database connection error'}), 500
    
    return render_template('auth/register.html')

@app.route('/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/auth/status')
def auth_status():
    return jsonify({'authenticated': 'user_id' in session})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File must be an image'}), 400

        logger.info(f"Processing image: {file.filename}")
        
        # Save the file stream position
        file.seek(0)
        
        # Process the image
        result = inference.predict(file)
        logger.info("Model prediction completed")
        
        # Reset file stream position for reading
        file.seek(0)
        
        # Convert predictions to images
        original_img = Image.open(file).convert('RGB')
        original_img = original_img.resize((224, 224), Image.BILINEAR)
        
        # Convert mask to image and apply colormap
        mask = result['mask']
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        # Convert images to base64
        buffered_original = BytesIO()
        buffered_mask = BytesIO()
        
        original_img.save(buffered_original, format="PNG")
        mask_img.save(buffered_mask, format="PNG")
        
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode()
        
        logger.info("Successfully processed image and generated results")
        logger.info(f"Original image size: {original_img.size}")
        logger.info(f"Mask image size: {mask_img.size}")
        
        response = {
            'original': f'data:image/png;base64,{original_base64}',
            'mask': f'data:image/png;base64,{mask_base64}',
            'objectness_scores': result['objectness_scores'].tolist() if result['objectness_scores'] is not None else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
@login_required
def history():
    # Implement history view
    return render_template('history.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 