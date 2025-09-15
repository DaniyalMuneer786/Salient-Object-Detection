from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_mysqldb import MySQL
import os
import hashlib
import functools
import time
from flask_cors import CORS
from datetime import datetime, timedelta, date
import datetime as dt
import stripe
import random
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt

from flask_mail import Mail, Message
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl  # For secure connections
import secrets

# AI Model Imports from tayyab.py
import torch
import torchvision.transforms as T
import yaml
import io
import base64
from argparse import Namespace
import logging
import traceback
from io import BytesIO
from utils.misc import get_model
from base_structure import BaseStructure
from werkzeug.datastructures import FileStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_very_secret_key_here'
CORS(app)


app.config['MYSQL_DATETIME'] = True  # Ensure datetime objects are returned
# Initialize MySQL (add this right after app config)
mysql = MySQL(app)  # <-- THIS LINE IS CRUCIAL

# Then your database configuration
app.config['MYSQL_HOST'] = "127.0.0.1"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "SOD"
CORS(app)

bcrypt = Bcrypt(app)

# Email Configuration (update with your real credentials or environment variables)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '211980031@gift.edu.pk'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'abodqcrfspoksthv'    # Use App Password for Gmail
app.config['MAIL_DEFAULT_SENDER'] = '211980031@gift.edu.pk'

mail = Mail(app)
# === Stripe Test Secret Key ===
stripe.api_key = 'sk_test_51RVq55PCPwXipCduM0js8eKiZLYupEIDG4Nvypss2WnFSSKw5Tu4pbzLpTQwxy0PMgI7uZRpnIFZ2AcNsUNCGrcZ00GFuV2dKc'  # Replace with your Stripe TEST Secret Key

# AI Model Class from tayyab.py
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
                T.Resize((224, 224)),
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
            mask_pred = outputs['mask_pred']
            objectness = outputs.get('objectness', None)
            
            # Debug shapes
            logger.info(f"mask_pred shape: {mask_pred.shape}")
            if objectness is not None:
                logger.info(f"objectness shape: {objectness.shape}")
            
            # Get best mask based on objectness scores
            if objectness is not None:
                last_layer_masks = mask_pred[0, -1]
                last_layer_objectness = objectness[0, -1].squeeze(-1)
                
                logger.info(f"Last layer masks shape: {last_layer_masks.shape}")
                logger.info(f"Last layer objectness shape: {last_layer_objectness.shape}")
                
                best_idx = torch.argmax(last_layer_objectness)
                logger.info(f"Best query index: {best_idx.item()}")
                
                best_mask = last_layer_masks[best_idx]
            else:
                best_mask = mask_pred[0, -1, 0]
            
            # Convert to numpy and ensure values are in [0, 1]
            best_mask = best_mask.cpu().numpy()
            best_mask = np.clip(best_mask, 0, 1)
            
            logger.info(f"Final mask shape: {best_mask.shape}")
            logger.info(f"Mask value range: [{best_mask.min()}, {best_mask.max()}]")

            # Convert original image to PIL Image
            if isinstance(image, FileStorage):
                original_img = Image.open(image.stream).convert('RGB')
                # Reset stream position
                image.stream.seek(0)
            else:
                original_img = image.convert('RGB')

            # Convert mask to image
            mask_array = (best_mask * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_array)
            mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)

            # Convert images to base64
            buffered_original = BytesIO()
            buffered_mask = BytesIO()
            
            try:
                original_img.save(buffered_original, format="PNG", quality=95)
                mask_img.save(buffered_mask, format="PNG", quality=95)
                
                original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
                mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode()
                
                logger.info("Images successfully converted to base64")
                
                response = {
                    'original': f'data:image/png;base64,{original_base64}',
                    'mask': f'data:image/png;base64,{mask_base64}',
                    'objectness_scores': objectness[0, -1].squeeze(-1).cpu().numpy() if objectness is not None else None
                }
                
                logger.info("Prediction completed successfully")
                return response
                
            except Exception as e:
                logger.error(f"Error converting images to base64: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Initialize inference pipeline
MODEL_PATH = "ckpt/nq20_ndl6_bc_sr10100_duts_pm_seed0_contrastive/latest_model.pt"
CONFIG_PATH = "configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"

try:
    logger.info("Initializing inference pipeline...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Config path: {CONFIG_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        
    inference = SelfMaskInference(MODEL_PATH, CONFIG_PATH)
    logger.info("Inference pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize inference pipeline: {str(e)}")
    logger.error(f"Error details: {type(e).__name__}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Helper decorator
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session and 'admin_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function




# Store pending signups with confirmation tokens
pending_signups = {}

def generate_confirmation_token(email):
    """Generate a secure confirmation token"""
    timestamp = str(int(time.time()))
    data = f"{email}:{timestamp}:{secrets.token_urlsafe(32)}"
    return hashlib.sha256(data.encode()).hexdigest()


# ======================= Index Route ============================
@app.route('/')
def index():
    session.clear()
    return render_template('First Page.html')

# ======================== Dashboard =============================
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')


# # Store signup OTPs and temporary user data
# signup_otps = {}

# @app.route('/user_signup', methods=['POST'])
# def signup():
#     try:
#         data = request.get_json()
#         user_id = random.randint(1, 1000)
        
#         user_data = {
#             'name': data.get('name'),
#             'email': data.get('email'),
#             'password': generate_password_hash(data.get('password')),
#             'phone': data.get('phone')
#         }

#         # Validate required fields
#         if not all([user_data['name'], user_data['email'], data.get('password')]):
#             return jsonify({
#                 'success': False,
#                 'message': 'Name, email and password are required',
#                 'field': 'all'
#             }), 400

#         cur = mysql.connection.cursor()
        
#         # Check if email exists
#         cur.execute('SELECT User_ID FROM user_management WHERE Email = %s', (user_data['email'],))
#         if cur.fetchone():
#             cur.close()
#             return jsonify({
#                 'success': False,
#                 'message': 'Email already registered!',
#                 'field': 'email'
#             }), 400

#         # Ensure ID is unique
#         while True:
#             cur.execute('SELECT User_ID FROM user_management WHERE User_ID = %s', (user_id,))
#             if not cur.fetchone():
#                 break
#             user_id = random.randint(1, 1000)

#         # Generate and send OTP
#         otp = str(secrets.randbelow(999999)).zfill(6)
#         expiration = datetime.now() + timedelta(minutes=10)
        
#         # Store user data and OTP in memory
#         signup_otps[user_data['email']] = {
#             'otp': otp,
#             'expires': expiration,
#             'user_data': user_data,
#             'user_id': user_id
#         }
        
#         # Send welcome email with OTP
#         send_signup_otp_email(user_data['email'], user_data['name'], otp)
        
#         return jsonify({
#             'success': True,
#             'message': 'Registration successful! Please check your email for the verification OTP.',
#             'needs_verification': True,
#             'email': user_data['email']
#         })

#     except Exception as e:
#         if 'cur' in locals():
#             cur.close()
#         return jsonify({
#             'success': False,
#             'message': f'Registration failed: {str(e)}'
#         }), 500


@app.route('/api/endpoint')
def api_endpoint():
    try:
        # Your logic here
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



# Store temporary user data for verification
signup_data = {}

@app.route('/user_signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        
        user_id = random.randint(1, 1000)
        
        user_data = {
            'name': data.get('name'),
            'email': data.get('email'),
            'password': generate_password_hash(data.get('password')),
            'phone': data.get('phone')
        }

        # Validate required fields
        if not all([user_data['name'], user_data['email'], data.get('password')]):
            return jsonify({
                'success': False,
                'message': 'Name, email and password are required',
                'field': 'all'
            }), 400

        cur = mysql.connection.cursor()
        
        # Check if email exists
        cur.execute('SELECT User_ID FROM user_management WHERE Email = %s', (user_data['email'],))
        if cur.fetchone():
            cur.close()
            return jsonify({
                'success': False,
                'message': 'Email already registered!',
                'field': 'email'
            }), 400

        # Ensure ID is unique
        while True:
            cur.execute('SELECT User_ID FROM user_management WHERE User_ID = %s', (user_id,))
            if not cur.fetchone():
                break
            user_id = random.randint(1, 1000)

        # Generate verification token
        verification_token = secrets.token_urlsafe(32)
        expiration = datetime.now() + timedelta(minutes=10)
        
        # Store user data in memory
        signup_data[verification_token] = {
            'expires': expiration,
            'user_data': user_data,
            'user_id': user_id
        }
        
        # Create verification link
        verification_link = url_for('verify_signup', token=verification_token, _external=True)
        
        # Send welcome email with verification link
        send_signup_email(user_data['email'], user_data['name'], verification_link)
        
        return jsonify({
            'success': True,
            'message': 'Registration initiated! Please check your email for the verification link.',
            'needs_verification': True,
            'email': user_data['email']
        })

    except Exception as e:
        if 'cur' in locals():
            cur.close()
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
        }), 500



@app.route('/verify_signup/<token>')
def verify_signup(token):
    try:
        # Get the verification data
        verification_data = signup_data.get(token)
        
        if not verification_data:
            flash('Invalid or expired verification link', 'error')
            return redirect(url_for('index'))
        
        if datetime.now() > verification_data['expires']:
            del signup_data[token]  # Clean up expired token
            flash('Verification link has expired', 'error')
            return redirect(url_for('index'))
        
        # Create the user account
        cur = mysql.connection.cursor()
        try:
            cur.execute('''INSERT INTO user_management (User_ID, Name, Email, Password, Phone)
                           VALUES (%s, %s, %s, %s, %s)''',
                        (verification_data['user_id'], 
                         verification_data['user_data']['name'],
                         verification_data['user_data']['email'], 
                         verification_data['user_data']['password'],
                         verification_data['user_data']['phone']))
            mysql.connection.commit()
            
            # Clean up verification data
            del signup_data[token]
            
            # Set session and redirect
            session['user_id'] = verification_data['user_id']
            flash('Email verified successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            mysql.connection.rollback()
            app.logger.error(f"Database error during verification: {str(e)}")
            flash('Account creation failed. Please try signing up again.', 'error')
            return redirect(url_for('index'))
        finally:
            cur.close()

    except Exception as e:
        app.logger.error(f"Error in verify_signup: {str(e)}")
        flash('Verification failed. Please try again.', 'error')
        return redirect(url_for('index'))        


def send_verification_link(email, name, token):
    try:
        verification_url = url_for('verify_email', token=token, _external=True)
        
        msg = Message(
            subject="🎉 Verify Your Email for Salient Object Detection",
            recipients=[email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        msg.html = f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h2>Welcome {name}!</h2>
            <p>Thank you for registering with Salient Object Detection.</p>
            <p>Please click the button below to verify your email address:</p>
            <a href="{verification_url}" style="
                display: inline-block;
                padding: 12px 24px;
                background-color: #00e6ac;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                margin: 20px 0;">
                Verify Email
            </a>
            <p>Or copy this link to your browser: {verification_url}</p>
            <p>This link will expire in 24 hours.</p>
        </body>
        </html>
        """
        
        mail.send(msg)
        return True
    except Exception as e:
        app.logger.error(f"Failed to send verification email: {str(e)}")
        return False



def send_signup_email(email, name, confirmation_url):
    try:
        msg = Message(
            subject="🎉 Welcome to Salient Object Detection - Confirm Your Email",
            recipients=[email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        msg.html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to Salient Object Detection</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .email-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: #ffffff;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                    position: relative;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #00e6ac 0%, #00b386 100%);
                    padding: 40px 30px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .logo {{
                    font-size: 28px;
                    font-weight: 700;
                    color: white;
                    margin-bottom: 10px;
                    position: relative;
                    z-index: 2;
                }}
                
                .welcome-title {{
                    font-size: 32px;
                    font-weight: 600;
                    color: white;
                    margin-bottom: 15px;
                    position: relative;
                    z-index: 2;
                }}
                
                .welcome-subtitle {{
                    font-size: 16px;
                    color: rgba(255,255,255,0.9);
                    position: relative;
                    z-index: 2;
                }}
                
                .content {{
                    padding: 50px 40px;
                    text-align: center;
                }}
                
                .greeting {{
                    font-size: 24px;
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 20px;
                }}
                
                .message {{
                    font-size: 16px;
                    color: #4a5568;
                    margin-bottom: 30px;
                    line-height: 1.7;
                }}
                
                .confirmation-container {{
                    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                    border: 2px solid #e2e8f0;
                    border-radius: 16px;
                    padding: 30px;
                    margin: 30px 0;
                    position: relative;
                    overflow: hidden;
                }}
                
                .confirmation-title {{
                    font-size: 18px;
                    font-weight: 600;
                    color: #00e6ac;
                    margin-bottom: 15px;
                }}
                
                .confirmation-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #00e6ac 0%, #00b386 100%);
                    color: white;
                    padding: 15px 40px;
                    border-radius: 50px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 8px 25px rgba(0,230,172,0.3);
                    transition: all 0.3s ease;
                    margin: 15px 0;
                }}
                
                .confirmation-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 12px 35px rgba(0,230,172,0.4);
                }}
                
                .expiry-note {{
                    font-size: 14px;
                    color: #e53e3e;
                    font-weight: 500;
                    margin-top: 15px;
                }}
                
                .team-section {{
                    background: #f8f9fa;
                    padding: 30px;
                    margin-top: 30px;
                }}
                
                .team-title {{
                    font-size: 18px;
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 20px;
                }}
                
                .team-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                
                .team-member {{
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    font-weight: 500;
                    color: #4a5568;
                    position: relative;
                    overflow: hidden;
                }}
                
                .team-member::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 4px;
                    height: 100%;
                    background: linear-gradient(135deg, #00e6ac, #667eea);
                }}
                
                .footer {{
                    background: #2d3748;
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                
                .footer-message {{
                    font-size: 16px;
                    margin-bottom: 15px;
                    font-style: italic;
                }}
                
                .footer-divider {{
                    width: 60px;
                    height: 2px;
                    background: linear-gradient(90deg, #00e6ac, #667eea);
                    margin: 20px auto;
                    border-radius: 2px;
                }}
                
                .security-note {{
                    background: #fff5f5;
                    border-left: 4px solid #fc8181;
                    padding: 15px 20px;
                    margin: 20px 0;
                    border-radius: 0 8px 8px 0;
                }}
                
                .security-note-text {{
                    font-size: 14px;
                    color: #742a2a;
                    margin: 0;
                }}
                
                @media (max-width: 600px) {{
                    .email-container {{
                        margin: 10px;
                        border-radius: 15px;
                    }}
                    
                    .content {{
                        padding: 30px 20px;
                    }}
                    
                    .team-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .confirmation-button {{
                        padding: 12px 30px;
                        font-size: 14px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <div class="logo">🎯 Salient Object Detection</div>
                    <div class="welcome-title">Welcome, {name}! 🎉</div>
                    <div class="welcome-subtitle">Thank you for joining our AI-powered platform</div>
                </div>
                
                <div class="content">
                    <div class="greeting">Almost there! Let's confirm your email</div>
                    <div class="message">
                        Thank you for registering with <strong>Salient Object Detection</strong>. 
                        We're excited to have you on board! To complete your registration and activate your account, 
                        please click the confirmation button below.
                    </div>
                    
                    <div class="confirmation-container">
                        <div class="confirmation-title">📧 Email Confirmation</div>
                        <div>Click the button below to confirm your email and activate your account:</div>
                        <a href="{confirmation_url}" class="confirmation-button">
                            ✅ Confirm My Email
                        </a>
                        <div class="expiry-note">⏰ This link expires in 24 hours</div>
                    </div>
                    
                    <div class="security-note">
                        <p class="security-note-text">
                            🔒 <strong>Security Note:</strong> If you didn't create an account with us, please ignore this email. 
                            This confirmation link is unique to your email address.
                        </p>
                    </div>
                </div>
                
                <div class="team-section">
                    <div class="team-title">👥 Meet Our Team</div>
                    <div class="team-grid">
                        <div class="team-member">👨‍💻 Daniyal Muneer</div>
                        <div class="team-member">👨‍💻 Tayyab Riaz</div>
                        <div class="team-member">👨‍💻 Noman Ali</div>
                        <div class="team-member">👩‍💻 Zonia Tariq</div>
                    </div>
                </div>
                
                <div class="footer">
                    <div class="footer-message">Thanks for choosing our SOD platform! 🚀</div>
                    <div class="footer-divider"></div>
                    <div style="font-size: 14px; opacity: 0.8;">
                        © 2024 Salient Object Detection Team. All rights reserved.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        msg.body = f"""
        Welcome to Salient Object Detection, {name}!
        
        Thank you for registering with our platform.
        
        To complete your registration, please click the following link:
        {confirmation_url}
        
        This link will expire in 24 hours.
        
        If you didn't create an account with us, please ignore this email.
        
        Best regards,
        The SOD Team
        
        Team Members:
        - Daniyal Muneer
        - Tayyab Riaz  
        - Noman Ali
        - Zonia Tariq
        """
        
        mail.send(msg)
        app.logger.info(f"Confirmation email sent to {email}")
        return True
    except Exception as e:
        app.logger.error(f"Failed to send confirmation email: {str(e)}")
        return False
        
#         If you have any questions, please don't hesitate to contact our support team.
        
#         Best regards,
#         The SOD Team
#         """
        
#         mail.send(msg)
#         app.logger.info(f"Welcome email sent to {user_email}")
#         return True
#     except Exception as e:
#         app.logger.error(f"Failed to send email to {user_email}: {str(e)}")
#         return False

# @app.route('/test_email')
# def test_email():
#     try:
#         # Send test email to yourself
#         success = send_welcome_email(app.config['MAIL_USERNAME'], "Test User")
#         if success:
#             return "Test email sent successfully!"
#         else:
#             return "Failed to send test email (check logs)"
#     except Exception as e:
#         return f"Error: {str(e)}"

# Store OTPs temporarily (in production, use Redis or database)
otp_storage = {}

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')
        
        # Check if email exists in database
        cur = mysql.connection.cursor()
        cur.execute('SELECT User_ID FROM user_management WHERE Email = %s', (email,))
        user = cur.fetchone()
        cur.close()
        
        if not user:
            return jsonify({'success': False, 'message': 'Email not found'}), 404
        
        # Generate 6-digit OTP
        otp = str(secrets.randbelow(999999)).zfill(6)
        expiration = datetime.now() + timedelta(minutes=10)
        
        # Store OTP with expiration
        otp_storage[email] = {
            'otp': otp,
            'expires': expiration,
            'user_id': user[0]
        }
        
        # Send OTP via email
        send_otp_email(email, otp)
        
        return jsonify({
            'success': True,
            'message': 'OTP sent to your email',
            'email': email  # Return email for verification step
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json()
        email = data.get('email')
        user_otp = data.get('otp')
        
        # Check if OTP exists and is not expired
        otp_data = otp_storage.get(email)
        
        if not otp_data or datetime.now() > otp_data['expires']:
            return jsonify({'success': False, 'message': 'OTP expired or invalid'}), 400
        
        if user_otp != otp_data['otp']:
            return jsonify({'success': False, 'message': 'Invalid OTP'}), 400
        
        # OTP is valid, return success and user_id for password reset
        return jsonify({
            'success': True,
            'message': 'OTP verified',
            'user_id': otp_data['user_id']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/reset_password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        if new_password != confirm_password:
            return jsonify({
                'success': False,
                'message': 'Passwords do not match'
            }), 400
        
        # Hash the new password
        hashed_password = generate_password_hash(new_password)
        
        # First get the user's email before updating
        cur = mysql.connection.cursor()
        cur.execute("SELECT Email FROM user_management WHERE User_ID = %s", (user_id,))
        user_email = cur.fetchone()[0]
        
        # Update password in database
        cur.execute(
            'UPDATE user_management SET Password = %s WHERE User_ID = %s',
            (hashed_password, user_id)
        )
        mysql.connection.commit()
        cur.close()
        
        return jsonify({
            'success': True,
            'message': 'Password updated successfully',
            'email': user_email  # Include the email in the response
        })
        
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({
            'success': False,
            'message': f'Error resetting password: {str(e)}'
        }), 500

def send_otp_email(email, otp):
    try:
        msg = Message(
            subject="🔐 Password Reset Request - Salient Object Detection",
            recipients=[email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        msg.html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password Reset - Salient Object Detection</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #fc7a57 0%, #f093fb 50%, #667eea 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .email-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: #ffffff;
                    border-radius: 20px;
                    box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                    overflow: hidden;
                    position: relative;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    padding: 40px 30px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="security" width="50" height="50" patternUnits="userSpaceOnUse"><path d="M25 10 L35 20 L25 30 L15 20 Z" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23security)"/></svg>');
                    animation: pulse 4s ease-in-out infinite;
                }}
                
                @keyframes pulse {{
                    0%, 100% {{ opacity: 0.3; transform: scale(1); }}
                    50% {{ opacity: 0.1; transform: scale(1.05); }}
                }}
                
                .security-icon {{
                    font-size: 48px;
                    margin-bottom: 15px;
                    position: relative;
                    z-index: 2;
                }}
                
                .header-title {{
                    font-size: 28px;
                    font-weight: 700;
                    color: white;
                    margin-bottom: 10px;
                    position: relative;
                    z-index: 2;
                }}
                
                .header-subtitle {{
                    font-size: 16px;
                    color: rgba(255,255,255,0.9);
                    position: relative;
                    z-index: 2;
                }}
                
                .content {{
                    padding: 50px 40px;
                    text-align: center;
                }}
                
                .alert-section {{
                    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                    border: 2px solid #ffc107;
                    border-radius: 16px;
                    padding: 25px;
                    margin-bottom: 30px;
                    position: relative;
                    overflow: hidden;
                }}
                
                .alert-section::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, #ff6b6b, #ffc107, #ee5a24, #ff6b6b);
                    background-size: 200% 100%;
                    animation: alertShimmer 2s ease-in-out infinite;
                }}
                
                @keyframes alertShimmer {{
                    0% {{ background-position: -200% 0; }}
                    100% {{ background-position: 200% 0; }}
                }}
                
                .alert-icon {{
                    font-size: 32px;
                    margin-bottom: 15px;
                }}
                
                .alert-title {{
                    font-size: 20px;
                    font-weight: 600;
                    color: #856404;
                    margin-bottom: 10px;
                }}
                
                .alert-message {{
                    font-size: 16px;
                    color: #856404;
                    line-height: 1.6;
                }}
                
                .main-message {{
                    font-size: 18px;
                    color: #4a5568;
                    margin-bottom: 30px;
                    line-height: 1.7;
                }}
                
                .otp-container {{
                    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    border: 3px solid #0ea5e9;
                    border-radius: 20px;
                    padding: 35px;
                    margin: 30px 0;
                    position: relative;
                    overflow: hidden;
                }}
                
                .otp-container::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 5px;
                    background: linear-gradient(90deg, #0ea5e9, #3b82f6, #6366f1, #0ea5e9);
                    background-size: 200% 100%;
                    animation: otpShimmer 3s ease-in-out infinite;
                }}
                
                @keyframes otpShimmer {{
                    0% {{ background-position: -200% 0; }}
                    100% {{ background-position: 200% 0; }}
                }}
                
                .otp-icon {{
                    font-size: 40px;
                    margin-bottom: 15px;
                }}
                
                .otp-title {{
                    font-size: 22px;
                    font-weight: 700;
                    color: #0369a1;
                    margin-bottom: 15px;
                }}
                
                .otp-label {{
                    font-size: 16px;
                    color: #0369a1;
                    margin-bottom: 15px;
                    font-weight: 500;
                }}
                
                .otp-code {{
                    font-size: 42px;
                    font-weight: 800;
                    color: #dc2626;
                    letter-spacing: 10px;
                    margin: 20px 0;
                    padding: 20px 30px;
                    background: white;
                    border-radius: 15px;
                    border: 3px dashed #0ea5e9;
                    display: inline-block;
                    box-shadow: 0 8px 25px rgba(14,165,233,0.3);
                    position: relative;
                }}
                
                .otp-code::before {{
                    content: '';
                    position: absolute;
                    top: -3px;
                    left: -3px;
                    right: -3px;
                    bottom: -3px;
                    background: linear-gradient(45deg, #0ea5e9, #3b82f6, #6366f1, #0ea5e9);
                    border-radius: 15px;
                    z-index: -1;
                    animation: borderGlow 2s ease-in-out infinite;
                }}
                
                @keyframes borderGlow {{
                    0%, 100% {{ opacity: 0.5; }}
                    50% {{ opacity: 1; }}
                }}
                
                .otp-validity {{
                    font-size: 16px;
                    color: #dc2626;
                    font-weight: 600;
                    margin-top: 15px;
                    background: #fef2f2;
                    padding: 10px 20px;
                    border-radius: 25px;
                    display: inline-block;
                }}
                
                .security-tips {{
                    background: #f0fdf4;
                    border: 2px solid #22c55e;
                    border-radius: 16px;
                    padding: 25px;
                    margin: 30px 0;
                    text-align: left;
                }}
                
                .security-tips-title {{
                    font-size: 18px;
                    font-weight: 600;
                    color: #166534;
                    margin-bottom: 15px;
                    text-align: center;
                }}
                
                .security-tip {{
                    font-size: 14px;
                    color: #166534;
                    margin-bottom: 8px;
                    padding-left: 20px;
                    position: relative;
                }}
                
                .security-tip::before {{
                    content: '🔒';
                    position: absolute;
                    left: 0;
                    top: 0;
                }}
                
                .footer {{
                    background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                
                .footer-message {{
                    font-size: 16px;
                    margin-bottom: 15px;
                }}
                
                .footer-divider {{
                    width: 80px;
                    height: 3px;
                    background: linear-gradient(90deg, #ff6b6b, #0ea5e9);
                    margin: 20px auto;
                    border-radius: 3px;
                }}
                
                @media (max-width: 600px) {{
                    .email-container {{
                        margin: 10px;
                        border-radius: 15px;
                    }}
                    
                    .content {{
                        padding: 30px 20px;
                    }}
                    
                    .otp-code {{
                        font-size: 32px;
                        letter-spacing: 6px;
                        padding: 15px 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <div class="security-icon">🛡️</div>
                    <div class="header-title">Password Reset Request</div>
                    <div class="header-subtitle">Secure your account with us</div>
                </div>
                
                <div class="content">
                    <div class="alert-section">
                        <div class="alert-icon">⚠️</div>
                        <div class="alert-title">Security Alert</div>
                        <div class="alert-message">
                            A password reset was requested for your account. If this wasn't you, 
                            please ignore this email and your password will remain unchanged.
                        </div>
                    </div>
                    
                    <div class="main-message">
                        To reset your password, please use the verification code below. 
                        This code is valid for a limited time only.
                    </div>
                    
                    <div class="otp-container">
                        <div class="otp-icon">🔑</div>
                        <div class="otp-title">Password Reset Code</div>
                        <div class="otp-label">Enter this code to proceed:</div>
                        <div class="otp-code">{otp}</div>
                        <div class="otp-validity">⏰ Expires in 10 minutes</div>
                    </div>
                    
                    <div class="security-tips">
                        <div class="security-tips-title">🔐 Security Tips</div>
                        <div class="security-tip">Never share your reset code with anyone</div>
                        <div class="security-tip">Use a strong, unique password</div>
                        <div class="security-tip">Enable two-factor authentication if available</div>
                        <div class="security-tip">Log out of all devices after changing password</div>
                    </div>
                </div>
                
                <div class="footer">
                    <div class="footer-message">🔒 Your security is our priority</div>
                    <div class="footer-divider"></div>
                    <div style="font-size: 14px; opacity: 0.8;">
                        © 2024 Salient Object Detection Team. All rights reserved.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        msg.body = f"""
        Password Reset Request - Salient Object Detection
        
        A password reset was requested for your account.
        
        Your password reset code is: {otp}
        
        This code is valid for 10 minutes.
        
        If you didn't request this reset, please ignore this email.
        
        Security Tips:
        - Never share your reset code with anyone
        - Use a strong, unique password
        - Enable two-factor authentication if available
        
        Best regards,
        The SOD Team
        """
        
        mail.send(msg)
        app.logger.info(f"Beautiful password reset OTP sent to {email}")
        
    except Exception as e:
        app.logger.error(f"Failed to send password reset OTP email: {str(e)}")
        raise

# ========================= Admin Entry ==========================
@app.route('/admin_entry')
def admin_entry():
    role = request.args.get('role')
    if role == 'admin':
        session['login_role'] = 'admin'
    return render_template('Admin.html')

# ========================= User Login ============================
@app.route('/user_login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        input_password = data.get('password')

        cur = mysql.connection.cursor()
        cur.execute('SELECT User_ID, Name, Password FROM user_management WHERE Email = %s', (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], input_password):  # ✅ Correctly compare hashed password
            session['user_id'] = user[0]
            return jsonify({
                'success': True,
                'message': 'User login successful!',
                'redirect': '/dashboard'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid email or password',
                'field': 'password'
            }), 401
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Login error: {str(e)}'
        }), 500

# ========================= Admin Login ===========================
@app.route('/admin_login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Fixed admin credentials (store these securely in production)
        FIXED_ADMINS = {
            'admin@gmail.com': {
                'id': 1000,
                'name': 'Daniyal',
                'password': '111',  # In production, use hashed password
                'salary': 100000
            }
        }

        # Check if email exists in fixed admins
        if email not in FIXED_ADMINS:
            return jsonify({
                'success': False,
                'message': 'Invalid admin credentials',
                'field': 'email'
            }), 401

        admin = FIXED_ADMINS[email]

        # Verify password (in production, use check_password_hash)
        if password != admin['password']:
            return jsonify({
                'success': False,
                'message': 'Invalid admin credentials',
                'field': 'password'
            }), 401

        # Check if admin exists in database
        cur = mysql.connection.cursor()
        cur.execute('SELECT Admin_ID FROM admin WHERE Email = %s', (email,))
        if not cur.fetchone():
            # Insert the fixed admin if not exists
            cur.execute(
                'INSERT INTO admin (Admin_ID, Name, Email, Password, Salary) '
                'VALUES (%s, %s, %s, %s, %s)',
                (admin['id'], admin['name'], email, admin['password'], admin['salary'])
            )
            mysql.connection.commit()

        session['admin_id'] = admin['id']
        cur.close()

        return jsonify({
            'success': True,
            'message': 'Admin login successful!',
            'redirect': '/view'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Login error: {str(e)}'
        }), 500    

# ============================ Logout ==============================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ======================== Data API Routes ==========================
@app.route('/api/data/<table_name>')
@login_required
def get_table_data(table_name):
    try:
        # Validate table name to prevent SQL injection
        valid_tables = {
            'users': 'user_management',
            'images': 'image',
            'results': 'result',
            'subscriptions': 'subscription',
            'feedback': 'feedback',
            'admin': 'admin'
        }
        
        if table_name not in valid_tables:
            return jsonify({
                'error': 'Invalid table name',
                'valid_tables': list(valid_tables.keys())
            }), 400

        actual_table_name = valid_tables[table_name]
        cur = mysql.connection.cursor()
        
        # Get column names first
        cur.execute(f"SHOW COLUMNS FROM {actual_table_name}")
        columns = [column[0] for column in cur.fetchall()]
        
        # Get table data with enhanced formatting
        cur.execute(f"SELECT * FROM {actual_table_name} ORDER BY 1 DESC LIMIT 100")
        data = cur.fetchall()
        
        # Convert to list of dictionaries with enhanced formatting
        result = []
        for row in data:
            row_dict = dict(zip(columns, row))
            
            # Format specific fields based on table type
            if table_name == 'users':
                # Format user data
                if 'created_at' in row_dict and row_dict['created_at']:
                    row_dict['created_at'] = row_dict['created_at'].strftime('%Y-%m-%d %H:%M')
                if 'Password' in row_dict:
                    # Mask password for security
                    password = row_dict['Password']
                    if password and len(password) > 4:
                        row_dict['Password'] = password[:4] + '*' * (len(password) - 4)
                    else:
                        row_dict['Password'] = '****'
            
            elif table_name == 'images':
                # Format image data
                if 'uploaded_at' in row_dict and row_dict['uploaded_at']:
                    row_dict['uploaded_at'] = row_dict['uploaded_at'].strftime('%Y-%m-%d %H:%M')
                if 'file_size' in row_dict and row_dict['file_size']:
                    row_dict['file_size'] = f"{row_dict['file_size']:.1f}"
            
            elif table_name == 'results':
                # Format result data
                if 'processed_at' in row_dict and row_dict['processed_at']:
                    row_dict['processed_at'] = row_dict['processed_at'].strftime('%Y-%m-%d %H:%M')
                # Truncate long result data
                if 'Result_data' in row_dict and row_dict['Result_data']:
                    result_text = row_dict['Result_data']
                    if len(result_text) > 100:
                        row_dict['Result_data'] = result_text[:100] + '...'
            
            elif table_name == 'subscriptions':
                # Format subscription data
                if 'Start_Date' in row_dict and row_dict['Start_Date']:
                    row_dict['Start_Date'] = row_dict['Start_Date'].strftime('%Y-%m-%d')
                if 'End_Date' in row_dict and row_dict['End_Date']:
                    row_dict['End_Date'] = row_dict['End_Date'].strftime('%Y-%m-%d')
                if 'Amount_Paid' in row_dict and row_dict['Amount_Paid']:
                    row_dict['Amount_Paid'] = f"{float(row_dict['Amount_Paid']):.2f}"
                # Capitalize status
                if 'Status' in row_dict and row_dict['Status']:
                    row_dict['Status'] = row_dict['Status'].title()
            
            elif table_name == 'feedback':
                # Format feedback data
                if 'Feedback_Text' in row_dict and row_dict['Feedback_Text']:
                    feedback_text = row_dict['Feedback_Text']
                    if len(feedback_text) > 80:
                        row_dict['Feedback_Text'] = feedback_text[:80] + '...'
                # Capitalize feedback type
                if 'Feedback_Type' in row_dict and row_dict['Feedback_Type']:
                    row_dict['Feedback_Type'] = row_dict['Feedback_Type'].title()
            
            elif table_name == 'admin':
                # Format admin data
                if 'Salary' in row_dict and row_dict['Salary']:
                    row_dict['Salary'] = f"{float(row_dict['Salary']):,.2f}"
            
            result.append(row_dict)
        
        cur.close()
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result),
            'table_info': {
                'name': table_name,
                'display_name': table_name.replace('_', ' ').title(),
                'total_records': len(result)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'Error fetching {table_name} data'
        }), 500

# ======================== Enhanced Statistics Routes =========================
@app.route('/api/stats/overview')
@login_required
def get_overview_stats():
    try:
        cur = mysql.connection.cursor()
        
        # Get comprehensive stats
        stats = {}
        
        # User stats
        cur.execute("SELECT COUNT(*) FROM user_management")
        stats['total_users'] = cur.fetchone()[0]
        
        # Image stats
        cur.execute("SELECT COUNT(*) FROM image")
        stats['total_images'] = cur.fetchone()[0]
        
        cur.execute("SELECT AVG(file_size) FROM image")
        avg_size = cur.fetchone()[0]
        stats['avg_file_size'] = round(float(avg_size or 0), 2)
        
        # Result stats
        cur.execute("SELECT COUNT(*) FROM result")
        stats['total_results'] = cur.fetchone()[0]
        
        # Subscription stats
        cur.execute("SELECT COUNT(*) FROM subscription WHERE Status = 'active'")
        stats['active_subscriptions'] = cur.fetchone()[0]
        
        cur.execute("SELECT SUM(Amount_Paid) FROM subscription WHERE Status = 'active'")
        total_revenue = cur.fetchone()[0]
        stats['total_revenue'] = float(total_revenue or 0)
        
        # Feedback stats
        cur.execute("SELECT COUNT(*) FROM feedback")
        stats['total_feedback'] = cur.fetchone()[0]
        
        cur.execute("SELECT AVG(Rating) FROM feedback")
        avg_rating = cur.fetchone()[0]
        stats['avg_rating'] = round(float(avg_rating or 0), 1)
        
        # Recent activity (last 7 days)
        cur.execute("""
            SELECT COUNT(*) FROM image 
            WHERE uploaded_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        stats['recent_uploads'] = cur.fetchone()[0]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ======================== Enhanced User Analytics =========================
@app.route('/api/analytics/users')
@login_required
def get_user_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # User registration trend (last 30 days)
        cur.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM user_management 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        registration_trend = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1]} 
                            for row in cur.fetchall()]
        
        # User activity levels
        cur.execute("""
            SELECT 
                CASE 
                    WHEN upload_count = 0 THEN 'Inactive'
                    WHEN upload_count BETWEEN 1 AND 5 THEN 'Low Activity'
                    WHEN upload_count BETWEEN 6 AND 20 THEN 'Medium Activity'
                    ELSE 'High Activity'
                END as activity_level,
                COUNT(*) as count
            FROM (
                SELECT u.User_ID, COUNT(i.image_id) as upload_count
                FROM user_management u
                LEFT JOIN image i ON u.User_ID = i.user_id
                GROUP BY u.User_ID
            ) user_activity
            GROUP BY activity_level
        """)
        activity_levels = [{'level': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Name length distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN LENGTH(Name) BETWEEN 1 AND 5 THEN '1-5 chars'
                    WHEN LENGTH(Name) BETWEEN 6 AND 10 THEN '6-10 chars'
                    ELSE '10+ chars'
                END as name_length,
                COUNT(*) as count
            FROM user_management
            GROUP BY name_length
        """)
        name_lengths = [{'length': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'registration_trend': registration_trend,
            'activity_levels': activity_levels,
            'name_lengths': name_lengths
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ======================== Enhanced Image Analytics =========================
@app.route('/api/analytics/images')
@login_required
def get_image_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # Upload trend (last 30 days)
        cur.execute("""
            SELECT DATE(uploaded_at) as date, COUNT(*) as count
            FROM image 
            WHERE uploaded_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(uploaded_at)
            ORDER BY date
        """)
        upload_trend = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1]} 
                       for row in cur.fetchall()]
        
        # File type distribution
        cur.execute("""
            SELECT file_type, COUNT(*) as count
            FROM image
            GROUP BY file_type
            ORDER BY count DESC
        """)
        file_types = [{'type': row[0].upper(), 'count': row[1]} for row in cur.fetchall()]
        
        # File size distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN file_size < 100 THEN '< 100 KB'
                    WHEN file_size BETWEEN 100 AND 500 THEN '100-500 KB'
                    WHEN file_size BETWEEN 500 AND 1000 THEN '500KB-1MB'
                    ELSE '> 1 MB'
                END as size_range,
                COUNT(*) as count
            FROM image
            GROUP BY size_range
        """)
        size_distribution = [{'range': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Hourly upload pattern
        cur.execute("""
            SELECT HOUR(uploaded_at) as hour, COUNT(*) as count
            FROM image
            GROUP BY HOUR(uploaded_at)
            ORDER BY hour
        """)
        hourly_pattern = [{'hour': f"{row[0]:02d}:00", 'count': row[1]} 
                         for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'upload_trend': upload_trend,
            'file_types': file_types,
            'size_distribution': size_distribution,
            'hourly_pattern': hourly_pattern
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ======================== Enhanced Result Analytics =========================
@app.route('/api/analytics/results')
@login_required
def get_result_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # Color detection distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%Red%' THEN 'Red'
                    WHEN Result_data LIKE '%Green%' THEN 'Green'
                    WHEN Result_data LIKE '%Blue%' THEN 'Blue'
                    WHEN Result_data LIKE '%Yellow%' THEN 'Yellow'
                    WHEN Result_data LIKE '%Purple%' THEN 'Purple'
                    ELSE 'Mixed/Other'
                END as dominant_color,
                COUNT(*) as count
            FROM result
            GROUP BY dominant_color
            ORDER BY count DESC
        """)
        color_distribution = [{'color': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Processing success rate over time
        cur.execute("""
            SELECT DATE(processed_at) as date, COUNT(*) as total_processed
            FROM result
            WHERE processed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(processed_at)
            ORDER BY date
        """)
        processing_trend = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1]} 
                           for row in cur.fetchall()]
        
        # File type vs color correlation
        cur.execute("""
            SELECT 
                CASE 
                    WHEN r.Result_data LIKE '%jpg%' OR r.Result_data LIKE '%jpeg%' THEN 'JPEG'
                    WHEN r.Result_data LIKE '%png%' THEN 'PNG'
                    WHEN r.Result_data LIKE '%gif%' THEN 'GIF'
                    ELSE 'Other'
                END as file_type,
                CASE 
                    WHEN r.Result_data LIKE '%Red%' THEN 'Red'
                    WHEN r.Result_data LIKE '%Green%' THEN 'Green'
                    WHEN r.Result_data LIKE '%Blue%' THEN 'Blue'
                    ELSE 'Other'
                END as color,
                COUNT(*) as count
            FROM result r
            GROUP BY file_type, color
            ORDER BY count DESC
        """)
        correlation_data = [{'file_type': row[0], 'color': row[1], 'count': row[2]} 
                           for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'color_distribution': color_distribution,
            'processing_trend': processing_trend,
            'correlation_data': correlation_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ======================== Enhanced Subscription Analytics =========================
@app.route('/api/analytics/subscriptions')
@login_required
def get_subscription_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # Subscription distribution
        cur.execute("""
            SELECT Plan_Type, COUNT(*) as count, SUM(Amount_Paid) as revenue
            FROM subscription
            WHERE Status = 'active'
            GROUP BY Plan_Type
            ORDER BY count DESC
        """)
        plan_distribution = []
        for row in cur.fetchall():
            plan_distribution.append({
                'plan': row[0].title(),
                'count': row[1],
                'revenue': float(row[2] or 0)
            })
        
        # Revenue trend (last 12 months)
        cur.execute("""
            SELECT 
                DATE_FORMAT(Start_Date, '%Y-%m') as month,
                SUM(Amount_Paid) as revenue,
                COUNT(*) as subscriptions
            FROM subscription
            WHERE Start_Date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
            GROUP BY DATE_FORMAT(Start_Date, '%Y-%m')
            ORDER BY month
        """)
        revenue_trend = []
        for row in cur.fetchall():
            revenue_trend.append({
                'month': row[0],
                'revenue': float(row[1] or 0),
                'subscriptions': row[2]
            })
        
        # Subscription duration analysis
        cur.execute("""
            SELECT 
                CASE 
                    WHEN DATEDIFF(End_Date, Start_Date) <= 30 THEN 'Monthly'
                    WHEN DATEDIFF(End_Date, Start_Date) <= 90 THEN 'Quarterly'
                    WHEN DATEDIFF(End_Date, Start_Date) <= 365 THEN 'Yearly'
                    ELSE 'Long-term'
                END as duration_type,
                COUNT(*) as count
            FROM subscription
            GROUP BY duration_type
        """)
        duration_analysis = [{'duration': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'plan_distribution': plan_distribution,
            'revenue_trend': revenue_trend,
            'duration_analysis': duration_analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ======================== Enhanced Feedback Analytics =========================
@app.route('/api/feedback/analytics')
@login_required
def get_feedback_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # Get total feedback count and average rating
        cur.execute("SELECT COUNT(*) as total_count, AVG(Rating) as avg_rating FROM feedback")
        total_stats = cur.fetchone()
        total_count = total_stats[0] or 0
        avg_rating = float(total_stats[1] or 0)
        
        # Get feedback types distribution
        cur.execute("""
            SELECT 
                Feedback_Type,
                COUNT(*) as count
            FROM feedback 
            GROUP BY Feedback_Type
        """)
        type_data = cur.fetchall()
        
        # Initialize type counts
        type_counts = {'bug': 0, 'feature': 0, 'general': 0, 'result': 0}
        for row in type_data:
            if row[0] in type_counts:
                type_counts[row[0]] = row[1]
        
        # Get rating distribution
        cur.execute("""
            SELECT 
                Rating,
                COUNT(*) as count
            FROM feedback 
            GROUP BY Rating 
            ORDER BY Rating
        """)
        rating_data = cur.fetchall()
        
        # Initialize rating counts
        rating_counts = []
        for i in range(1, 6):
            count = 0
            for row in rating_data:
                if row[0] == i:
                    count = row[1]
                    break
            rating_counts.append({'rating': i, 'count': count})
        
        # Get sentiment analysis
        cur.execute("SELECT Feedback_Text FROM feedback WHERE Feedback_Text IS NOT NULL")
        feedback_texts = cur.fetchall()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'happy', 'satisfied', 'awesome', 'wonderful']
        negative_words = ['bad', 'poor', 'hate', 'worst', 'terrible', 'disappointed', 'awful', 'issue', 'problem', 'broken']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for row in feedback_texts:
            if row[0]:
                text = row[0].lower()
                if any(word in text for word in positive_words):
                    positive_count += 1
                elif any(word in text for word in negative_words):
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Get general vs result feedback counts
        cur.execute("""
            SELECT 
                COUNT(*) as general_count
            FROM feedback 
            WHERE Result_ID IS NULL
        """)
        general_count = cur.fetchone()[0] or 0
        
        cur.execute("""
            SELECT 
                COUNT(*) as result_count
            FROM feedback 
            WHERE Result_ID IS NOT NULL
        """)
        result_count = cur.fetchone()[0] or 0
        
        cur.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total': total_count,
                'avg_rating': avg_rating,
                'types': type_counts,
                'general': general_count,
                'result': result_count,
                'sentiment': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'ratings': rating_counts
            },
            'trend': []  # No trend data since we don't have created_at column
        })
        
    except Exception as e:
        print(f"Error in get_feedback_analytics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
            

# ======================== Admin Analytics =========================
@app.route('/api/analytics/admin')
@login_required
def get_admin_analytics():
    try:
        cur = mysql.connection.cursor()
        
        # Admin profile data
        if 'admin_id' in session:
            admin_id = session['admin_id']
            cur.execute("SELECT Admin_ID, Name, Email, Salary FROM admin WHERE Admin_ID = %s", (admin_id,))
            admin_data = cur.fetchone()
            
            if admin_data:
                admin_profile = {
                    'admin_id': admin_data[0],
                    'name': admin_data[1],
                    'email': admin_data[2],
                    'salary': float(admin_data[3] or 0)
                }
            else:
                admin_profile = None
        else:
            admin_profile = None
        
        # System performance metrics
        cur.execute("SELECT COUNT(*) FROM user_management")
        total_users = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM image")
        total_images = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM result")
        total_results = cur.fetchone()[0]
        
        cur.execute("SELECT SUM(Amount_Paid) FROM subscription WHERE Status = 'active'")
        total_revenue = float(cur.fetchone()[0] or 0)
        
        system_metrics = {
            'total_users': total_users,
            'total_images': total_images,
            'total_results': total_results,
            'total_revenue': total_revenue,
            'processing_efficiency': round((total_results / max(total_images, 1)) * 100, 1)
        }
        
        cur.close()
        
        return jsonify({
            'success': True,
            'admin_profile': admin_profile,
            'system_metrics': system_metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500











@app.route('/api/feedback/general')
@login_required
def get_general_feedback():
    try:
        cur = mysql.connection.cursor()
        
        # Get general feedback stats (feedback without result_id)
        cur.execute("""
            SELECT 
                COUNT(*) as count,
                AVG(Rating) as avg_rating,
                SUM(CASE WHEN Feedback_Type = 'bug' THEN 1 ELSE 0 END) as bug_count,
                SUM(CASE WHEN Feedback_Type = 'feature' THEN 1 ELSE 0 END) as feature_count,
                SUM(CASE WHEN Feedback_Type = 'general' THEN 1 ELSE 0 END) as general_count,
                SUM(CASE WHEN Feedback_Text LIKE '%good%' OR Feedback_Text LIKE '%great%' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN Feedback_Text LIKE '%bad%' OR Feedback_Text LIKE '%poor%' THEN 1 ELSE 0 END) as negative,
                SUM(Rating = 1) as rating_1,
                SUM(Rating = 2) as rating_2,
                SUM(Rating = 3) as rating_3,
                SUM(Rating = 4) as rating_4,
                SUM(Rating = 5) as rating_5
            FROM feedback
            WHERE Result_ID IS NULL
        """)
        row = cur.fetchone()
        
        # Get trend data (last 7 days)
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count,
                AVG(Rating) as avg_rating
            FROM feedback
            WHERE Result_ID IS NULL
              AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        trend = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1], 'avg_rating': float(row[2] or 0)} 
                for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'data': {
                'count': row[0],
                'avg_rating': float(row[1] or 0),
                'types': {
                    'bug': row[2],
                    'feature': row[3],
                    'general': row[4]
                },
                'sentiment': {
                    'positive': row[5],
                    'negative': row[6],
                    'neutral': row[0] - row[5] - row[6]
                },
                'ratings': [
                    {'rating': 1, 'count': row[7]},
                    {'rating': 2, 'count': row[8]},
                    {'rating': 3, 'count': row[9]},
                    {'rating': 4, 'count': row[10]},
                    {'rating': 5, 'count': row[11]}
                ],
                'trend': trend
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/feedback/result')
@login_required
def get_result_feedback():
    try:
        cur = mysql.connection.cursor()
        
        # Get result feedback stats (feedback with result_id)
        cur.execute("""
            SELECT 
                COUNT(*) as count,
                AVG(Rating) as avg_rating,
                SUM(CASE WHEN Feedback_Type = 'bug' THEN 1 ELSE 0 END) as bug_count,
                SUM(CASE WHEN Feedback_Type = 'feature' THEN 1 ELSE 0 END) as feature_count,
                SUM(CASE WHEN Feedback_Type = 'result' THEN 1 ELSE 0 END) as result_count,
                SUM(CASE WHEN Feedback_Text LIKE '%good%' OR Feedback_Text LIKE '%great%' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN Feedback_Text LIKE '%bad%' OR Feedback_Text LIKE '%poor%' THEN 1 ELSE 0 END) as negative,
                SUM(Rating = 1) as rating_1,
                SUM(Rating = 2) as rating_2,
                SUM(Rating = 3) as rating_3,
                SUM(Rating = 4) as rating_4,
                SUM(Rating = 5) as rating_5
            FROM feedback
            WHERE Result_ID IS NOT NULL
        """)
        row = cur.fetchone()
        
        # Get trend data (last 7 days)
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count,
                AVG(Rating) as avg_rating
            FROM feedback
            WHERE Result_ID IS NOT NULL
              AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        trend = [{'date': row[0].strftime('%Y-%m-%d'), 'count': row[1], 'avg_rating': float(row[2] or 0)} 
                for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'data': {
                'count': row[0],
                'avg_rating': float(row[1] or 0),
                'types': {
                    'bug': row[2],
                    'feature': row[3],
                    'result': row[4]
                },
                'sentiment': {
                    'positive': row[5],
                    'negative': row[6],
                    'neutral': row[0] - row[5] - row[6]
                },
                'ratings': [
                    {'rating': 1, 'count': row[7]},
                    {'rating': 2, 'count': row[8]},
                    {'rating': 3, 'count': row[9]},
                    {'rating': 4, 'count': row[10]},
                    {'rating': 5, 'count': row[11]}
                ],
                'trend': trend
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    

# ======================== Statistics Route =========================
@app.route('/api/stats')
@login_required
def get_stats():
    try:
        stats = {}
        tables = ['user_management', 'image', 'result', 'subscription', 'feedback', 'admin']
        
        cur = mysql.connection.cursor()
        
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                stats[table] = count
            except:
                stats[table] = 0
        
        cur.close()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error fetching statistics'
        }), 500

@app.route('/hash_passwords', methods=['GET'])
def hash_passwords():
    try:
        cur = mysql.connection.cursor()
        
        # Get all users with plain text passwords
        cur.execute("SELECT User_ID, Password FROM user_management")
        users = cur.fetchall()
        
        for user_id, plain_password in users:
            if not plain_password.startswith('$2b$'):  # Check if password is not already hashed
                hashed_password = generate_password_hash(plain_password)
                cur.execute("UPDATE user_management SET Password = %s WHERE User_ID = %s", 
                          (hashed_password, user_id))
        
        mysql.connection.commit()
        cur.close()
        return "Passwords hashed successfully"
    except Exception as e:
        return str(e), 500

@app.route('/view')
@login_required
def view():
    try:
        admin_name = "Admin"
        role = session.get('login_role', 'user')
        
        if 'admin_id' in session:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Name FROM admin WHERE Admin_ID = %s", (session['admin_id'],))
            result = cur.fetchone()
            if result:
                admin_name = result[0]
            cur.close()

        return render_template('View.html', admin_username=admin_name, login_role=role)
    
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('View.html', admin_username="Admin", login_role="user")

# ======================= Visualization ======================
@app.route('/api/visualize/<table_name>')
@login_required
def visualize_table(table_name):
    try:
        mapping = {
            'users': 'user_management',
            'images': 'image',
            'results': 'result',
            'subscriptions': 'subscription',
            'feedback': 'feedback',
            'admin': 'admin'
        }

        if table_name not in mapping:
            return jsonify({'success': False, 'message': 'Invalid table name'})

        table = mapping[table_name]
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT * FROM {table}")
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
        data = [dict(zip(columns, row)) for row in rows]
        cur.close()

        # ========== Find suitable non-ID categorical column ==========
        def is_suitable(col_name, values):
            if "id" in col_name.lower():
                return False  # skip ID fields
            unique_vals = list(set(values))
            return 1 < len(unique_vals) <= 20 and all(isinstance(v, str) or isinstance(v, int) for v in unique_vals)

        for col in columns:
            values = [row[col] for row in data if row[col] is not None]
            if is_suitable(col, values):
                counts = {}
                for v in values:
                    v = str(v)
                    counts[v] = counts.get(v, 0) + 1
                return jsonify({
                    'success': True,
                    'label': col,
                    'labels': list(counts.keys()),
                    'counts': list(counts.values())
                })

        return jsonify({'success': False, 'message': 'No suitable column found to plot'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_recent_uploaded_times', methods=['GET'])
@login_required
def get_recent_uploaded_times():
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT image_id, uploaded_image, uploaded_at
            FROM image
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            LIMIT 3
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()

        timestamps = [{
            'image_id': row[0],
            'image_url': row[1],
            'uploaded_at': row[2].strftime('%Y-%m-%d %H:%M:%S')
        } for row in rows]

        return jsonify({'success': True, 'timestamps': timestamps})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_recent_results')
@login_required
def get_recent_results():
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT Result_ID, processed_at 
            FROM result 
            WHERE User_ID = %s 
            ORDER BY processed_at DESC 
            LIMIT 5
        """, (user_id,))
        results = [{
            'result_id': row[0], 
            'processed_at': row[1].strftime('%Y-%m-%d %H:%M:%S')
        } for row in cur.fetchall()]
        cur.close()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# FIXED FEEDBACK SUBMISSION ROUTE
@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        # Verify user is logged in
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Please login first'}), 401

        user_id = session['user_id']
        
        # Get form data with proper validation
        feedback_type = request.form.get('Feedback_Type', '').strip().lower()
        rating = request.form.get('rating', type=int)
        feedback_text = request.form.get('Feedback_Text', '').strip()
        result_id = request.form.get('result_id')

        print(f"Received feedback - Type: {feedback_type}, Rating: {rating}, Text: {feedback_text}, Result ID: {result_id}")  # Debug log

        # Validate required fields
        if not all([feedback_type, rating, feedback_text]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        # Validate feedback type
        valid_feedback_types = ['general', 'result', 'bug', 'suggestion']
        if feedback_type not in valid_feedback_types:
            return jsonify({
                'success': False,
                'message': f'Invalid feedback type. Must be one of: {", ".join(valid_feedback_types)}'
            }), 400

        # Validate rating
        if not 1 <= rating <= 5:
            return jsonify({'success': False, 'message': 'Rating must be between 1 and 5'}), 400

        # For result feedback, verify the result exists and belongs to user
        if feedback_type == 'result':
            cur = mysql.connection.cursor()
            
            # Use provided result_id or get the most recent one
            if not result_id:
                cur.execute("""
                    SELECT Result_ID FROM result 
                    WHERE User_ID = %s 
                    ORDER BY processed_at DESC 
                    LIMIT 1
                """, (user_id,))
                recent_result = cur.fetchone()
                if recent_result:
                    result_id = recent_result[0]
                else:
                    cur.close()
                    return jsonify({
                        'success': False,
                        'message': 'No recent results found for result feedback'
                    }), 400

            # Verify the result belongs to the user
            if result_id:
                cur.execute("""
                    SELECT Result_ID FROM result 
                    WHERE Result_ID = %s AND User_ID = %s
                """, (result_id, user_id))
                if not cur.fetchone():
                    cur.close()
                    return jsonify({
                        'success': False,
                        'message': 'Invalid result ID'
                    }), 400
            cur.close()

        # Generate feedback ID
        feedback_id = random.randint(1000, 9999)

        # Insert into database
        cur = mysql.connection.cursor()
        
        if feedback_type == 'result' and result_id:
            cur.execute("""
                INSERT INTO feedback (
                    Feedback_ID, Rating, Feedback_Type, 
                    Feedback_Text, User_ID, Result_ID
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (feedback_id, rating, feedback_type, feedback_text, user_id, result_id))
        else:
            cur.execute("""
                INSERT INTO feedback (
                    Feedback_ID, Rating, Feedback_Type, 
                    Feedback_Text, User_ID
                ) VALUES (%s, %s, %s, %s, %s)
            """, (feedback_id, rating, feedback_type, feedback_text, user_id))

        mysql.connection.commit()
        cur.close()

        return jsonify({
            'success': True, 
            'message': 'Feedback submitted successfully!'
        })

    except Exception as e:
        mysql.connection.rollback()
        print(f"Error in submit_feedback: {str(e)}")  # Debug log
        return jsonify({
            'success': False, 
            'message': 'An error occurred while submitting feedback'
        }), 500

# FIXED FEEDBACK RETRIEVAL ROUTE
@app.route('/get_feedback', methods=['GET'])
@login_required
def get_feedback():
    try:
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'message': 'Please login first',
                'login_required': True  # Add this flag for frontend handling
            }), 401

        user_id = session['user_id']
        
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT Feedback_ID, Rating, Feedback_Type, Feedback_Text, User_ID, Result_ID
            FROM feedback 
            WHERE User_ID = %s
            ORDER BY Feedback_ID DESC
        """, (user_id,))
        
        columns = [col[0] for col in cur.description]
        feedback_data = []
        
        for row in cur.fetchall():
            feedback_data.append(dict(zip(columns, row)))
        
        cur.close()
        
        return jsonify({
            'success': True,
            'data': feedback_data
        })
    
    except Exception as e:
        app.logger.error(f"Error in get_feedback: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error retrieving feedback',
            'error': str(e)
        }), 500    

def get_dominant_color(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    most_common = Counter(map(tuple, pixels)).most_common(1)[0][0]
    r, g, b = most_common
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Mixed"

@app.route('/save_image_info', methods=['POST'])
@login_required
def save_image_info():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Empty file name'}), 400
            
        user_id = session['user_id']
        cur = mysql.connection.cursor()

        # Get active subscription with proper null handling
        cur.execute("""
            SELECT Subscription_ID, Plan_Type, Start_Date, End_Date, 
                   COALESCE(Upload_Limit, 0) as Upload_Limit, 
                   COALESCE(Uploads_Used, 0) as Uploads_Used, 
                   Status
            FROM subscription 
            WHERE User_ID = %s AND Status = 'active'
            ORDER BY End_Date DESC
            LIMIT 1
        """, (user_id,))
        subscription = cur.fetchone()

        # If no active subscription, check free tier limit
        if not subscription:
            cur.execute("SELECT COUNT(*) FROM image WHERE user_id = %s", (user_id,))
            upload_count = cur.fetchone()[0] or 0  # Ensure we get 0 if None
            
            if upload_count >= 3:  # ✅ This blocks the 4th upload
                cur.close()
                return jsonify({
                    'success': False,
                    'message': 'You have reached the maximum free upload limit (3 images). Please subscribe to continue.',
                    'redirect': '/#payment',
                    'limit_reached': True
                }), 403
        else:
            # Subscription exists - check both limits (time and upload count)
            sub_id, plan_type, start_date, end_date, upload_limit, uploads_used, status = subscription
            
            # Convert dates to datetime objects if they aren't already
            if isinstance(end_date, dt.date) and not isinstance(end_date, dt.datetime):
                end_date = dt.datetime.combine(end_date, dt.datetime.min.time())
            
            now = dt.datetime.now()

            # Check if subscription has expired by time
            if now > end_date:
                cur.execute("""
                    UPDATE subscription 
                    SET Status = 'expired',
                        End_Date = NOW()
                    WHERE Subscription_ID = %s
                """, (sub_id,))
                mysql.connection.commit()
                
                cur.close()
                return jsonify({
                    'success': False,
                    'message': f'Your {plan_type} subscription has expired (time limit reached).',
                    'redirect': '/#payment',
                    'subscription_expired': True,
                    'reason': 'time'
                }), 403

            # Check if upload limit reached (upload_limit of -1 means unlimited)
            if upload_limit != -1 and uploads_used >= upload_limit:
                cur.execute("""
                    UPDATE subscription 
                    SET Status = 'expired',
                        End_Date = NOW()
                    WHERE Subscription_ID = %s
                """, (sub_id,))
                mysql.connection.commit()
                
                cur.close()
                return jsonify({
                    'success': False,
                    'message': f'Your {plan_type} subscription has expired (upload limit reached).',
                    'redirect': '/#payment',
                    'subscription_expired': True,
                    'reason': 'uploads'
                }), 403

        # File validation and processing
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
        upload_folder = os.path.join(app.static_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Database operations
        image_id = random.randint(1000, 2000)
        file_size_kb = round(os.path.getsize(file_path)/1024, 2)
        
        # Store image info
        cur.execute("""
            INSERT INTO image (image_id, uploaded_image, file_type, file_size, user_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (image_id, filename, file_ext, file_size_kb, user_id))

        # Process and store results
        dominant_color = get_dominant_color(file_path)
        result_data = f"Dominant Color: {dominant_color}, Type: {file_ext}, Size: {file_size_kb} KB"
        result_id = random.randint(2000, 3000)

        cur.execute("""
            INSERT INTO result (result_id, result_data, image_id, user_id)
            VALUES (%s, %s, %s, %s)
        """, (result_id, result_data, image_id, user_id))

        # Update upload count in subscription if exists
        if subscription:
            sub_id = subscription[0]
            cur.execute("""
                UPDATE subscription 
                SET Uploads_Used = Uploads_Used + 1
                WHERE Subscription_ID = %s
            """, (sub_id,))

        mysql.connection.commit()
        
        # Calculate remaining uploads
        if subscription:
            upload_limit = subscription[4]
            uploads_used = subscription[5] + 1  # +1 for the current upload
            if upload_limit == -1:
                remaining_uploads = "Unlimited"
                message = "You have unlimited uploads remaining in your subscription."
            else:
                remaining_uploads = max(0, upload_limit - uploads_used)
                message = f"Your {remaining_uploads} images are remaining to upload."
        else:
            cur.execute("SELECT COUNT(*) FROM image WHERE user_id = %s", (user_id,))
            upload_count = cur.fetchone()[0] or 0
            remaining_uploads = max(0, 3 - upload_count)
            message = f"Your remaining free trial is {remaining_uploads} images."

        cur.close()

        return jsonify({
            'success': True,
            'message': message,
            'image_id': image_id,
            'result_id': result_id,
            'file_url': f"/static/uploads/{filename}",
            'result_data': result_data,
            'remaining_uploads': remaining_uploads,
            'has_subscription': bool(subscription)
        })

    except Exception as e:
        mysql.connection.rollback()
        app.logger.error(f"Error in save_image_info: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"An error occurred: {str(e)}"
        }), 500




@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'message': 'Please login first'}), 401

    user_id = session['user_id']
    cur = mysql.connection.cursor()

    # ✅ Check if user has an active subscription
    cur.execute("SELECT Plan_Type, Upload_Limit, Uploads_Used FROM subscription WHERE User_ID = %s AND Status = 'Active'", (user_id,))
    sub = cur.fetchone()

    if sub:
        plan_type = sub[0]
        upload_limit = sub[1] if sub[1] is not None else float('inf')  # unlimited if None
        uploads_used = sub[2] if sub[2] is not None else 0
        
        if upload_limit != float('inf') and uploads_used >= upload_limit:
            return jsonify({'message': 'You have used all your uploads for this plan. Please upgrade.'}), 403

        # ✅ Process image upload (only if under the limit)
        # --- Add your image saving logic here ---

        # Update upload count
        cur.execute("UPDATE subscription SET Uploads_Used = Uploads_Used + 1 WHERE User_ID = %s", (user_id,))
        mysql.connection.commit()
        cur.close()

        return jsonify({
            'message': f'Upload successful! You have used {uploads_used + 1} of {upload_limit if upload_limit else "∞"} uploads in your {plan_type} plan.'
        }), 200

    # ✅ Free trial logic if no subscription found
    if 'free_trial_uploads' not in session:
        session['free_trial_uploads'] = 0

    uploads = session['free_trial_uploads']

    if uploads >= 3:
        cur.close()
        return jsonify({
            'message': "You have reached the maximum free upload limit (3 images). Please subscribe to continue."
        }), 403

    # ✅ Allow image upload for free users (under 3 uploads)
    # --- Add your image saving logic here ---

    session['free_trial_uploads'] += 1
    uploads = session['free_trial_uploads']

    if uploads == 1:
        message = "Upload successful! You have 2 remaining images in free trial."
    elif uploads == 2:
        message = "Upload successful! You have 1 remaining image in free trial."
    elif uploads == 3:
        message = "Upload successful! You have used your free trial."

    cur.close()
    return jsonify({'message': message}), 200


# ==================== Subscription Routes ====================
@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        data = request.get_json()
        plan = data.get('plan')
        price_map = {
            'basic': 999,
            'pro': 2499,
            'enterprise': 9999
        }

        if plan not in price_map:
            return jsonify({'error': 'Invalid plan selected'}), 400

        # Store selected plan in session
        session['selected_plan'] = plan

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='payment',
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f'{plan.capitalize()} Plan',
                    },
                    'unit_amount': price_map[plan],
                },
                'quantity': 1,
            }],
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('index', _external=True),
        )

        return jsonify({'checkout_url': checkout_session.url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/payment-success')
@login_required
def payment_success():
    try:
        if 'user_id' not in session:
            return redirect(url_for('index'))

        user_id = session['user_id']
        plan_type = session.get('selected_plan', 'basic')

        plans = {
            'basic': {'amount': 999, 'upload_limit': 100, 'duration_days': 30},
            'pro': {'amount': 2999, 'upload_limit': 500, 'duration_days': 30},
            'enterprise': {'amount': 9999, 'upload_limit': 1500, 'duration_days': 30}
        }

        subscription_id = random.randint(3000, 4000)
        start_date = datetime.now()
        end_date = start_date + timedelta(days=plans[plan_type]['duration_days'])
        payment_method = "Stripe"  # Explicitly defined

        cur = mysql.connection.cursor()
        
        # Debug print to verify values before insertion
        print(f"""Inserting subscription:
            ID: {subscription_id}
            Plan: {plan_type}
            Start: {start_date}
            End: {end_date}
            Payment Method: {payment_method}
            User: {user_id}
        """)

        # Modified INSERT statement with all columns explicitly listed
        cur.execute("""
            INSERT INTO subscription (
                Subscription_ID, Plan_Type, Start_Date, End_Date, Status, 
                Upload_Limit, Uploads_Used, Amount_Paid, Payment_Method, User_ID
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            subscription_id, 
            plan_type, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'), 
            'active',
            plans[plan_type]['upload_limit'], 
            0,  # Uploads_Used
            plans[plan_type]['amount'] / 100,
            payment_method,  # Explicitly included
            user_id
        ))
        
        mysql.connection.commit()
        cur.close()

        # Verify insertion
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT Payment_Method FROM subscription 
            WHERE Subscription_ID = %s
        """, (subscription_id,))
        result = cur.fetchone()
        print(f"Verified payment method in DB: {result[0] if result else 'Not found'}")
        cur.close()

        if 'selected_plan' in session:
            del session['selected_plan']

        return render_template('pay.html',
            subscription_id=subscription_id,
            plan_type=plan_type.capitalize(),
            payment_method=payment_method,  # Pass to template
            amount_paid=f"{plans[plan_type]['amount'] / 100:.2f}",
            upload_limit="Unlimited" if plans[plan_type]['upload_limit'] == -1 else plans[plan_type]['upload_limit'],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

    except Exception as e:
        mysql.connection.rollback()
        print(f"Payment processing error: {str(e)}")
        return f"Error processing payment: {str(e)}", 500




@app.route('/get_payment_info')
@login_required
def get_payment_info():
    try:
        user_id = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT Subscription_ID, Plan_Type, Start_Date, End_Date, Upload_Limit, Amount_Paid
            FROM subscription
            WHERE User_ID = %s
            ORDER BY Start_Date DESC
        """, (user_id,))
        columns = [col[0] for col in cur.description]
        rows = cur.fetchall()
        cur.close()

        # Format dates to exclude time
        data = []
        for row in rows:
            record = dict(zip(columns, row))
            record['Start_Date'] = record['Start_Date'].strftime('%Y-%m-%d')
            record['End_Date'] = record['End_Date'].strftime('%Y-%m-%d')
            data.append(record)

        return jsonify({'success': True, 'data': data})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# ======================== Legacy Visualization API Routes ========================
@app.route('/api/visualization/user_stats')
@login_required
def user_stats():
    try:
        cur = mysql.connection.cursor()
        
        # Get total user count
        cur.execute("SELECT COUNT(*) FROM user_management")
        total_users = cur.fetchone()[0]
        
        # User activity based on image uploads
        cur.execute("""
            SELECT 
                CASE 
                    WHEN upload_count = 0 THEN 'No Uploads'
                    WHEN upload_count = 1 THEN '1 Upload'
                    WHEN upload_count BETWEEN 2 AND 5 THEN '2-5 Uploads'
                    WHEN upload_count BETWEEN 6 AND 10 THEN '6-10 Uploads'
                    ELSE '10+ Uploads'
                END as activity_level,
                COUNT(*) as user_count
            FROM (
                SELECT u.User_ID, COALESCE(COUNT(i.image_id), 0) as upload_count
                FROM user_management u
                LEFT JOIN image i ON u.User_ID = i.user_id
                GROUP BY u.User_ID
            ) user_activity
            GROUP BY activity_level
            ORDER BY 
                CASE activity_level
                    WHEN 'No Uploads' THEN 1
                    WHEN '1 Upload' THEN 2
                    WHEN '2-5 Uploads' THEN 3
                    WHEN '6-10 Uploads' THEN 4
                    ELSE 5
                END
        """)
        user_activity = [{'level': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Subscription status distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN s.Subscription_ID IS NOT NULL AND s.Status = 'active' THEN 'Active Subscription'
                    WHEN s.Subscription_ID IS NOT NULL AND s.Status = 'inactive' THEN 'Inactive Subscription'
                    ELSE 'No Subscription'
                END as subscription_status,
                COUNT(*) as user_count
            FROM user_management u
            LEFT JOIN subscription s ON u.User_ID = s.User_ID
            GROUP BY subscription_status
            ORDER BY 
                CASE subscription_status
                    WHEN 'Active Subscription' THEN 1
                    WHEN 'Inactive Subscription' THEN 2
                    ELSE 3
                END
        """)
        subscription_status = [{'status': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'total_users': total_users,
            'user_activity': user_activity,
            'subscription_status': subscription_status
        })
        
    except Exception as e:
        print(f"Error in user_stats: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
                
@app.route('/api/user_name_lengths')
@login_required
def user_name_lengths():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT 
                CASE 
                    WHEN LENGTH(Name) BETWEEN 1 AND 5 THEN '1-5 chars'
                    WHEN LENGTH(Name) BETWEEN 6 AND 10 THEN '6-10 chars'
                    ELSE '10+ chars'
                END as name_length,
                COUNT(*) as count
            FROM user_management
            GROUP BY name_length
        """)
        results = [{"length": row[0], "count": row[1]} for row in cur.fetchall()]
        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/visualization/subscription_stats')
@login_required
def subscription_stats():
    try:
        cur = mysql.connection.cursor()
        
        # Subscription distribution - Fixed status check
        cur.execute("""
            SELECT Plan_Type, COUNT(*) as count, SUM(Amount_Paid) as revenue
            FROM subscription
            WHERE Status = 'active'
            GROUP BY Plan_Type
            ORDER BY count DESC
        """)
        subscriptions = []
        for row in cur.fetchall():
            subscriptions.append({
                'plan': row[0].title() if row[0] else 'Unknown',
                'count': row[1],
                'revenue': float(row[2]) if row[2] else 0
            })
        
        # Total revenue
        cur.execute("SELECT SUM(Amount_Paid) FROM subscription WHERE Status = 'active'")
        total_revenue = float(cur.fetchone()[0] or 0)
        
        # Get subscription trend data (last 12 months)
        cur.execute("""
            SELECT 
                DATE_FORMAT(Start_Date, '%Y-%m') as month,
                COUNT(*) as count,
                SUM(Amount_Paid) as revenue
            FROM subscription
            WHERE Start_Date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
            GROUP BY DATE_FORMAT(Start_Date, '%Y-%m')
            ORDER BY month
        """)
        trend_data = []
        for row in cur.fetchall():
            trend_data.append({
                'month': row[0],
                'count': row[1],
                'revenue': float(row[2] or 0)
            })
        
        cur.close()
        
        return jsonify({
            'success': True,
            'subscriptions': subscriptions,
            'total_revenue': total_revenue,
            'trend': trend_data
        })
    except Exception as e:
        print(f"Error in subscription_stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/visualization/image_stats')
@login_required
def image_stats():
    try:
        cur = mysql.connection.cursor()
        
        # Image uploads (last 30 days) with better date formatting
        cur.execute("""
            SELECT DATE(uploaded_at) as date, COUNT(*) as count 
            FROM image 
            WHERE uploaded_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY DATE(uploaded_at)
            ORDER BY date
        """)
        uploads = [{'date': row[0].strftime('%b %d'), 'count': row[1]} for row in cur.fetchall()]
        
        # File type distribution with better categorization
        cur.execute("""
            SELECT 
                CASE 
                    WHEN file_type IN ('jpg', 'jpeg') THEN 'JPEG'
                    WHEN file_type = 'png' THEN 'PNG'
                    WHEN file_type = 'gif' THEN 'GIF'
                    WHEN file_type = 'webp' THEN 'WebP'
                    WHEN file_type = 'bmp' THEN 'BMP'
                    ELSE 'Other'
                END as type,
                COUNT(*) as count 
            FROM image 
            GROUP BY type
            ORDER BY count DESC
        """)
        file_types = [{'type': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # File size distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN file_size < 100 THEN '< 100 KB'
                    WHEN file_size BETWEEN 100 AND 500 THEN '100-500 KB'
                    WHEN file_size BETWEEN 500 AND 1000 THEN '500KB-1MB'
                    WHEN file_size BETWEEN 1000 AND 5000 THEN '1-5 MB'
                    ELSE '> 5 MB'
                END as size_range,
                COUNT(*) as count
            FROM image
            GROUP BY size_range
            ORDER BY 
                CASE size_range
                    WHEN '< 100 KB' THEN 1
                    WHEN '100-500 KB' THEN 2
                    WHEN '500KB-1MB' THEN 3
                    WHEN '1-5 MB' THEN 4
                    ELSE 5
                END
        """)
        size_distribution = [{'range': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Hourly upload pattern
        cur.execute("""
            SELECT HOUR(uploaded_at) as hour, COUNT(*) as count
            FROM image
            GROUP BY HOUR(uploaded_at)
            ORDER BY hour
        """)
        hourly_pattern = [{'hour': f"{row[0]:02d}:00", 'count': row[1]} for row in cur.fetchall()]
        
        # User upload activity
        cur.execute("""
            SELECT 
                CASE 
                    WHEN upload_count = 1 THEN '1 Upload'
                    WHEN upload_count BETWEEN 2 AND 5 THEN '2-5 Uploads'
                    WHEN upload_count BETWEEN 6 AND 10 THEN '6-10 Uploads'
                    ELSE '10+ Uploads'
                END as activity_level,
                COUNT(*) as user_count
            FROM (
                SELECT user_id, COUNT(*) as upload_count
                FROM image
                GROUP BY user_id
            ) user_uploads
            GROUP BY activity_level
            ORDER BY 
                CASE activity_level
                    WHEN '1 Upload' THEN 1
                    WHEN '2-5 Uploads' THEN 2
                    WHEN '6-10 Uploads' THEN 3
                    ELSE 4
                END
        """)
        user_activity = [{'level': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Total statistics
        cur.execute("SELECT COUNT(*) FROM image")
        total_images = cur.fetchone()[0]
        
        cur.execute("SELECT AVG(file_size) FROM image")
        avg_file_size = round(float(cur.fetchone()[0] or 0), 2)
        
        cur.close()
        
        return jsonify({
            'success': True,
            'uploads': uploads,
            'file_types': file_types,
            'size_distribution': size_distribution,
            'hourly_pattern': hourly_pattern,
            'user_activity': user_activity,
            'total_images': total_images,
            'avg_file_size': avg_file_size
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/visualization/result_stats')
@login_required
def result_stats():
    try:
        cur = mysql.connection.cursor()
        
        # Enhanced dominant color analysis
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%Red%' THEN 'Red'
                    WHEN Result_data LIKE '%Green%' THEN 'Green'
                    WHEN Result_data LIKE '%Blue%' THEN 'Blue'
                    WHEN Result_data LIKE '%Yellow%' THEN 'Yellow'
                    WHEN Result_data LIKE '%Purple%' THEN 'Purple'
                    WHEN Result_data LIKE '%Orange%' THEN 'Orange'
                    WHEN Result_data LIKE '%Pink%' THEN 'Pink'
                    WHEN Result_data LIKE '%Brown%' THEN 'Brown'
                    WHEN Result_data LIKE '%Black%' THEN 'Black'
                    WHEN Result_data LIKE '%White%' THEN 'White'
                    WHEN Result_data LIKE '%Gray%' OR Result_data LIKE '%Grey%' THEN 'Gray'
                    ELSE 'Mixed/Other'
                END as color,
                COUNT(*) as count
            FROM result
            GROUP BY color
            ORDER BY count DESC
        """)
        colors = [{'color': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # File type analysis from results
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%jpg%' OR Result_data LIKE '%jpeg%' THEN 'JPEG'
                    WHEN Result_data LIKE '%png%' THEN 'PNG'
                    WHEN Result_data LIKE '%gif%' THEN 'GIF'
                    WHEN Result_data LIKE '%webp%' THEN 'WebP'
                    WHEN Result_data LIKE '%bmp%' THEN 'BMP'
                    ELSE 'Other'
                END as file_type,
                COUNT(*) as count
            FROM result
            GROUP BY file_type
            ORDER BY count DESC
        """)
        result_file_types = [{'type': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Processing success rate (assuming successful if result exists)
        cur.execute("SELECT COUNT(*) FROM result")
        successful_results = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM image")
        total_images = cur.fetchone()[0]
        
        success_rate = round((successful_results / total_images * 100) if total_images > 0 else 0, 1)
        
        # Average file size
        cur.execute("SELECT AVG(file_size) FROM image")
        avg_file_size = round(float(cur.fetchone()[0] or 0), 2)
        
        # Most active processing hours
        cur.execute("""
            SELECT HOUR(i.uploaded_at) as hour, COUNT(*) as count
            FROM image i
            JOIN result r ON i.image_id = r.image_id
            GROUP BY HOUR(i.uploaded_at)
            ORDER BY count DESC
            LIMIT 5
        """)
        peak_hours = [{'hour': f"{row[0]:02d}:00", 'count': row[1]} for row in cur.fetchall()]
        
        cur.close()
        
        return jsonify({
            'success': True,
            'colors': colors,
            'file_types': result_file_types,
            'avg_file_size': avg_file_size,
            'success_rate': success_rate,
            'total_processed': successful_results,
            'peak_hours': peak_hours
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/visualization/feedback_stats')
@login_required
def feedback_stats():
    try:
        cur = mysql.connection.cursor()

        # Count feedback types
        cur.execute("SELECT Feedback_Type, COUNT(*) as count FROM feedback GROUP BY Feedback_Type")
        types = cur.fetchall()
        type_stats = [{'type': row[0], 'count': row[1]} for row in types]

        # Count ratings
        cur.execute("SELECT Rating, COUNT(*) as count FROM feedback GROUP BY Rating ORDER BY Rating")
        ratings = cur.fetchall()
        rating_stats = [{'rating': row[0], 'count': row[1]} for row in ratings]

        # Sentiment analysis (simple version)
        cur.execute("SELECT Feedback_Text FROM feedback")
        all_text = " ".join(row[0] for row in cur.fetchall() if row[0])
        
        # Simple word frequency analysis (top 20 words)
        words = [word.lower() for word in all_text.split() if len(word) > 3 and word.isalpha()]
        word_freq = Counter(words).most_common(20)
        word_stats = [{'word': word[0], 'count': word[1]} for word in word_freq]

        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'happy', 'satisfied', 'awesome', 'wonderful']
        negative_words = ['bad', 'poor', 'hate', 'worst', 'terrible', 'disappointed', 'awful', 'issue', 'problem', 'broken']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        cur.execute("SELECT Feedback_Text FROM feedback")
        for row in cur.fetchall():
            if row[0]:
                text = row[0].lower()
                if any(word in text for word in positive_words):
                    positive_count += 1
                elif any(word in text for word in negative_words):
                    negative_count += 1
                else:
                    neutral_count += 1

        cur.close()
        
        return jsonify({
            'success': True,
            'feedback_types': type_stats,
            'ratings': rating_stats,
            'word_stats': word_stats,
            'sentiment': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin_data')
@login_required
def get_admin_profile():
    try:
        if 'admin_id' not in session:
            return jsonify({'success': False, 'message': 'Not logged in'}), 401

        admin_id = session['admin_id']
        cur = mysql.connection.cursor()
        cur.execute("SELECT Admin_ID, Name, Email, Salary FROM admin WHERE Admin_ID = %s", (admin_id,))
        row = cur.fetchone()
        cur.close()

        if row:
            return jsonify({
                'success': True,
                'admin_id': row[0],
                'name': row[1],
                'email': row[2],
                'salary': row[3]
            })
        else:
            return jsonify({'success': False, 'message': 'Admin not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/visualization/result_data')
@login_required
def get_result_data():
    try:
        cur = mysql.connection.cursor()
        
        # Enhanced dominant color distribution with better categorization
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%Red%' THEN 'Red'
                    WHEN Result_data LIKE '%Green%' THEN 'Green'
                    WHEN Result_data LIKE '%Blue%' THEN 'Blue'
                    WHEN Result_data LIKE '%Yellow%' THEN 'Yellow'
                    WHEN Result_data LIKE '%Purple%' THEN 'Purple'
                    WHEN Result_data LIKE '%Orange%' THEN 'Orange'
                    WHEN Result_data LIKE '%Pink%' THEN 'Pink'
                    WHEN Result_data LIKE '%Brown%' THEN 'Brown'
                    WHEN Result_data LIKE '%Black%' THEN 'Black'
                    WHEN Result_data LIKE '%White%' THEN 'White'
                    WHEN Result_data LIKE '%Gray%' OR Result_data LIKE '%Grey%' THEN 'Gray'
                    ELSE 'Mixed/Other'
                END as color,
                COUNT(*) as count
            FROM result
            GROUP BY color
            ORDER BY count DESC
        """)
        color_data = [{'name': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Enhanced file type distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%jpg%' OR Result_data LIKE '%jpeg%' THEN 'JPEG'
                    WHEN Result_data LIKE '%png%' THEN 'PNG'
                    WHEN Result_data LIKE '%gif%' THEN 'GIF'
                    WHEN Result_data LIKE '%webp%' THEN 'WebP'
                    WHEN Result_data LIKE '%bmp%' THEN 'BMP'
                    ELSE 'Other'
                END as file_type,
                COUNT(*) as count
            FROM result
            GROUP BY file_type
            ORDER BY count DESC
        """)
        file_type_data = [{'name': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # File size analysis from results
        cur.execute("""
            SELECT 
                CASE 
                    WHEN CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(Result_data, 'Size: ', -1), ' KB', 1) AS DECIMAL(10,2)) < 100 THEN '< 100 KB'
                    WHEN CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(Result_data, 'Size: ', -1), ' KB', 1) AS DECIMAL(10,2)) BETWEEN 100 AND 500 THEN '100-500 KB'
                    WHEN CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(Result_data, 'Size: ', -1), ' KB', 1) AS DECIMAL(10,2)) BETWEEN 500 AND 1000 THEN '500KB-1MB'
                    WHEN CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(Result_data, 'Size: ', -1), ' KB', 1) AS DECIMAL(10,2)) BETWEEN 1000 AND 5000 THEN '1-5 MB'
                    ELSE '> 5 MB'
                END as size_range,
                COUNT(*) as count
            FROM result
            WHERE Result_data LIKE '%Size:%'
            GROUP BY size_range
            ORDER BY 
                CASE size_range
                    WHEN '< 100 KB' THEN 1
                    WHEN '100-500 KB' THEN 2
                    WHEN '500KB-1MB' THEN 3
                    WHEN '1-5 MB' THEN 4
                    ELSE 5
                END
        """)
        size_data = [{'name': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        # Processing trend (last 30 days)
        cur.execute("""
            SELECT DATE(i.uploaded_at) as date, COUNT(*) as count
            FROM result r
            JOIN image i ON r.image_id = i.image_id
            WHERE i.uploaded_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY DATE(i.uploaded_at)
            ORDER BY date
        """)
        processing_trend = [{'date': row[0].strftime('%b %d'), 'count': row[1]} for row in cur.fetchall()]
        
        # Color vs File Type correlation
        cur.execute("""
            SELECT 
                CASE 
                    WHEN Result_data LIKE '%jpg%' OR Result_data LIKE '%jpeg%' THEN 'JPEG'
                    WHEN Result_data LIKE '%png%' THEN 'PNG'
                    WHEN Result_data LIKE '%gif%' THEN 'GIF'
                    WHEN Result_data LIKE '%webp%' THEN 'WebP'
                    ELSE 'Other'
                END as file_type,
                CASE 
                    WHEN Result_data LIKE '%Red%' THEN 'Red'
                    WHEN Result_data LIKE '%Green%' THEN 'Green'
                    WHEN Result_data LIKE '%Blue%' THEN 'Blue'
                    WHEN Result_data LIKE '%Yellow%' THEN 'Yellow'
                    WHEN Result_data LIKE '%Purple%' THEN 'Purple'
                    ELSE 'Other'
                END as color,
                COUNT(*) as count
            FROM result
            GROUP BY file_type, color
            ORDER BY count DESC
            LIMIT 10
        """)
        correlation_data = [{'file_type': row[0], 'color': row[1], 'count': row[2]} for row in cur.fetchall()]
        
        # Success rate calculation
        cur.execute("SELECT COUNT(*) FROM result")
        successful_results = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM image")
        total_images = cur.fetchone()[0]
        
        success_rate = round((successful_results / total_images * 100) if total_images > 0 else 0, 1)
        
        cur.close()
        
        return jsonify({
            'success': True,
            'color_data': color_data,
            'file_type_data': file_type_data,
            'size_data': size_data,
            'processing_trend': processing_trend,
            'correlation_data': correlation_data,
            'success_rate': success_rate,
            'total_processed': successful_results,
            'total_images': total_images
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ======================= AI Prediction Route ======================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and AI prediction"""
    try:
        logger.info("Received AI prediction request")
        
        # Check if AI model is initialized
        if 'inference' not in globals() or inference is None:
            logger.error("AI model not initialized")
            return jsonify({
                'success': False,
                'error': 'AI model not available. Please try again later.'
            }), 500
        
        user_id = session['user_id']
        cur = None  # Initialize cursor
        
        try:
            # No DB storage or upload limit check here!
            if 'image' not in request.files:
                logger.error("No image file in request")
                return jsonify({
                    'success': False,
                    'error': 'No image file provided'
                }), 400
                
            file = request.files['image']
            if file.filename == '':
                logger.error("Empty filename")
                return jsonify({
                    'success': False,
                    'error': 'No selected file'
                }), 400
                
            # Validate file extension
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            if ('.' not in file.filename or 
                file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions):
                logger.error(f"Invalid file extension: {file.filename}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'
                }), 400

            logger.info(f"Processing image: {file.filename}")
            
            # Save the file temporarily for processing
            temp_dir = os.path.join(app.static_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            try:
                # Verify it's a valid image file
                with Image.open(temp_path) as img:
                    img.verify()
                    
                # Process the image with AI model
                logger.info("Sending image to AI model for prediction")
                try:
                    with open(temp_path, 'rb') as f:
                        file_storage = FileStorage(f)
                        result = inference.predict(file_storage)
                    
                    logger.info("AI model prediction completed")
                except Exception as ai_error:
                    logger.error(f"AI model prediction failed: {str(ai_error)}")
                    return jsonify({
                        'success': False,
                        'error': f'AI processing failed: {str(ai_error)}'
                    }), 500
                
                # Ensure all numpy arrays are converted to lists
                if 'objectness_scores' in result and result['objectness_scores'] is not None:
                    result['objectness_scores'] = result['objectness_scores'].tolist()
                    
                return jsonify({
                    'success': True,
                    'data': result
                })
                
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Image processing error: {str(e)}'
                }), 500
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Database error: {str(e)}'
            }), 500
            
        finally:
            if cur:
                cur.close()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500




# =========================== Run App ==============================
if __name__ == '__main__':
    app.run(debug=True, port=5000)

# =========================== Global Error Handler ==============================
@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler to return JSON instead of HTML error pages"""
    import traceback
    
    # Log the error for debugging
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Return JSON error response
    return jsonify({
        'success': False,
        'error': str(e),
        'message': 'An unexpected error occurred. Please try again.',
        'type': type(e).__name__
    }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Route not found',
        'message': 'The requested endpoint does not exist.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An internal server error occurred. Please try again.'
    }), 500
