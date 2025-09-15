from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import os
import functools
from flask_cors import CORS
from datetime import datetime, timedelta
import stripe
import random
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
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

bcrypt = Bcrypt(app)

# === Stripe Test Secret Key ===
stripe.api_key = 'sk_test_51RVq55PCPwXipCduM0js8eKiZLYupEIDG4Nvypss2WnFSSKw5Tu4pbzLpTQwxy0PMgI7uZRpnIFZ2AcNsUNCGrcZ00GFuV2dKc'

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
    # Decorator does nothing, allows all access
    return f

# ======================= Index Route ============================
@app.route('/')
def index():
    # Show the main website page directly
    return render_template('index.html')

# ======================== Dashboard =============================
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

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
        input_password = data.get('password')  # plain password from user

        cur = mysql.connection.cursor()
        cur.execute('SELECT User_ID, Name, Password FROM user_management WHERE Email = %s', (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user['Password'], input_password):
            session['user_id'] = user['User_ID']
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
    

# ========================= User Signup ===========================
@app.route('/user_signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        user_id = data.get('id')
        name = data.get('name')
        email = data.get('email')
        password = generate_password_hash(data.get('password'))
        phone = data.get('phone')

        cur = mysql.connection.cursor()
        cur.execute('SELECT User_ID FROM user_management WHERE Email = %s', (email,))
        if cur.fetchone():
            return jsonify({
                'success': False,
                'message': 'Email already registered!',
                'field': 'email'
            }), 400

        cur.execute('INSERT INTO user_management (User_ID, Name, Email, Password, Phone) VALUES (%s, %s, %s, %s, %s)',
                    (user_id, name, email, password, phone))
        mysql.connection.commit()
        cur.close()

        return jsonify({
            'success': True,
            'message': 'User registration successful!',
            'redirect': '/dashboard'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
        }), 500




# ========================= Admin Login ===========================
@app.route('/admin_login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        cur = mysql.connection.cursor()
        cur.execute('SELECT Admin_ID, Name, Password FROM admin WHERE Email = %s', (email,))
        admin = cur.fetchone()
        cur.close()

        if admin and check_password_hash(admin['Password'], password):
            session['admin_id'] = admin['Admin_ID']
            return jsonify({
                'success': True,
                'message': 'Admin login successful!',
                'redirect': '/view'
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


# ========================= Admin Signup ===========================
@app.route('/admin_signup', methods=['POST'])
def admin_signup():
    try:
        data = request.get_json()
        admin_id = data.get('id')
        name = data.get('name')
        email = data.get('email')
        password = generate_password_hash(data.get('password'))
        salary = data.get('salary')

        cur = mysql.connection.cursor()
        cur.execute('SELECT Admin_ID FROM admin WHERE Email = %s', (email,))
        if cur.fetchone():
            return jsonify({
                'success': False,
                'message': 'Email already registered!',
                'field': 'email'
            }), 400

        cur.execute('INSERT INTO admin (Admin_ID, Name, Email, Password, Salary) VALUES (%s, %s, %s, %s, %s)',
                    (admin_id, name, email, password, salary))
        mysql.connection.commit()

        session['admin_id'] = admin_id
        cur.close()

        return jsonify({
            'success': True,
            'message': 'Admin registration successful!',
            'redirect': '/view'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
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
        
        # Get table data
        cur.execute(f"SELECT * FROM {actual_table_name}")
        data = cur.fetchall()
        
        # Convert to list of dictionaries
        result = []
        for row in data:
            result.append(dict(zip(columns, row)))
        
        cur.close()
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'Error fetching {table_name} data'
        }), 500

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
    







# ====================== View Route with Data ======================
@app.route('/view')
@login_required
def view():
    try:
        # Get admin username for display
        if 'admin_id' in session:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Name FROM admin WHERE Admin_ID = %s", (session['admin_id'],))
            admin = cur.fetchone()
            admin_name = admin[0] if admin else "Admin"
            cur.close()
        else:
            admin_name = "Admin"
            
        return render_template('View.html', admin_username=admin_name)
                            
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('View.html', admin_username="Admin")

# ======================= Feedback Submission ======================
@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'message': 'Please login first'
            }), 401

        # Get form data
        fid = random.randint(100, 200)
        rating = request.form.get('rating')
        feedback_type = request.form.get('Feedback_Type')
        feedback_text = request.form.get('Feedback_Text')
        user_id = session['user_id']

        # Insert into database
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO feedback (Feedback_ID, Rating, Feedback_Type, Feedback_Text, User_ID)
            VALUES (%s, %s, %s, %s, %s)
        """, (fid, rating, feedback_type, feedback_text, user_id))
        mysql.connection.commit()
        cur.close()

        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully!'
        })

    except Exception as e:
        mysql.connection.rollback()
        return jsonify({
            'success': False,
            'message': f'Error submitting feedback: {str(e)}'
        }), 500

# ======================= View Feedback ===========================
@app.route('/get_feedback', methods=['GET'])
@login_required
def get_feedback():
    try:
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'message': 'Please login first'
            }), 401

        user_id = session['user_id']
        
        # Get user's feedback from database
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT Feedback_ID, Rating, Feedback_Type, Feedback_Text, User_ID
            FROM feedback 
            WHERE User_ID = %s
            ORDER BY Feedback_ID DESC
        """, (user_id,))
        
        # Get column names from cursor description
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
            'error': str(e)  # Include actual error in response
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

# ======================= Save Image Info ===========================
@app.route('/save_image_info', methods=['POST'])
@login_required
def save_image_info():
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'User not logged in'}), 401

        user_id = session['user_id']
        cur = mysql.connection.cursor()

        # Check current upload count for this user
        cur.execute("SELECT COUNT(*) FROM image WHERE user_id = %s", (user_id,))
        upload_count = cur.fetchone()[0]
        
        # Check if user has active subscription
        cur.execute("""
            SELECT * FROM subscription 
            WHERE User_ID = %s AND End_Date >= CURDATE() AND Status = 'active'
            LIMIT 1
        """, (user_id,))
        has_active_subscription = cur.fetchone() is not None
        
        # Enforce free upload limit (3 images) if no active subscription
        if not has_active_subscription and upload_count >= 3:
            cur.close()
            return jsonify({
                'success': False,
                'message': 'You have reached the maximum free upload limit (3 images). Please subscribe to continue.',
                'redirect': '/#payment',
                'limit_reached': True
            }), 403

        # File validation
        if 'image' not in request.files:
            cur.close()
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            cur.close()
            return jsonify({'success': False, 'message': 'Empty file name'}), 400

        # Secure file handling
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

        mysql.connection.commit()
        cur.close()

        # Calculate remaining uploads
        remaining_uploads = max(0, 3 - (upload_count + 1))

        return jsonify({
            'success': True,
            'message': 'Image processed successfully!',
            'image_id': image_id,
            'file_url': f"/static/uploads/{filename}",
            'result_data': result_data,
            'remaining_uploads': remaining_uploads
        })

    except Exception as e:
        if 'cur' in locals():
            cur.close()
        return jsonify({'success': False, 'message': str(e)}), 500                

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
            'enterprise': {'amount': 9999, 'upload_limit': -1, 'duration_days': 30}
        }

        subscription_id = random.randint(1, 100)
        start_date = datetime.now()
        end_date = start_date + timedelta(days=plans[plan_type]['duration_days'])
        payment_method = "Stripe"

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO subscription (
                Subscription_ID, Plan_Type, Start_Date, End_Date, Status, 
                Upload_Limit, Uploads_Used, Amount_Paid, Payment_Method, User_ID
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            subscription_id, plan_type, start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'), 'active',
            plans[plan_type]['upload_limit'], 0,
            plans[plan_type]['amount'] / 100,
            payment_method, user_id
        ))
        mysql.connection.commit()
        cur.close()

        if 'selected_plan' in session:
            del session['selected_plan']

        return render_template('pay.html',
            subscription_id=subscription_id,
            plan_type=plan_type.capitalize(),
            payment_method=payment_method,
            amount_paid=f"{plans[plan_type]['amount'] / 100:.2f}",
            upload_limit="Unlimited" if plans[plan_type]['upload_limit'] == -1 else plans[plan_type]['upload_limit'],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

    except Exception as e:
        mysql.connection.rollback()
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

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    try:
        logger.info("Received image upload request")
        
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            return jsonify({'error': 'File must be an image'}), 400

        logger.info(f"Processing image: {file.filename}")
        
        # Save the file stream position
        file.seek(0)
        
        # Process the image
        logger.info("Sending image to model for prediction")
        result = inference.predict(file)
        logger.info("Model prediction completed")
        
        # Convert numpy arrays to lists for JSON serialization
        if result.get('objectness_scores') is not None:
            result['objectness_scores'] = result['objectness_scores'].tolist()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# =========================== Run App ==============================
if __name__ == '__main__':
    app.run(debug=True, port=5001)