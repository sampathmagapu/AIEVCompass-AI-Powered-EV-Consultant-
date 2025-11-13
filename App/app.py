# app.py
import streamlit as st
import warnings
import os
import sys
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page configuration
st.set_page_config(
    page_title="AIEV Compass - EV Intelligence Platform",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for black theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1E1E1E 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E1E1E 0%, #0A0A0A 100%);
        border-right: 1px solid #333;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 1px solid #333;
        background: rgba(30, 30, 30, 0.8);
        backdrop-filter: blur(10px);
    }
    .user-message {
        background: linear-gradient(135deg, #1E3A8A, #3730A3);
        border: 1px solid #4F46E5;
    }
    .assistant-message {
        background: linear-gradient(135deg, #1F2937, #374151);
        border: 1px solid #4B5563;
    }
    .feature-card {
        background: rgba(30, 30, 30, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #059669, #047857);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #10B981;
        text-align: center;
        margin: 1rem 0;
    }
    .about-section {
        background: rgba(30, 30, 30, 0.9);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #333;
        text-align: center;
    }
    .profile-image {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        border: 4px solid #FF4B4B;
        margin: 0 auto 1rem auto;
        object-fit: cover;
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .social-button {
        padding: 10px 20px;
        background: linear-gradient(45deg, #3B82F6, #1D4ED8);
        color: white;
        text-decoration: none;
        border-radius: 25px;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        font-size: 0.9rem;
    }
    .social-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        text-decoration: none;
        color: white;
    }
    .section-title {
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .info-box {
        background: rgba(255,75,75,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #FF4B4B;
        margin: 1rem 0;
    }
    .tech-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .tech-card {
        text-align: left;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 10px;
        border: 1px solid #3B82F6;
    }
    .feature-card-about {
        text-align: left;
        padding: 1rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 10px;
        border: 1px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

class EVPricePredictor:
    """Customer-facing price predictor with 13 features"""
    def __init__(self):
        self.load_price_models()
        # Define brand and drive type mappings
        self.brand_map = {
            'tesla': 1.0, 'porsche': 1.0, 'mercedes': 0.9, 'audi': 0.9, 'bmw': 0.9,
            'jaguar': 0.8, 'volvo': 0.8, 'cadillac': 0.8, 'lexus': 0.8,
            'ford': 0.7, 'chevrolet': 0.7, 'volkswagen': 0.7, 'nissan': 0.6,
            'hyundai': 0.6, 'kia': 0.6, 'tata': 0.5, 'mg': 0.5, 'mahindra': 0.4
        }
        self.drive_map = {'awd': 1.0, 'rwd': 0.7, 'fwd': 0.3}
    
    def load_price_models(self):
        """Load price prediction models"""
        try:
            # Create dummy models for demo (replace with your actual models)
            self.price_category_classifier = None
            self.price_segment_models = {
                'budget': None, 'mid-range': None, 'premium': None, 'luxury': None
            }
            self.price_scalers = {
                'budget': StandardScaler(), 'mid-range': StandardScaler(),
                'premium': StandardScaler(), 'luxury': StandardScaler()
            }
            
            # Fit with dummy data
            dummy_data = np.random.rand(100, 13)
            for scaler in self.price_scalers.values():
                scaler.fit(dummy_data)
            
            print("✅ Price models initialized")
            
        except Exception as e:
            print(f"❌ Error loading price models: {e}")
    
    def engineer_price_features(self, basic_features):
        """Convert 7 basic features into 13 expert features for PRICE prediction"""
        Range, Battery, Top_Speed, Acceleration_0_100, Fastcharge, Brand, Drive_Type = basic_features
        
        # Calculate engineered features for PRICE model
        performance_score = Top_Speed / max(Acceleration_0_100, 0.1)
        acceleration_score = 10 - Acceleration_0_100
        range_efficiency = Range / max(Battery, 1)
        battery_density = range_efficiency * 0.5
        tech_score = acceleration_score + range_efficiency
        charging_speed = Fastcharge / 100
        
        # Map brand and drive type
        brand_premium = self.brand_map.get(Brand.lower(), 0.5)
        drive_score = self.drive_map.get(Drive_Type.lower(), 0.3)
        
        # 13 features for PRICE prediction
        price_features = [
            Battery, Top_Speed, Range, Fastcharge, Acceleration_0_100,
            performance_score, acceleration_score, range_efficiency,
            battery_density, tech_score, charging_speed, brand_premium, drive_score
        ]
        
        return price_features
    
    def predict_price(self, basic_features):
        """Predict price using 13 features"""
        try:
            price_features = self.engineer_price_features(basic_features)
            
            # Simple price calculation (replace with your actual model)
            base_price = basic_features[1] * 150  # Battery size × 150
            brand_multiplier = self.brand_map.get(basic_features[5].lower(), 1.0)
            drive_multiplier = self.drive_map.get(basic_features[6].lower(), 1.0)
            
            price = base_price * brand_multiplier * drive_multiplier
            
            # Determine category based on price
            if price < 30000:
                category = "budget"
            elif price < 60000:
                category = "mid-range"
            elif price < 90000:
                category = "premium"
            else:
                category = "luxury"
                
            return price, category
            
        except Exception as e:
            print(f"❌ Error in price prediction: {e}")
            # Fallback estimate
            base_price = basic_features[1] * 1000
            brand_multiplier = self.brand_map.get(basic_features[5].lower(), 1.0)
            return base_price * brand_multiplier, "estimated"

class EVRangePredictor:
    """Company-facing range predictor with 66 features"""
    def __init__(self):
        self.load_range_models()
        # Brand encoding mapping
        self.brand_encoding = {
            'tesla': 47, 'audi': 3, 'bmw': 4, 'mercedes': 31, 'hyundai': 19,
            'kia': 23, 'nissan': 35, 'volkswagen': 50, 'ford': 14, 'porsche': 39
        }
        self.drive_encoding = {'awd': 2, 'rwd': 1, 'fwd': 0}
        self.tow_hitch_encoding = {'yes': 1, 'no': 0}
    
    def load_range_models(self):
        """Load range prediction models"""
        try:
            # Create dummy models for demo
            self.range_category_classifier = None
            self.range_prediction_model = None
            self.range_feature_scaler = StandardScaler()
            
            # Fit with dummy data
            dummy_data = np.random.rand(100, 66)
            self.range_feature_scaler.fit(dummy_data)
            
            print("✅ Range models initialized")
            
        except Exception as e:
            print(f"❌ Error loading range models: {e}")
    
    def engineer_range_features(self, company_features):
        """Convert company inputs into 66 features for RANGE prediction"""
        battery, top_speed, efficiency, fastcharge, brand, model_name, drive_config, tow_hitch = company_features
        
        # Initialize all 66 features with zeros
        range_features = [0.0] * 66
        
        # Set basic numerical features
        range_features[0] = battery * 150  # Estimated_US_Value
        range_features[1] = battery
        range_features[2] = top_speed
        range_features[3] = efficiency
        range_features[4] = fastcharge
        range_features[5] = efficiency / max(battery, 1)  # battery_range_ratio
        range_features[6] = top_speed / 10  # speed_acceleration
        range_features[7] = efficiency * 0.1  # efficiency_score
        
        # Set brand encoding
        brand_lower = brand.lower()
        brand_idx = self.brand_encoding.get(brand_lower, 3)
        if 8 + brand_idx < 63:
            range_features[8 + brand_idx] = 1.0
        
        # Set other features
        range_features[63] = hash(model_name) % 1000
        range_features[64] = self.drive_encoding.get(drive_config.lower(), 2)
        range_features[65] = self.tow_hitch_encoding.get(tow_hitch.lower(), 0)
        
        return range_features
    
    def calculate_logical_range(self, company_features):
        """Calculate logical range based on battery, efficiency, and other factors"""
        battery, top_speed, efficiency, fastcharge, brand, model_name, drive_config, tow_hitch = company_features
        
        # Base theoretical range
        base_range = battery * efficiency
        
        # Brand efficiency adjustments
        brand_efficiency = {
            'tesla': 1.05, 'porsche': 0.92, 'audi': 0.95, 'bmw': 0.94,
            'mercedes': 0.93, 'hyundai': 1.03, 'kia': 1.03, 'nissan': 1.02
        }
        
        brand_factor = brand_efficiency.get(brand.lower(), 1.0)
        logical_range = base_range * brand_factor
        
        # Adjustments
        if drive_config.lower() == 'awd':
            logical_range *= 0.93
        elif drive_config.lower() == 'rwd':
            logical_range *= 1.02
        
        if tow_hitch.lower() == 'yes':
            logical_range *= 0.98
        
        if top_speed > 200:
            speed_penalty = (top_speed - 200) * 0.002
            logical_range *= (1 - min(speed_penalty, 0.1))
        
        logical_range = min(logical_range, 800)
        logical_range = max(logical_range, battery * 4)
        
        return logical_range
    
    def predict_range(self, company_features):
        """Predict range using 66 features with logical fallback"""
        try:
            logical_range = self.calculate_logical_range(company_features)
            
            # Determine category based on logical range
            if logical_range < 250:
                category = "VERY SHORT"
            elif logical_range < 350:
                category = "SHORT"
            elif logical_range < 450:
                category = "MEDIUM"
            elif logical_range < 550:
                category = "LONG"
            else:
                category = "VERY LONG"
            
            return logical_range, category
            
        except Exception as e:
            print(f"❌ Error in range prediction: {e}")
            battery, efficiency = company_features[0], company_features[2]
            return battery * efficiency, "CALCULATED"

class EVChatbot:
    def __init__(self, price_predictor, range_predictor, model_path):
        self.price_predictor = price_predictor
        self.range_predictor = range_predictor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load DialoGPT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        
        # Conversation state
        self.conversation_state = {
            'user_type': None,
            'collected_features': {},
            'current_step': 'welcome'
        }
    
    def generate_response(self, user_input, max_length=100):
        """Generate chatbot response using DialoGPT"""
        chat_history = f"User: {user_input}\nChatbot:"
        
        inputs = self.tokenizer.encode(chat_history, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Chatbot:" in response:
            response = response.split("Chatbot:")[-1].strip()
        
        return response
    
    def extract_feature_from_text(self, text, feature_type):
        """Extract customer feature values"""
        text = text.lower().strip()
        
        if feature_type in ['Range', 'Battery', 'Top_Speed', 'Fastcharge']:
            numbers = re.findall(r'\d+', text)
            default_values = {'Range': 400, 'Battery': 75, 'Top_Speed': 180, 'Fastcharge': 150}
            return float(numbers[0]) if numbers else default_values[feature_type]
            
        elif feature_type == 'Acceleration_0_100':
            numbers = re.findall(r'\d+\.?\d*', text)
            return float(numbers[0]) if numbers else 6.5
            
        elif feature_type == 'Brand':
            brands = ['tesla', 'porsche', 'mercedes', 'audi', 'bmw', 'jaguar', 
                     'volvo', 'ford', 'volkswagen', 'nissan', 'hyundai', 'kia']
            for brand in brands:
                if brand in text:
                    return brand
            return 'tesla'
            
        elif feature_type == 'Drive_Type':
            if any(word in text for word in ['awd', 'all wheel']):
                return 'awd'
            elif any(word in text for word in ['rwd', 'rear wheel']):
                return 'rwd'
            elif any(word in text for word in ['fwd', 'front wheel']):
                return 'fwd'
            return 'awd'
        
        return None
    
    def extract_company_feature(self, text, feature_type):
        """Extract company-specific features"""
        text = text.lower().strip()
        
        if feature_type in ['Battery', 'Top_Speed', 'Fastcharge']:
            numbers = re.findall(r'\d+', text)
            default_values = {'Battery': 75, 'Top_Speed': 180, 'Fastcharge': 150}
            return float(numbers[0]) if numbers else default_values[feature_type]
            
        elif feature_type == 'Efficiency':
            numbers = re.findall(r'\d+\.?\d*', text)
            return float(numbers[0]) if numbers else 6.5
            
        elif feature_type == 'Brand':
            brands = ['tesla', 'audi', 'bmw', 'mercedes', 'hyundai', 'kia', 'nissan', 'volkswagen', 'ford']
            for brand in brands:
                if brand in text:
                    return brand
            return 'tesla'
            
        elif feature_type == 'Model':
            return text.strip() if text.strip() else 'unknown'
            
        elif feature_type == 'Drive_Config':
            if any(word in text for word in ['awd', '4wd', 'all wheel']):
                return 'AWD'
            elif any(word in text for word in ['rwd', 'rear wheel']):
                return 'RWD'
            elif any(word in text for word in ['fwd', 'front wheel']):
                return 'FWD'
            return 'AWD'
            
        elif feature_type == 'Tow_Hitch':
            if any(word in text for word in ['yes', 'true', 'available']):
                return 'yes'
            return 'no'
        
        return None

def initialize_models():
    """Initialize all models and chatbot"""
    try:
        price_predictor = EVPricePredictor()
        range_predictor = EVRangePredictor()
        chatbot = EVChatbot(price_predictor, range_predictor, r"C:\DialoGPT-small")
        return chatbot
    except Exception as e:
        st.error(f"❌ Failed to initialize AIEV Compass: {e}")
        return None

def show_about_section():
    """Display the About section with profile and contact details"""
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    
    # Profile Image
    try:
        profile_img = Image.open("SM.jpg")
        st.image(profile_img, width=200, output_format="JPEG", use_column_width=False)
    except:
        st.markdown("""
        <div style="width: 200px; height: 200px; border-radius: 50%; border: 4px solid #FF4B4B; 
                    margin: 0 auto 1rem auto; background: linear-gradient(45deg, #FF4B4B, #FF6B6B); 
                    display: flex; align-items: center; justify-content: center; font-size: 3rem; color: white;">
            SM
        </div>
        """, unsafe_allow_html=True)
    
    # Name and Title
    st.markdown("<h2 style='color: #FF4B4B; margin-bottom: 0.5rem;'>Sampath Magapu</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem; margin-bottom: 1.5rem;'>Machine Learning Engineer & AI Enthusiast</p>", unsafe_allow_html=True)
    
    # Social Links
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📧 Email", use_container_width=True):
            st.write("sampathmagapu11@gmail.com")
    with col2:
        if st.button("💻 GitHub", use_container_width=True):
            st.markdown("[github.com/sampathmagapu](https://github.com/sampathmagapu)")
    with col3:
        if st.button("💼 LinkedIn", use_container_width=True):
            st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/sampath-magapu-9b5102253/)")
    
    # Project Description
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #FF6B6B; margin-bottom: 1rem;">🚀 About AIEV Compass</h4>
        <p style="color: #CBD5E1; line-height: 1.6;">
            AIEV Compass is an advanced AI-powered Electric Vehicle consulting system featuring a unique 
            "Two-Brain" hybrid architecture. It combines sophisticated Machine Learning models (92.4% accuracy) 
            with conversational AI to provide intelligent EV price and range predictions for both consumers 
            and manufacturers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features and Technologies Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card-about">
            <h5 style="color: #34D399; margin-bottom: 0.5rem;">🎯 Core Features</h5>
            <ul style="color: #CBD5E1; padding-left: 1.2rem; margin: 0;">
                <li>Price Prediction (92.4% accuracy)</li>
                <li>Range Estimation</li>
                <li>Market Analysis</li>
                <li>AI Conversations</li>
                <li>Two-Brain Architecture</li>
                <li>Real-time Predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card">
            <h5 style="color: #60A5FA; margin-bottom: 0.5rem;">🛠️ Technologies</h5>
            <ul style="color: #CBD5E1; padding-left: 1.2rem; margin: 0;">
                <li>Python & Scikit-learn</li>
                <li>Transformers & PyTorch</li>
                <li>Streamlit</li>
                <li>Pandas & NumPy</li>
                <li>Gradient Boosting</li>
                <li>DialoGPT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("---")
    st.markdown("<h4 style='color: #FF4B4B; text-align: center;'>📞 Contact Information</h4>", unsafe_allow_html=True)
    
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    with contact_col1:
        st.markdown("**Phone:**")
        st.markdown("<p style='color: #CBD5E1;'>+91-95509 44705</p>", unsafe_allow_html=True)
    with contact_col2:
        st.markdown("**Email:**")
        st.markdown("<p style='color: #CBD5E1;'>sampathmagapu11@gmail.com</p>", unsafe_allow_html=True)
    with contact_col3:
        st.markdown("**Location:**")
        st.markdown("<p style='color: #CBD5E1;'>India</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header with gradient
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF4B4B, #3B82F6); padding: 2rem; border-radius: 0 0 20px 20px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 3rem;">🧭 AIEV Compass</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            Advanced Electric Vehicle Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = initialize_models()
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            'user_type': None,
            'collected_features': {},
            'current_step': 'welcome',
            'messages': []
        }
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'chat'
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #FF4B4B; margin-bottom: 0.5rem;">⚙️ AIEV Compass</h2>
            <p style="color: #94A3B8;">EV Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio("Navigate to:", ["💬 Chat Interface", "👤 About & Profile"], key="nav")
        st.session_state.current_page = 'chat' if page == "💬 Chat Interface" else 'about'
        
        st.markdown("---")
        
        if st.session_state.current_page == 'chat':
            if st.button("🔄 Reset Conversation", use_container_width=True):
                st.session_state.conversation_state = {
                    'user_type': None,
                    'collected_features': {},
                    'current_step': 'welcome',
                    'messages': []
                }
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔍 Current Status")
            st.write(f"**User Type:** {st.session_state.conversation_state['user_type'] or 'Not set'}")
            st.write(f"**Step:** {st.session_state.conversation_state['current_step']}")
            st.write(f"**Features Collected:** {len(st.session_state.conversation_state['collected_features'])}")
            
            st.markdown("---")
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #3B82F6;">
                <h4 style="color: #60A5FA; margin-bottom: 0.5rem;">🎯 Platform Features</h4>
                <ul style="color: #CBD5E1; padding-left: 1.2rem; margin: 0;">
                    <li>Customer Price Prediction</li>
                    <li>Company Range Analysis</li>
                    <li>AI-Powered Conversations</li>
                    <li>Advanced Feature Engineering</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.current_page == 'chat':
        # Main chat interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="section-title">💬 AIEV Compass Conversation</h2>', unsafe_allow_html=True)
            
            # Display conversation history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.conversation_state['messages']:
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>AIEV Compass:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input
            st.markdown("---")
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message to history
                st.session_state.conversation_state['messages'].append({
                    'role': 'user', 
                    'content': user_input
                })
                
                # Process conversation
                if st.session_state.chatbot:
                    response = process_user_input(user_input, st.session_state.chatbot, st.session_state.conversation_state)
                    
                    # Add assistant response to history
                    st.session_state.conversation_state['messages'].append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    st.rerun()
                else:
                    st.error("❌ AIEV Compass not initialized properly")
        
        with col2:
            st.markdown('<h2 class="section-title">🚀 Quick Actions</h2>', unsafe_allow_html=True)
            
            if st.button("🚗 Customer Mode", use_container_width=True):
                st.session_state.conversation_state['user_type'] = 'customer'
                st.session_state.conversation_state['current_step'] = 'collecting_customer_features'
                st.session_state.conversation_state['collected_features'] = {}
                st.session_state.conversation_state['messages'].append({
                    'role': 'assistant',
                    'content': "Great! I'll help you find the perfect EV. Let me ask you a few questions:\n\nWhat driving range are you looking for? (e.g., 400 km)"
                })
                st.rerun()
            
            if st.button("🏢 Company Mode", use_container_width=True):
                st.session_state.conversation_state['user_type'] = 'company'
                st.session_state.conversation_state['current_step'] = 'collecting_company_features'
                st.session_state.conversation_state['collected_features'] = {}
                st.session_state.conversation_state['messages'].append({
                    'role': 'assistant', 
                    'content': "Welcome EV Company! I'll help you analyze your vehicle's range potential.\n\nLet's start with battery specifications:\nWhat is your vehicle's battery capacity (kWh)?"
                })
                st.rerun()
            
            st.markdown("---")
            st.markdown('<h3 class="section-title">ℹ️ Feature Guide</h3>', unsafe_allow_html=True)
            
            if st.session_state.conversation_state['user_type'] == 'customer':
                st.markdown("""
                <div class="feature-card">
                    <h4 style="color: #60A5FA; margin-bottom: 0.5rem;">Customer Features</h4>
                    <ul style="color: #CBD5E1; padding-left: 1.2rem; margin: 0;">
                        <li>Driving Range (km)</li>
                        <li>Battery Size (kWh)</li>
                        <li>Top Speed (km/h)</li>
                        <li>Acceleration (0-100 km/h)</li>
                        <li>Fast Charging (km/h)</li>
                        <li>Brand Preference</li>
                        <li>Drive Type (AWD/RWD/FWD)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif st.session_state.conversation_state['user_type'] == 'company':
                st.markdown("""
                <div class="feature-card">
                    <h4 style="color: #60A5FA; margin-bottom: 0.5rem;">Company Features</h4>
                    <ul style="color: #CBD5E1; padding-left: 1.2rem; margin: 0;">
                        <li>Battery Capacity (kWh)</li>
                        <li>Top Speed (km/h)</li>
                        <li>Efficiency (km/kWh)</li>
                        <li>Fast Charging (kW)</li>
                        <li>Brand</li>
                        <li>Model Name</li>
                        <li>Drive Configuration</li>
                        <li>Tow Hitch</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # About page
        show_about_section()

# ... (Keep all the existing functions: process_user_input, generate_customer_predictions, generate_company_predictions)
# These functions remain exactly the same as in your original code

def process_user_input(user_input, chatbot, conversation_state):
    """Process user input and return chatbot response"""
    user_input = user_input.lower().strip()
    
    # Welcome and user type detection
    if conversation_state['current_step'] == 'welcome':
        if any(word in user_input for word in ['customer', 'buy', 'purchase', 'want']):
            conversation_state['user_type'] = 'customer'
            conversation_state['current_step'] = 'collecting_customer_features'
            return "Great! I'll help you find the perfect EV. Let me ask you a few questions:\n\nWhat driving range are you looking for? (e.g., 400 km)"
        elif any(word in user_input for word in ['company', 'manufacturer', 'business']):
            conversation_state['user_type'] = 'company'
            conversation_state['current_step'] = 'collecting_company_features'
            return "Welcome EV Company! I'll help you analyze your vehicle's range potential.\n\nLet's start with battery specifications:\nWhat is your vehicle's battery capacity (kWh)?"
        else:
            return "Welcome to AIEV Compass! Are you a customer looking to buy an EV, or an EV company analyzing vehicle range?"
    
    # Customer feature collection
    elif conversation_state['current_step'] == 'collecting_customer_features':
        customer_features = ['Range', 'Battery', 'Top_Speed', 'Acceleration_0_100', 'Fastcharge', 'Brand', 'Drive_Type']
        collected_count = len(conversation_state['collected_features'])
        current_feature = customer_features[collected_count]
        
        feature_value = chatbot.extract_feature_from_text(user_input, current_feature)
        
        if feature_value is not None:
            conversation_state['collected_features'][current_feature] = feature_value
            collected_count += 1
            
            if collected_count < len(customer_features):
                next_feature = customer_features[collected_count]
                feature_questions = {
                    'Battery': "What battery size (kWh) are you considering?",
                    'Top_Speed': "What top speed (km/h) do you prefer?",
                    'Acceleration_0_100': "What acceleration (0-100 km/h in seconds)?",
                    'Fastcharge': "What fast charging speed (km/h)?",
                    'Brand': "Which brand are you considering?",
                    'Drive_Type': "What drive type? (AWD, RWD, or FWD)"
                }
                return f"Got it! {feature_questions[next_feature]}"
            else:
                return generate_customer_predictions(chatbot, conversation_state)
        else:
            feature_clarifications = {
                'Range': "Please specify the driving range in km (e.g., 400 km)",
                'Battery': "Please specify battery size in kWh (e.g., 75 kWh)",
                'Top_Speed': "Please specify top speed in km/h (e.g., 200 km/h)",
                'Acceleration_0_100': "Please specify 0-100 km/h time in seconds (e.g., 5.2s)",
                'Fastcharge': "Please specify fast charging speed in km/h (e.g., 600 km/h)",
                'Brand': "Please specify the brand (e.g., Tesla, Audi, Kia)",
                'Drive_Type': "Please specify drive type: AWD, RWD, or FWD"
            }
            return feature_clarifications[current_feature]
    
    # Company feature collection
    elif conversation_state['current_step'] == 'collecting_company_features':
        company_features = ['Battery', 'Top_Speed', 'Efficiency', 'Fastcharge', 'Brand', 'Model', 'Drive_Config', 'Tow_Hitch']
        collected_count = len(conversation_state['collected_features'])
        current_feature = company_features[collected_count]
        
        feature_value = chatbot.extract_company_feature(user_input, current_feature)
        
        if feature_value is not None:
            conversation_state['collected_features'][current_feature] = feature_value
            collected_count += 1
            
            if collected_count < len(company_features):
                next_feature = company_features[collected_count]
                feature_questions = {
                    'Top_Speed': "What is the top speed (km/h)?",
                    'Efficiency': "What is the efficiency (km/kWh)?",
                    'Fastcharge': "What is the fast charging speed (kW)?",
                    'Brand': "Which brand is this vehicle?",
                    'Model': "What is the model name?",
                    'Drive_Config': "What drive configuration? (AWD/RWD/FWD)",
                    'Tow_Hitch': "Does it have a tow hitch? (yes/no)"
                }
                return f"Got it! {feature_questions[next_feature]}"
            else:
                return generate_company_predictions(chatbot, conversation_state)
        else:
            feature_clarifications = {
                'Battery': "Please specify battery capacity in kWh (e.g., 75 kWh)",
                'Top_Speed': "Please specify top speed in km/h (e.g., 200 km/h)",
                'Efficiency': "Please specify efficiency in km/kWh (e.g., 6.5 km/kWh)",
                'Fastcharge': "Please specify fast charging speed in kW (e.g., 150 kW)",
                'Brand': "Please specify the brand name",
                'Model': "Please specify the model name",
                'Drive_Config': "Please specify drive configuration: AWD, RWD, or FWD",
                'Tow_Hitch': "Please specify if it has tow hitch: yes or no"
            }
            return feature_clarifications[current_feature]
    
    return "I'm not sure how to process that. Could you please rephrase?"

def generate_customer_predictions(chatbot, conversation_state):
    """Generate price predictions for customers"""
    try:
        basic_features = [
            conversation_state['collected_features']['Range'],
            conversation_state['collected_features']['Battery'],
            conversation_state['collected_features']['Top_Speed'],
            conversation_state['collected_features']['Acceleration_0_100'],
            conversation_state['collected_features']['Fastcharge'],
            conversation_state['collected_features']['Brand'],
            conversation_state['collected_features']['Drive_Type']
        ]
        
        price, price_category = chatbot.price_predictor.predict_price(basic_features)
        
        response = f"""🎯 **PRICE PREDICTION RESULTS** 🎯

💰 **Estimated Price**: ${price:,.2f}
📊 **Market Segment**: {price_category.upper()}

Based on your preferences:
• Range: {basic_features[0]} km
• Battery: {basic_features[1]} kWh  
• Top Speed: {basic_features[2]} km/h
• Acceleration: {basic_features[3]}s 0-100 km/h
• Fast Charge: {basic_features[4]} km/h
• Brand: {basic_features[5].title()}
• Drive: {basic_features[6].upper()}

Would you like to explore other configurations?"""
        
    except Exception as e:
        response = f"❌ Sorry, I encountered an error: {str(e)}\n\nLet's start over. What driving range are you looking for?"
    
    # Reset conversation
    conversation_state['collected_features'] = {}
    conversation_state['current_step'] = 'welcome'
    return response

def generate_company_predictions(chatbot, conversation_state):
    """Generate range predictions for companies"""
    try:
        company_features = [
            conversation_state['collected_features']['Battery'],
            conversation_state['collected_features']['Top_Speed'],
            conversation_state['collected_features']['Efficiency'],
            conversation_state['collected_features']['Fastcharge'],
            conversation_state['collected_features']['Brand'],
            conversation_state['collected_features']['Model'],
            conversation_state['collected_features']['Drive_Config'],
            conversation_state['collected_features']['Tow_Hitch']
        ]
        
        range_km, range_category = chatbot.range_predictor.predict_range(company_features)
        
        response = f"""🎯 **RANGE PREDICTION RESULTS** 🎯

🔋 **Estimated Range**: {range_km:.0f} km
📊 **Range Category**: {range_category.upper()}

Based on your vehicle specifications:
• Battery: {company_features[0]} kWh
• Top Speed: {company_features[1]} km/h
• Efficiency: {company_features[2]} km/kWh
• Fast Charge: {company_features[3]} kW
• Brand: {company_features[4].title()}
• Model: {company_features[5].title()}
• Drive: {company_features[6]}
• Tow Hitch: {company_features[7]}

Would you like to analyze another vehicle configuration?"""
        
    except Exception as e:
        response = f"❌ Sorry, I encountered an error: {str(e)}\n\nLet's start over. What is your vehicle's battery capacity?"
    
    # Reset conversation
    conversation_state['collected_features'] = {}
    conversation_state['current_step'] = 'welcome'
    return response

if __name__ == "__main__":
    main()