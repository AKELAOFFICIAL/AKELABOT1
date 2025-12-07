import telebot
from telebot import types
import time
import threading
import requests
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# --- GLOBAL STATE ---
pending_requests = {} 
user_sessions = {}  # Track user sessions
prediction_queue = {}  # Auto-prediction queue
win_loss_history = {}  # Win/loss history per user
user_wins = {}  # Track user wins count

# --- STICKER IDs ---
WIN_STICKER = "CAACAgUAAxkBAAEP4WRpJ7Hkns8R4bVdlNB6lqHx4Y-yfQAC7BUAAvJ7WVT9Plmc3olelDYE"
LOSS_STICKER = "CAACAgUAAxkBAAEP4WhpJ7IYVnpkGjkymzF0f7E60b4lxQACkRUAAmWW4FShKckc3KGLKTYE"

# --- BOT CONFIGURATION ---
BOT_TOKEN = '8569876877:AAG7xkyPtXRgRsIcO-ZjViNNB9Brg2cJX2M' 

# Channel IDs और Invite Link
CHANNEL_INFO = {
    'channel1': {
        'id': -1002782160527, 
        'link': 'https://t.me/+Db9BlHtMooIyZGM1' 
    },
    'channel2': {
        'id': -1003157366871, 
        'link': 'https://t.me/+34UOAHAECedmZGNl' 
    }
}

# Valid Keys (Case Sensitive)
VALID_KEYS = [
    'ABCDEFGH',
    'abcdefgh',
    'akelatrader87',
    'AKELAOFFICIAL',
    'SHADOW998',
    'ANKIT667'
]

bot = telebot.TeleBot(BOT_TOKEN)

# --- AI LIBRARY IMPORT ---
AI_AVAILABLE = False
TENSORFLOW_AVAILABLE = False
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

AI_AVAILABLE = TENSORFLOW_AVAILABLE or SKLEARN_AVAILABLE or XGBOOST_AVAILABLE

# --- WINGO PREDICTION SYSTEM ---
class AdvancedWinGoPredictor:
    """ADVANCED AI PREDICTION SYSTEM"""
    
    def __init__(self):
        self.api_url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
        self.data_file = "wingo_ai_data.json"
        self.history_file = "ai_prediction_history.json"
        self.win_loss_file = "win_loss_tracking.json"
        self.ai_threshold = 30
        self.sequence_length = 15
        self.num_classes = 10
        
        self.data = []
        self.history = []
        self.win_loss_data = []
        self.models_trained = False
        self.total_records = 0
        self.current_period = "00000"
        self.next_period = "00001"
        self.active_models = []
        
        # Initialize all AI models
        self.initialize_all_models()
        
        # Load existing data
        self.load_data()
        
        # Calculate current and next period
        self.calculate_periods()
        
        # Train models if enough data
        if len(self.data) >= self.ai_threshold and AI_AVAILABLE:
            self.train_all_models()
        
        # Start background data collection
        threading.Thread(target=self.background_data_collector, daemon=True).start()
        
        # Start auto-prediction system
        threading.Thread(target=self.auto_prediction_system, daemon=True).start()
    
    def calculate_periods(self):
        """Calculate current and next period numbers"""
        try:
            if self.data:
                latest_record = self.data[0]
                current_period = latest_record.get('issueNumber', '00000')
                try:
                    period_num = int(current_period)
                    next_period_num = period_num + 1
                    format_length = len(current_period)
                    self.current_period = current_period
                    self.next_period = str(next_period_num).zfill(format_length)
                except:
                    self.current_period = "00000"
                    self.next_period = "00001"
            else:
                self.current_period = "00000"
                self.next_period = "00001"
        except Exception as e:
            self.current_period = "00000"
            self.next_period = "00001"
    
    def initialize_all_models(self):
        """Initialize ALL AI models"""
        try:
            self.active_models = []
            
            # 🔥 DEEP LEARNING MODELS
            if TENSORFLOW_AVAILABLE:
                # LSTM Model
                self.lstm_model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 8), dropout=0.2),
                    BatchNormalization(),
                    LSTM(64, return_sequences=False, dropout=0.2),
                    BatchNormalization(),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(self.num_classes, activation='softmax')
                ])
                self.lstm_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.active_models.append("LSTM")
                
                # CNN Model
                self.cnn_model = Sequential([
                    Conv1D(64, 3, activation='relu', input_shape=(self.sequence_length, 8)),
                    MaxPooling1D(2),
                    Conv1D(128, 3, activation='relu'),
                    MaxPooling1D(2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(self.num_classes, activation='softmax')
                ])
                self.cnn_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.active_models.append("CNN")
            
            # 🔥 MACHINE LEARNING MODELS
            if SKLEARN_AVAILABLE:
                self.rf_model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10,
                    n_jobs=-1
                )
                self.active_models.append("Random Forest")
                
                self.mlp_model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    random_state=42,
                    max_iter=1000,
                    solver='adam'
                )
                self.active_models.append("MLP")
                
                self.gb_model = GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=5
                )
                self.active_models.append("Gradient Boosting")
                
                self.svm_model = SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
                self.active_models.append("SVM")
                
                self.lr_model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    n_jobs=-1
                )
                self.active_models.append("Logistic Regression")
                
                # Voting Ensemble
                estimators = []
                if self.rf_model: estimators.append(('rf', self.rf_model))
                if self.mlp_model: estimators.append(('mlp', self.mlp_model))
                if self.gb_model: estimators.append(('gb', self.gb_model))
                if self.svm_model: estimators.append(('svm', self.svm_model))
                
                if len(estimators) >= 2:
                    self.voting_model = VotingClassifier(
                        estimators=estimators,
                        voting='soft',
                        n_jobs=-1
                    )
                    self.active_models.append("Voting Ensemble")
                else:
                    self.voting_model = None
            
            # 🔥 XGBOOST MODEL
            if XGBOOST_AVAILABLE:
                self.xgb_model = XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1
                )
                self.active_models.append("XGBoost")
            
        except Exception as e:
            print(f"Model initialization failed: {e}")
    
    def load_data(self):
        """Load historical data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    self.data = data_dict.get('list', [])
                    self.total_records = len(self.data)
                    
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f).get('predictions', [])
                    
            if os.path.exists(self.win_loss_file):
                with open(self.win_loss_file, 'r', encoding='utf-8') as f:
                    self.win_loss_data = json.load(f).get('records', [])
                    
        except Exception as e:
            self.data = []
            self.history = []
            self.win_loss_data = []
    
    def save_data(self):
        """Save data to files"""
        try:
            # Save main data
            data_dict = {
                'list': self.data,
                'total_records': len(self.data),
                'last_updated': datetime.now().isoformat(),
                'current_period': self.current_period,
                'next_period': self.next_period
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
            
            # Save history
            history_dict = {
                'predictions': self.history[:100],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, indent=2, ensure_ascii=False)
            
            # Save win/loss data
            win_loss_dict = {
                'records': self.win_loss_data[:200],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.win_loss_file, 'w', encoding='utf-8') as f:
                json.dump(win_loss_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            pass
    
    def fetch_latest_data(self):
        """Fetch latest data from API"""
        try:
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'list' in data['data']:
                    return data['data']['list']
        except Exception as e:
            pass
        return []
    
    def process_new_data(self, new_records):
        """Process new data records"""
        current_issue_numbers = {item.get('issueNumber') for item in self.data}
        added = 0
        
        for record in new_records:
            issue_no = record.get('issueNumber')
            if issue_no and issue_no not in current_issue_numbers:
                self.data.insert(0, record)
                added += 1
                
                # Keep only recent data
                if len(self.data) > 200:
                    self.data = self.data[:200]
        
        if added > 0:
            self.total_records = len(self.data)
            self.calculate_periods()
            self.save_data()
            
            # Retrain if enough new data
            if added >= 3 and len(self.data) >= self.ai_threshold and AI_AVAILABLE:
                self.train_all_models()
                
        return added
    
    def background_data_collector(self):
        """Background thread for data collection"""
        while True:
            try:
                new_data = self.fetch_latest_data()
                if new_data:
                    added = self.process_new_data(new_data)
                    if added > 0:
                        self.check_predictions_against_results()
                time.sleep(10)
            except Exception as e:
                time.sleep(30)
    
    def auto_prediction_system(self):
        """Auto-prediction system"""
        while True:
            try:
                for user_id in list(prediction_queue.keys()):
                    if prediction_queue[user_id]['active']:
                        target_wins = prediction_queue[user_id]['target_wins']
                        current_wins = user_wins.get(user_id, 0)
                        
                        # Check if target wins reached
                        if current_wins >= target_wins:
                            prediction_queue[user_id]['active'] = False
                            try:
                                bot.send_message(user_id, f"🎉 **TARGET ACHIEVED!**\n\nYou won {current_wins} times!\n\nAuto mode stopped. Start again for new target.", parse_mode="Markdown")
                                bot.send_sticker(user_id, WIN_STICKER)
                            except:
                                pass
                            continue
                        
                        # Get prediction
                        prediction = self.get_next_prediction()
                        
                        # Send prediction
                        try:
                            next_period = prediction.get('next_period', 'N/A')
                            pred_number = prediction.get('predicted_number', 'N/A')
                            pred_size = prediction.get('predicted_size', 'N/A')
                            pred_color = prediction.get('predicted_color', 'N/A')
                            confidence = prediction.get('confidence', 70.0)
                            
                            prediction_text = f"""
**PERIOD :** `{next_period}`

🎯 Prediction: `{pred_number} ({pred_size})`
🎨 Color: `{pred_color}`
📊 Confidence: `{confidence:.1f}%`

🎯 Target: {current_wins}/{target_wins} wins
🔁 Auto Mode: ACTIVE
                            """
                            
                            bot.send_message(user_id, prediction_text, parse_mode="Markdown")
                            
                        except Exception as e:
                            pass
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                time.sleep(60)
    
    def check_predictions_against_results(self):
        """Check previous predictions against actual results"""
        try:
            if len(self.data) < 2 or len(self.history) == 0:
                return
            
            latest_result = self.data[0]
            actual_period = latest_result.get('issueNumber')
            actual_number = latest_result.get('number', '-1')
            actual_color = latest_result.get('color', '').split(',')[0].capitalize()
            
            # Check each prediction in history
            for prediction in self.history[:20]:  # Check last 20 predictions
                predicted_period = prediction.get('next_period')
                
                if predicted_period == actual_period and not prediction.get('verified'):
                    predicted_number = prediction.get('predicted_number')
                    predicted_color = prediction.get('predicted_color')
                    
                    # Determine win/loss
                    is_win = False
                    win_type = ""
                    
                    try:
                        actual_num = int(actual_number)
                        predicted_num = int(predicted_number)
                        
                        # Check number match
                        if actual_num == predicted_num:
                            is_win = True
                            win_type = "NUMBER WIN"
                        # Check size match
                        elif (actual_num >= 5 and predicted_num >= 5) or (actual_num < 5 and predicted_num < 5):
                            is_win = True
                            win_type = "SIZE WIN"
                        # Check color match
                        elif actual_color.lower() == predicted_color.lower():
                            is_win = True
                            win_type = "COLOR WIN"
                    except:
                        pass
                    
                    # Update prediction record
                    prediction['verified'] = True
                    prediction['actual_period'] = actual_period
                    prediction['actual_number'] = actual_number
                    prediction['actual_color'] = actual_color
                    prediction['result'] = "WIN" if is_win else "LOSS"
                    prediction['win_type'] = win_type if is_win else "LOSS"
                    prediction['verified_at'] = datetime.now().isoformat()
                    
                    # Add to win/loss records
                    win_loss_record = {
                        'period': actual_period,
                        'predicted_number': predicted_number,
                        'actual_number': actual_number,
                        'predicted_color': predicted_color,
                        'actual_color': actual_color,
                        'result': "WIN" if is_win else "LOSS",
                        'win_type': win_type if is_win else "",
                        'timestamp': datetime.now().isoformat()
                    }
                    self.win_loss_data.insert(0, win_loss_record)
                    
                    # Keep only recent records
                    if len(self.win_loss_data) > 200:
                        self.win_loss_data = self.win_loss_data[:200]
                    
                    # Send win/loss notification
                    self.send_win_loss_notification(prediction, is_win, win_type)
            
            self.save_data()
            
        except Exception as e:
            pass
    
    def send_win_loss_notification(self, prediction, is_win, win_type):
        """Send win/loss notification to users"""
        try:
            predicted_period = prediction.get('next_period')
            predicted_number = prediction.get('predicted_number')
            actual_number = prediction.get('actual_number')
            predicted_color = prediction.get('predicted_color')
            actual_color = prediction.get('actual_color')
            
            # Check all users who might be interested
            for user_id in list(prediction_queue.keys()):
                if prediction_queue[user_id]['active']:
                    try:
                        if is_win:
                            # Update user wins count
                            if user_id not in user_wins:
                                user_wins[user_id] = 0
                            user_wins[user_id] += 1
                            
                            # Send win message
                            message = f"""
🎉 **WIN DETECTED!** 🎉

✅ Period: {predicted_period}
✅ Prediction: {predicted_number} ({predicted_color})
✅ Actual: {actual_number} ({actual_color})
✅ Win Type: {win_type}

🎯 Your Wins: {user_wins[user_id]}/{prediction_queue[user_id]['target_wins']}
                            """
                            bot.send_message(user_id, message, parse_mode="Markdown")
                            bot.send_sticker(user_id, WIN_STICKER)
                        else:
                            # Send loss message
                            message = f"""
😔 **LOSS DETECTED**

❌ Period: {predicted_period}
❌ Prediction: {predicted_number} ({predicted_color})
❌ Actual: {actual_number} ({actual_color})

🎯 Your Wins: {user_wins.get(user_id, 0)}/{prediction_queue[user_id]['target_wins']}
                            """
                            bot.send_message(user_id, message, parse_mode="Markdown")
                            bot.send_sticker(user_id, LOSS_STICKER)
                    except:
                        pass
        except:
            pass
    
    def create_features(self, data):
        """Create advanced features for ML models"""
        numbers = []
        for item in data[:50]:
            try:
                num = int(item.get('number', -1))
                if 0 <= num <= 9:
                    numbers.append(num)
            except:
                continue
        
        if len(numbers) < self.sequence_length + 5:
            return None, None
        
        features = []
        labels = []
        
        for i in range(len(numbers) - self.sequence_length - 1):
            seq = numbers[i:i+self.sequence_length]
            next_num = numbers[i + self.sequence_length]
            
            feature_vector = [
                np.mean(seq),
                np.std(seq),
                np.median(seq),
                seq[-1] - seq[0],
                seq[-1] - seq[-3] if len(seq) > 3 else 0,
                Counter(seq).most_common(1)[0][0],
                len(set(seq)),
                sum(1 for n in seq if n >= 5) / len(seq)
            ]
            
            features.append(feature_vector)
            labels.append(next_num)
        
        return np.array(features), np.array(labels)
    
    def create_sequences(self, data):
        """Create sequences for deep learning models"""
        numbers = []
        for item in data[:50]:
            try:
                num = int(item.get('number', -1))
                if 0 <= num <= 9:
                    numbers.append(num)
            except:
                continue
        
        if len(numbers) < self.sequence_length + 5:
            return None, None
        
        sequences = []
        labels = []
        
        for i in range(len(numbers) - self.sequence_length - 1):
            seq = numbers[i:i+self.sequence_length]
            next_num = numbers[i + self.sequence_length]
            
            seq_features = []
            for j, num in enumerate(seq):
                seq_features.append([
                    num / 9.0,
                    (num % 2 == 0) * 1,
                    1 if num >= 5 else 0,
                    j / self.sequence_length,
                    np.sin(2 * np.pi * j / self.sequence_length),
                    np.cos(2 * np.pi * j / self.sequence_length),
                    (num - seq[j-1])/9.0 if j > 0 else 0,
                    np.mean(seq[max(0, j-2):j+1])/9.0 if j >= 2 else 0
                ])
            
            sequences.append(seq_features)
            labels.append(next_num)
        
        if AI_AVAILABLE:
            return np.array(sequences), tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        else:
            categorical_labels = []
            for num in labels:
                one_hot = [0] * self.num_classes
                one_hot[num] = 1
                categorical_labels.append(one_hot)
            return np.array(sequences), np.array(categorical_labels)
    
    def train_all_models(self):
        """Train ALL AI models"""
        if not AI_AVAILABLE or len(self.data) < self.ai_threshold:
            return False
        
        try:
            # Train ML Models
            if SKLEARN_AVAILABLE:
                X, y = self.create_features(self.data)
                if X is not None and len(X) > 10:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train individual models
                    if self.rf_model:
                        self.rf_model.fit(X_train_scaled, y_train)
                    if self.mlp_model:
                        self.mlp_model.fit(X_train_scaled, y_train)
                    if self.gb_model:
                        self.gb_model.fit(X_train_scaled, y_train)
                    if self.svm_model:
                        self.svm_model.fit(X_train_scaled, y_train)
                    if self.lr_model:
                        self.lr_model.fit(X_train_scaled, y_train)
                    
                    # Train XGBoost
                    if XGBOOST_AVAILABLE and self.xgb_model:
                        self.xgb_model.fit(X_train_scaled, y_train)
                    
                    # Train Voting Ensemble
                    if self.voting_model:
                        self.voting_model.fit(X_train_scaled, y_train)
            
            # Train Deep Learning Models
            if TENSORFLOW_AVAILABLE:
                X_seq, y_seq = self.create_sequences(self.data)
                if X_seq is not None and len(X_seq) > 10:
                    split_idx = int(0.8 * len(X_seq))
                    X_train_seq = X_seq[:split_idx]
                    y_train_seq = y_seq[:split_idx]
                    X_val_seq = X_seq[split_idx:]
                    y_val_seq = y_seq[split_idx:]
                    
                    # Train LSTM
                    self.lstm_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=20,
                        batch_size=16,
                        validation_data=(X_val_seq, y_val_seq),
                        verbose=0
                    )
                    
                    # Train CNN
                    self.cnn_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=20,
                        batch_size=16,
                        validation_data=(X_val_seq, y_val_seq),
                        verbose=0
                    )
            
            self.models_trained = True
            return True
            
        except Exception as e:
            return False
    
    def get_size_prediction(self, number):
        """Get Big/Small prediction"""
        try:
            num = int(number)
            if num >= 5:
                return 'Big'
            else:
                return 'Small'
        except:
            return 'Big'
    
    def get_color_prediction(self, recent_data):
        """Predict color based on recent data"""
        colors = []
        for item in recent_data[:15]:
            color = item.get('color', '').split(',')[0].lower()
            if color in ['red', 'green', 'violet']:
                colors.append(color)
        
        if not colors:
            return 'Red'
        
        recent_weight = 0.7
        if len(colors) >= 5:
            recent_colors = colors[:5]
            all_colors = colors
        else:
            recent_colors = colors
            all_colors = colors
        
        recent_counts = Counter(recent_colors)
        all_counts = Counter(all_colors)
        
        color_scores = {}
        for color in set(colors):
            recent_prob = recent_counts.get(color, 0) / len(recent_colors) if recent_colors else 0
            all_prob = all_counts.get(color, 0) / len(all_colors) if all_colors else 0
            
            weighted_score = (recent_prob * recent_weight) + (all_prob * (1 - recent_weight))
            color_scores[color] = weighted_score
        
        best_color = max(color_scores, key=color_scores.get)
        return best_color.capitalize()
    
    def predict_with_all_models(self):
        """Make prediction using ALL AI models"""
        if not AI_AVAILABLE or not self.models_trained or len(self.data) < 20:
            return self.statistical_prediction()
        
        try:
            predictions = {}
            confidences = {}
            
            # 🔥 ML MODEL PREDICTIONS
            if SKLEARN_AVAILABLE:
                X, _ = self.create_features(self.data)
                if X is not None and len(X) > 0:
                    latest_features = X[-1:].reshape(1, -1)
                    
                    scaler = StandardScaler()
                    X_all, _ = self.create_features(self.data)
                    if X_all is not None and len(X_all) > 0:
                        scaler.fit(X_all)
                        latest_scaled = scaler.transform(latest_features)
                        
                        ml_models = []
                        if self.rf_model: ml_models.append(('RF', self.rf_model, 1.2))
                        if self.mlp_model: ml_models.append(('MLP', self.mlp_model, 1.1))
                        if self.gb_model: ml_models.append(('GB', self.gb_model, 1.0))
                        if self.svm_model: ml_models.append(('SVM', self.svm_model, 1.0))
                        if self.lr_model: ml_models.append(('LR', self.lr_model, 0.9))
                        
                        for name, model, weight in ml_models:
                            try:
                                pred = model.predict(latest_scaled)[0]
                                predictions[name] = int(pred)
                                
                                if hasattr(model, 'predict_proba'):
                                    probas = model.predict_proba(latest_scaled)[0]
                                    pred_confidence = np.max(probas) * 100
                                else:
                                    pred_confidence = 75.0
                                
                                confidences[name] = pred_confidence * weight
                            except:
                                continue
            
            # 🔥 XGBOOST PREDICTION
            if XGBOOST_AVAILABLE and self.xgb_model:
                try:
                    X, _ = self.create_features(self.data)
                    if X is not None and len(X) > 0:
                        latest_features = X[-1:].reshape(1, -1)
                        scaler = StandardScaler()
                        X_all, _ = self.create_features(self.data)
                        if X_all is not None and len(X_all) > 0:
                            scaler.fit(X_all)
                            latest_scaled = scaler.transform(latest_features)
                            pred = self.xgb_model.predict(latest_scaled)[0]
                            predictions['XGB'] = int(pred)
                            
                            if hasattr(self.xgb_model, 'predict_proba'):
                                probas = self.xgb_model.predict_proba(latest_scaled)[0]
                                xgb_confidence = np.max(probas) * 100
                            else:
                                xgb_confidence = 78.0
                            
                            confidences['XGB'] = xgb_confidence
                except:
                    pass
            
            # 🔥 DEEP LEARNING PREDICTIONS
            if TENSORFLOW_AVAILABLE:
                X_seq, _ = self.create_sequences(self.data)
                if X_seq is not None and len(X_seq) > 0:
                    latest_sequence = X_seq[-1:]
                    
                    # LSTM Prediction
                    lstm_pred = self.lstm_model.predict(latest_sequence, verbose=0)
                    lstm_num = np.argmax(lstm_pred)
                    predictions['LSTM'] = int(lstm_num)
                    confidences['LSTM'] = float(np.max(lstm_pred)) * 100 * 1.2
                    
                    # CNN Prediction
                    cnn_pred = self.cnn_model.predict(latest_sequence, verbose=0)
                    cnn_num = np.argmax(cnn_pred)
                    predictions['CNN'] = int(cnn_num)
                    confidences['CNN'] = float(np.max(cnn_pred)) * 100 * 1.1
            
            # 🔥 VOTING ENSEMBLE PREDICTION
            if self.voting_model and len(predictions) >= 2:
                try:
                    X, _ = self.create_features(self.data)
                    if X is not None and len(X) > 0:
                        latest_features = X[-1:].reshape(1, -1)
                        scaler = StandardScaler()
                        X_all, _ = self.create_features(self.data)
                        if X_all is not None and len(X_all) > 0:
                            scaler.fit(X_all)
                            latest_scaled = scaler.transform(latest_features)
                            vote_pred = self.voting_model.predict(latest_scaled)[0]
                            predictions['ENSEMBLE'] = int(vote_pred)
                            
                            if hasattr(self.voting_model, 'predict_proba'):
                                probas = self.voting_model.predict_proba(latest_scaled)[0]
                                vote_confidence = np.max(probas) * 100
                            else:
                                vote_confidence = 85.0
                            
                            confidences['ENSEMBLE'] = vote_confidence
                except:
                    pass
            
            # 🔥 FINAL DECISION: Weighted voting
            if predictions:
                weighted_votes = {}
                total_confidence = 0
                for model_name, number in predictions.items():
                    confidence = confidences.get(model_name, 70.0)
                    weighted_votes[number] = weighted_votes.get(number, 0) + confidence
                    total_confidence += confidence
                
                final_number = max(weighted_votes, key=weighted_votes.get)
                final_confidence = weighted_votes[final_number] / len(predictions)
                
                final_size = self.get_size_prediction(final_number)
                final_color = self.get_color_prediction(self.data)
                
                return {
                    'predicted_number': final_number,
                    'predicted_size': final_size,
                    'predicted_color': final_color,
                    'confidence': final_confidence,
                    'ai_used': True
                }
            else:
                return self.statistical_prediction()
                
        except Exception as e:
            return self.statistical_prediction()
    
    def statistical_prediction(self):
        """Statistical fallback prediction"""
        numbers = []
        for item in self.data[:30]:
            try:
                num = int(item.get('number', -1))
                if 0 <= num <= 9:
                    numbers.append(num)
            except:
                continue
        
        if not numbers:
            pred_num = 5
        else:
            weights = np.linspace(0.3, 1.0, len(numbers))
            weighted_avg = np.average(numbers, weights=weights)
            pred_num = int(round(weighted_avg))
            pred_num = max(0, min(9, pred_num))
        
        pred_size = self.get_size_prediction(pred_num)
        pred_color = self.get_color_prediction(self.data)
        
        return {
            'predicted_number': pred_num,
            'predicted_size': pred_size,
            'predicted_color': pred_color,
            'confidence': 70.0,
            'ai_used': False
        }
    
    def get_next_prediction(self):
        """Get next prediction with AI and period information"""
        prediction = self.predict_with_all_models()
        
        prediction['current_period'] = self.current_period
        prediction['next_period'] = self.next_period
        prediction['timestamp'] = datetime.now().strftime('%H:%M:%S')
        
        # Save to history
        history_entry = prediction.copy()
        history_entry['period'] = self.next_period
        self.history.insert(0, history_entry)
        
        if len(self.history) > 100:
            self.history = self.history[:100]
        
        self.save_data()
        
        return prediction
    
    def get_system_status(self):
        """Get system status information"""
        return {
            'ai_available': AI_AVAILABLE,
            'models_trained': self.models_trained,
            'total_records': self.total_records,
            'current_period': self.current_period,
            'next_period': self.next_period,
            'active_models': self.active_models,
            'win_loss_stats': self.get_win_loss_stats()
        }
    
    def get_win_loss_stats(self):
        """Get win/loss statistics"""
        if not self.win_loss_data:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0}
        
        total = len(self.win_loss_data)
        wins = sum(1 for record in self.win_loss_data if record.get('result') == 'WIN')
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate
        }
    
    def get_recent_history(self, limit=10):
        """Get recent prediction history"""
        return self.win_loss_data[:limit]

# Initialize predictor
wingo_predictor = AdvancedWinGoPredictor()

# --- HELPER FUNCTIONS ---
def check_channel_status(user_id):
    user_status = {
        'channel1': {'is_member': False, 'has_requested': False},
        'channel2': {'is_member': False, 'has_requested': False}
    }
    
    try:
        member1 = bot.get_chat_member(CHANNEL_INFO['channel1']['id'], user_id)
        user_status['channel1']['is_member'] = member1.status in ['member', 'administrator', 'creator']
    except:
        pass
    
    try:
        member2 = bot.get_chat_member(CHANNEL_INFO['channel2']['id'], user_id)
        user_status['channel2']['is_member'] = member2.status in ['member', 'administrator', 'creator']
    except:
        pass
    
    if user_id in pending_requests:
        user_status['channel1']['has_requested'] = CHANNEL_INFO['channel1']['id'] in pending_requests[user_id]
        user_status['channel2']['has_requested'] = CHANNEL_INFO['channel2']['id'] in pending_requests[user_id]
    
    return user_status

def is_verified(user_id):
    status = check_channel_status(user_id)
    channel1_ok = status['channel1']['is_member'] or status['channel1']['has_requested']
    channel2_ok = status['channel2']['is_member'] or status['channel2']['has_requested']
    return channel1_ok and channel2_ok

# --- HANDLERS ---
@bot.chat_join_request_handler()
def handle_chat_join_request(join_request):
    user_id = join_request.from_user.id
    chat_id = join_request.chat.id
    
    if user_id not in pending_requests:
        pending_requests[user_id] = set()
    
    pending_requests[user_id].add(chat_id)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    
    if is_verified(user_id):
        key_request_msg = bot.send_message(
            chat_id, 
            "✅ 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐒𝐔𝐂𝐂𝐄𝐒𝐒𝐅𝐔𝐋!\n\n"
            "𝐘𝐎𝐔𝐑 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐈𝐒 𝐂𝐎𝐌𝐏𝐋𝐄𝐓𝐄. 𝐘𝐎𝐔 𝐂𝐀𝐍 𝐍𝐎𝐖 𝐔𝐒𝐄 𝐓𝐇𝐄 𝐁𝐎𝐓 🚀\n\n"
            "🔑 𝐍𝐎𝐖 𝐏𝐋𝐄𝐀𝐒𝐄 𝐄𝐍𝐓𝐄𝐑 𝐓𝐇𝐄 𝐒𝐄𝐂𝐑𝐄𝐓 𝐊𝐄𝐘 𝐓𝐎 𝐀𝐂𝐂𝐄𝐒𝐒 𝐁𝐎𝐓:\n"
            "(𝐀𝐒𝐊 𝐀𝐃𝐌𝐈𝐍 𝐅𝐎𝐑 𝐊𝐄𝐘)",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(key_request_msg, process_key_input)
        return
    
    loading_msg = bot.send_message(
        chat_id, 
        "‎ ‎ ‎ ‎ ‎ ‎ ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▱▱▱▱▱▱▱▱▱▱▱▱▱ 0%\n\n‎ ‎ 𝗗𝗘𝗞𝗛 𝗞𝗬𝗔 𝗥𝗛𝗔 𝗛𝗔𝗜 𝗔𝗕𝗛𝗜 𝗟𝗜𝗚𝗛𝗧 𝗡𝗔𝗛𝗜 𝗛𝗔𝗜🔮.", 
        parse_mode="Markdown"
    )
    
    threading.Thread(
        target=simulate_loading, 
        args=(chat_id, loading_msg.message_id, user_id)
    ).start()

def process_key_input(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_input = message.text.strip()
    
    if user_input in VALID_KEYS:
        # Reset user wins if new session
        if user_id not in user_wins:
            user_wins[user_id] = 0
        
        bot.send_message(
            chat_id,
            "🔓 𝐀𝐂𝐂𝐄𝐒𝐒 𝐆𝐑𝐀𝐍𝐓𝐄𝐃!\n\n"
            "✅ 𝐊𝐄𝐘 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐒𝐔𝐂𝐂𝐄𝐒𝐅𝐔𝐋\n\n"
            f"🎯 Current Wins: {user_wins[user_id]}\n\n"
            "🎯 Please enter how many wins you want (1-100):",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(message, ask_target_wins)
    else:
        bot.send_message(
            chat_id,
            "❌ 𝐈𝐍𝐕𝐀𝐋𝐈𝐃 𝐊𝐄𝐘!\n\n"
            "𝐏𝐥𝐞𝐚𝐬𝐞 𝐜𝐨𝐧𝐭𝐚𝐜𝐭 𝐚𝐝𝐦𝐢𝐧 𝐟𝐨𝐫 𝐯𝐚𝐥𝐢𝐝 𝐤𝐞𝐲.\n\n"
            "📌 𝐓𝐫𝐲 𝐚𝐠𝐚𝐢𝐧: /start",
            parse_mode="Markdown"
        )

def ask_target_wins(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    try:
        target_wins = int(message.text.strip())
        if 1 <= target_wins <= 100:
            # Set target wins
            if user_id not in prediction_queue:
                prediction_queue[user_id] = {'active': False, 'target_wins': 0}
            
            prediction_queue[user_id]['target_wins'] = target_wins
            
            bot.send_message(
                chat_id,
                f"🎯 Target set: {target_wins} wins\n\n"
                f"✅ Current Wins: {user_wins.get(user_id, 0)}\n\n"
                "Now you can use all features:",
                parse_mode="Markdown"
            )
            
            show_main_menu(chat_id, user_id)
        else:
            bot.send_message(chat_id, "❌ Please enter number between 1-100")
            bot.register_next_step_handler(message, ask_target_wins)
    except:
        bot.send_message(chat_id, "❌ Invalid number. Please enter between 1-100")
        bot.register_next_step_handler(message, ask_target_wins)

def show_main_menu(chat_id, user_id):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    
    btn1 = types.KeyboardButton("📊 GET PREDICTION")
    btn2 = types.KeyboardButton("🎯 MY WINS")
    btn3 = types.KeyboardButton("🔄 AUTO MODE")
    btn4 = types.KeyboardButton("📈 STATUS")
    btn5 = types.KeyboardButton("📊 DATA")
    btn6 = types.KeyboardButton("📜 HISTORY")
    btn7 = types.KeyboardButton("💎 VIP CHANNEL")
    btn8 = types.KeyboardButton("ℹ️ HELP")
    btn9 = types.KeyboardButton("🛑 STOP AUTO")
    btn10 = types.KeyboardButton("🔄 RESET WINS")
    
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, btn10)
    
    bot.send_message(
        chat_id,
        f"🎯 𝐌𝐀𝐈𝐍 𝐌𝐄𝐍𝐔\n\n"
        f"🎯 Current Wins: {user_wins.get(user_id, 0)}\n"
        f"🎯 Target: {prediction_queue.get(user_id, {}).get('target_wins', 0)} wins\n\n"
        "Select an option:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

@bot.message_handler(func=lambda message: message.text == "📊 GET PREDICTION")
def handle_get_prediction(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    try:
        prediction = wingo_predictor.get_next_prediction()
        
        next_period = prediction.get('next_period', 'N/A')
        pred_number = prediction.get('predicted_number', 'N/A')
        pred_size = prediction.get('predicted_size', 'N/A')
        pred_color = prediction.get('predicted_color', 'N/A')
        confidence = prediction.get('confidence', 70.0)
        
        prediction_text = f"""
**PERIOD :** `{next_period}`

🎯 Prediction: `{pred_number} ({pred_size})`
🎨 Color: `{pred_color}`
📊 Confidence: `{confidence:.1f}%`

🎯 Your Wins: {user_wins.get(user_id, 0)}
🎯 Target: {prediction_queue.get(user_id, {}).get('target_wins', 0)}
        """
        
        bot.send_message(chat_id, prediction_text, parse_mode="Markdown")
        
    except Exception as e:
        bot.send_message(chat_id, "❌ Error generating prediction")

@bot.message_handler(func=lambda message: message.text == "🎯 MY WINS")
def handle_my_wins(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    current_wins = user_wins.get(user_id, 0)
    target_wins = prediction_queue.get(user_id, {}).get('target_wins', 0)
    
    response = f"""
🎯 **YOUR WIN STATS**

Current Wins: {current_wins}
Target Wins: {target_wins}
Progress: {current_wins}/{target_wins}
Auto Mode: {'🟢 ACTIVE' if prediction_queue.get(user_id, {}).get('active', False) else '🔴 INACTIVE'}

Win Types Detected:
• Number Match
• Size Match (Big/Small)
• Color Match

Use '🔄 RESET WINS' to reset counter
    """
    
    bot.send_message(chat_id, response, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == "🔄 AUTO MODE")
def handle_auto_mode(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    if prediction_queue.get(user_id, {}).get('active', False):
        bot.send_message(chat_id, "✅ Auto mode is already active!")
    else:
        prediction_queue[user_id] = {
            'active': True, 
            'target_wins': prediction_queue.get(user_id, {}).get('target_wins', 10)
        }
        bot.send_message(
            chat_id,
            "🔄 **AUTO MODE ACTIVATED**\n\n"
            f"🎯 Target Wins: {prediction_queue[user_id]['target_wins']}\n"
            f"🎯 Current Wins: {user_wins.get(user_id, 0)}\n\n"
            "Predictions will be sent automatically every 30 seconds.\n"
            "Auto mode will stop when:\n"
            "1. Target wins reached ✅\n"
            "2. You stop it manually 🛑\n\n"
            "Win/Loss notifications will be sent automatically!",
            parse_mode="Markdown"
        )

@bot.message_handler(func=lambda message: message.text == "🛑 STOP AUTO")
def handle_stop_auto(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    if user_id in prediction_queue:
        prediction_queue[user_id]['active'] = False
        bot.send_message(
            chat_id, 
            f"🛑 Auto mode stopped.\n\n"
            f"🎯 Final Wins: {user_wins.get(user_id, 0)}\n"
            f"🎯 Target: {prediction_queue[user_id]['target_wins']}",
            parse_mode="Markdown"
        )
    else:
        bot.send_message(chat_id, "ℹ️ Auto mode is not active.")

@bot.message_handler(func=lambda message: message.text == "🔄 RESET WINS")
def handle_reset_wins(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    user_wins[user_id] = 0
    bot.send_message(
        chat_id,
        "🔄 **WINS RESET**\n\n"
        "Your win counter has been reset to 0.\n"
        "Set a new target and start again!",
        parse_mode="Markdown"
    )

@bot.message_handler(func=lambda message: message.text == "📈 STATUS")
def handle_status(message):
    chat_id = message.chat.id
    
    status = wingo_predictor.get_system_status()
    win_stats = status['win_loss_stats']
    
    response = f"""
📈 **SYSTEM STATUS**

🤖 AI Status: {'🟢 ACTIVE' if status['ai_available'] else '🔴 INACTIVE'}
🧠 Models Trained: {'✅ YES' if status['models_trained'] else '❌ NO'}
📊 Total Records: {status['total_records']}

🎯 **ACTIVE MODELS:**
{', '.join(status['active_models']) if status['active_models'] else 'No models active'}

📈 **WIN/LOSS STATS:**
Total Predictions: {win_stats['total']}
✅ Wins: {win_stats['wins']}
❌ Losses: {win_stats['losses']}
📊 Win Rate: {win_stats['win_rate']:.1f}%

🔄 Current Period: {status['current_period']}
🎯 Next Period: {status['next_period']}
    """
    
    bot.send_message(chat_id, response, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == "📊 DATA")
def handle_data(message):
    chat_id = message.chat.id
    
    status = wingo_predictor.get_system_status()
    
    response = f"""
📊 **SYSTEM DATA**

Total Records: {status['total_records']}
Current Period: {status['current_period']}
Next Period: {status['next_period']}

🤖 **AI MODELS:**
- TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}
- Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}
- XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}

📈 **PERFORMANCE:**
Models Trained: {'✅' if status['models_trained'] else '❌'}
Active Models: {len(status['active_models'])}

🕒 Last Updated: {datetime.now().strftime('%H:%M:%S')}
    """
    
    bot.send_message(chat_id, response, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == "📜 HISTORY")
def handle_history(message):
    chat_id = message.chat.id
    
    recent_history = wingo_predictor.get_recent_history(10)
    
    if not recent_history:
        bot.send_message(chat_id, "📜 No prediction history available yet.")
        return
    
    response = "📜 **RECENT PREDICTION HISTORY**\n\n"
    
    for i, record in enumerate(recent_history[:10], 1):
        period = record.get('period', 'N/A')
        pred_num = record.get('predicted_number', 'N/A')
        actual_num = record.get('actual_number', 'N/A')
        result = record.get('result', 'N/A')
        win_type = record.get('win_type', '')
        
        emoji = "✅" if result == "WIN" else "❌"
        result_text = f"{emoji} {result}"
        if win_type:
            result_text += f" ({win_type})"
        
        response += f"{i}. Period: {period}\n"
        response += f"   Predicted: {pred_num} | Actual: {actual_num}\n"
        response += f"   Result: {result_text}\n"
        response += "   " + "-"*30 + "\n"
    
    win_stats = wingo_predictor.get_win_loss_stats()
    response += f"\n📊 **STATS:** Wins: {win_stats['wins']} | Losses: {win_stats['losses']} | Rate: {win_stats['win_rate']:.1f}%"
    
    bot.send_message(chat_id, response, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == "💎 VIP CHANNEL")
def handle_vip_channel(message):
    chat_id = message.chat.id
    markup = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton("Join VIP Channel", url='https://t.me/+Db9BlHtMooIyZGM1')
    markup.add(btn)
    
    bot.send_message(
        chat_id,
        "💎 **VIP CHANNEL ACCESS**\n\n"
        "Get exclusive predictions and signals!\n"
        "Click below to join:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

@bot.message_handler(func=lambda message: message.text == "ℹ️ HELP")
def handle_help(message):
    chat_id = message.chat.id
    markup = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton("Contact Admin", url='https://t.me/AKELAOFFICIAL1')
    markup.add(btn)
    
    bot.send_message(
        chat_id,
        "ℹ️ **HELP & SUPPORT**\n\n"
        "📋 **HOW TO USE:**\n"
        "1. Get prediction manually or use auto mode\n"
        "2. Win/Loss detected automatically\n"
        "3. Win counter tracks your success\n"
        "4. Set target wins for auto mode\n"
        "5. Check status for system info\n"
        "6. View history for past results\n\n"
        "🎯 **WIN TYPES:**\n"
        "- Number Match: Exact number prediction\n"
        "- Size Match: Big/Small prediction\n"
        "- Color Match: Color prediction\n\n"
        "🔄 **AUTO MODE:**\n"
        "- Predictions every 30 seconds\n"
        "- Stops when target wins reached\n"
        "- Win/Loss notifications automatic\n\n"
        "Need help? Contact admin:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

@bot.message_handler(commands=['win'])
def handle_win_command(message):
    chat_id = message.chat.id
    bot.send_sticker(chat_id, WIN_STICKER)
    bot.send_message(chat_id, "🎉 **WINNER!** Congratulations on your win! 🎉", parse_mode="Markdown")

@bot.message_handler(commands=['loss'])
def handle_loss_command(message):
    chat_id = message.chat.id
    bot.send_sticker(chat_id, LOSS_STICKER)
    bot.send_message(chat_id, "😔 **Loss Detected** - Better luck next time!", parse_mode="Markdown")

def simulate_loading(chat_id, message_id, user_id):
    loading_stages = [
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 10%\n\n‎ ‎ ☆ 𝗔𝗕𝗛𝗜 𝗟𝗜𝗚𝗛𝗧 𝗡𝗔𝗛𝗜 𝗛𝗔𝗜 𝗜𝗦𝗜 𝗟𝗜𝗬𝗘 𝗦𝗘𝗥𝗩𝗘𝗥 𝗢𝗙𝗙 𝗛𝗔𝗜 🫡 ", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 20%\n\n‎ ‎ ☆ 𝗥𝗨𝗞𝗢 𝗚𝗘𝗡𝗘𝗥𝗔𝗧𝗢𝗥 𝗢𝗡 𝗞𝗔𝗥 𝗥𝗛𝗔 𝗛𝗨 🤫 ", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱ 30%\n\n‎ ‎ ☆ 𝗥𝗨𝗞𝗢 𝗦𝗔𝗕𝗔𝗥 𝗞𝗔𝗥𝗢 😮‍💨", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎‎ ‎  ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱ 40%\n\n‎ ‎ ☆𝗪𝗔𝗜𝗧......", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱ 50%\n\n‎ ‎ ☆ 𝗦𝗧𝗔𝗥𝗧𝗜𝗡𝗚 𝗦𝗘𝗥𝗩𝗘𝗥 💥 ", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱60%\n\n‎ ‎ ☆  𝗙𝗘𝗧𝗖𝗛𝗜𝗡𝗚 𝗗𝗔𝗧𝗔 𝗙𝗥𝗢𝗠 𝗡𝗔𝗦𝗔 ☠️", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱ 70%\n\n‎ ‎ ☆ 𝗖𝗔𝗟𝗟𝗜𝗡𝗚 𝗜𝗦𝗥𝗢 𝗖𝗘𝗢 🪄", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱ 80%\n\n‎ ‎ ☆ 𝗛𝗔𝗖𝗞 𝗡𝗔𝗦𝗔 𝗦𝗘𝗥𝗩𝗘𝗥 😎", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎ ‎  \n\n▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱ 90%\n\n‎ ‎ ☆ 𝗔𝗟𝗠𝗢𝗦𝗧 𝗗𝗢𝗡𝗘 💫", "delay": 1.8},
        {"text": "‎ ‎ ‎ ‎ ‎ ‎‎ ‎  𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 𝗔𝗞𝗘𝗟𝗔 𝗔𝗜 𝗨𝗟𝗧𝗥𝗔‎  \n\n▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰ 100%\n\n‎ ‎ ☆ 𝗔𝗕 𝗔𝗣𝗞𝗔 𝗕𝗛𝗔𝗩𝗜𝗦𝗛𝗠𝗔𝗡𝗜 𝗬𝗔𝗡𝗧𝗥 𝗦𝗨𝗥𝗨 𝗛𝗢 𝗥𝗛𝗔 𝗛𝗔𝗜 ", "delay": 1.8}
    ]
    
    for stage in loading_stages:
        try:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=stage["text"],
                parse_mode="Markdown"
            )
        except:
            pass
        
        time.sleep(stage["delay"])

    send_verification_buttons(chat_id, message_id, user_id)

def send_verification_buttons(chat_id, message_id, user_id):
    verification_text = "𝗝𝗢𝗜𝗡 𝗔𝗟𝗟 𝗖𝗛𝗔𝗡𝗡𝗘𝗟𝗦 𝗧𝗢 𝗩𝗘𝗥𝗜𝗙𝗬 𝗕𝗢𝗧\n\n"
    
    markup = types.InlineKeyboardMarkup()
    
    channel1_button = types.InlineKeyboardButton("ᴀᴋᴇʟᴀ ᴀɪ ᴜʟᴛʀᴀ", url=CHANNEL_INFO['channel1']['link'])
    channel2_button = types.InlineKeyboardButton("sʜᴀᴅᴏᴡ ᴘʀᴇᴅɪᴄᴛɪᴏɴ", url=CHANNEL_INFO['channel2']['link'])
    
    verify_button = types.InlineKeyboardButton("𝗩𝗘𝗥𝗜𝗙𝗬", callback_data='verify_channels')
    
    markup.add(channel1_button, channel2_button)
    markup.add(verify_button)
    
    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=verification_text,
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, verification_text, reply_markup=markup, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data == 'verify_channels')
def callback_handler(call):
    user_id = call.from_user.id
    
    bot.answer_callback_query(call.id, "𝐂𝐇𝐄𝐂𝐊𝐈𝐍𝐆 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐒𝐓𝐀𝐓𝐔𝐒...")
    
    if is_verified(user_id):
        verification_status = (
            "✅ 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐒𝐔𝐂𝐂𝐄𝐒𝐒𝐅𝐔𝐋!\n\n"
            "𝐘𝐎𝐔𝐑 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐈𝐒 𝐂𝐎𝐌𝐏𝐋𝐄𝐓𝐄. 𝐘𝐎𝐔 𝐂𝐀𝐍 𝐍𝐎𝐖 𝐔𝐒𝐄 𝐓𝐇𝐄 𝐁𝐎𝐓 🚀\n\n"
            "🔑 𝐍𝐎𝐖 𝐏𝐋𝐄𝐀𝐒𝐄 𝐄𝐍𝐓𝐄𝐑 𝐓𝐇𝐄 𝐒𝐄𝐂𝐑𝐄𝐓 𝐊𝐄𝐘 𝐓𝐎 𝐀𝐂𝐂𝐄𝐒𝐒 𝐁𝐎𝐓:\n"
            "(𝐀𝐒𝐊 𝐀𝐃𝐌𝐈𝐍 𝐅𝐎𝐑 𝐊𝐄𝐘)"
        )
        
        try:
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=verification_status,
                parse_mode="Markdown"
            )
        except Exception as e:
            bot.send_message(call.message.chat.id, verification_status, parse_mode="Markdown")
        
        bot.register_next_step_handler(call.message, process_key_input)
        
    else:
        status = check_channel_status(user_id)
        alert_text = "❌ 𝐕𝐄𝐑𝐈𝐅𝐈𝐂𝐀𝐓𝐈𝐎𝐍 𝐅𝐀𝐈𝐋𝐄𝐃!\n\n"
        
        if not (status['channel1']['is_member'] or status['channel1']['has_requested']):
            alert_text += "• 𝐘𝐎𝐔 𝐇𝐀𝐕𝐄 𝐍𝐎𝐓 𝐉𝐎𝐈𝐍𝐄𝐃 𝐂𝐇𝐀𝐍𝐍𝐄𝐋 𝟏\n"
        
        if not (status['channel2']['is_member'] or status['channel2']['has_requested']):
            alert_text += "• 𝐘𝐎𝐔 𝐇𝐀𝐕𝐄 𝐍𝐎𝐓 𝐉𝐎𝐈𝐍𝐄𝐃 𝐂𝐇𝐀𝐍𝐍𝐄𝐋 𝟐\n"
        
        alert_text += "\n𝐏𝐋𝐄𝐀𝐒𝐄 𝐉𝐎𝐈𝐍 𝐁𝐎𝐓𝐇 𝐂𝐇𝐀𝐍𝐍𝐄𝐋𝐒 𝐎𝐑 𝐒𝐄𝐍𝐃 𝐉𝐎𝐈𝐍 𝐑𝐄𝐐𝐔𝐄𝐒𝐓 𝐀𝐍𝐃 𝐓𝐑𝐘 𝐀𝐆𝐀𝐈𝐍."
        
        bot.answer_callback_query(call.id, alert_text, show_alert=True)

# --- START BOT ---
print("=" * 60)
print("🤖 Akela AI Telegram Bot - WIN/LOSS SYSTEM")
print("🎯 Complete Prediction System with Win/Loss Detection")
print("✅ No Payment System | 🎯 Win Tracking Only")
print("=" * 60)
print(f"✅ TensorFlow: {'AVAILABLE' if TENSORFLOW_AVAILABLE else 'NOT AVAILABLE'}")
print(f"✅ Scikit-learn: {'AVAILABLE' if SKLEARN_AVAILABLE else 'NOT AVAILABLE'}")
print(f"✅ XGBoost: {'AVAILABLE' if XGBOOST_AVAILABLE else 'NOT AVAILABLE'}")
print(f"✅ AI Mode: {'ACTIVE' if AI_AVAILABLE else 'BASIC'}")
print(f"✅ Active Models: {len(wingo_predictor.active_models)}")
print("=" * 60)

try:
    bot.infinity_polling()
except Exception as e:
    print(f"❌ Bot polling failed: {e}")
