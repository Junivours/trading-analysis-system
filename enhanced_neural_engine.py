"""
🤖 ENHANCED NEURAL NETWORK ENGINE
=================================
- LSTM for time series prediction
- Ensemble models (multiple algorithms)
- Auto-training with historical data
- Advanced feature engineering
- Prediction accuracy tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import pickle
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TF_AVAILABLE = True
    print("✅ TensorFlow available for enhanced neural networks")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available - using simplified models")

class EnhancedNeuralEngine:
    """🤖 Advanced Neural Network with LSTM and Ensemble Models"""
    
    def __init__(self):
        self.tf_available = TF_AVAILABLE
        self.models = {}
        self.scalers = {}
        self.model_accuracy = {}
        self.sequence_length = 60  # Look back 60 periods
        self.prediction_horizon = [1, 4, 24]  # 1h, 4h, 24h predictions
        
        # Model paths
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models on startup
        self.load_existing_models()
        
        print("🤖 Enhanced Neural Network Engine initialized")
    
    def get_model_path(self, symbol: str, interval: str, horizon: int, model_type: str = "lstm") -> str:
        """Get the file path for a specific model"""
        filename = f"{symbol}_{interval}_{horizon}h_{model_type}.h5"
        return os.path.join(self.model_dir, filename)
    
    def get_scaler_path(self, symbol: str, interval: str, horizon: int) -> str:
        """Get the file path for a specific scaler"""
        filename = f"{symbol}_{interval}_{horizon}h_scaler.pkl"
        return os.path.join(self.model_dir, filename)
    
    def get_metadata_path(self, symbol: str, interval: str, horizon: int) -> str:
        """Get the file path for model metadata"""
        filename = f"{symbol}_{interval}_{horizon}h_metadata.json"
        return os.path.join(self.model_dir, filename)
    
    def save_model(self, symbol: str, interval: str, horizon: int, model, scaler, accuracy: float):
        """Save trained model, scaler and metadata"""
        try:
            model_path = self.get_model_path(symbol, interval, horizon)
            scaler_path = self.get_scaler_path(symbol, interval, horizon)
            metadata_path = self.get_metadata_path(symbol, interval, horizon)
            
            # Save TensorFlow model
            model.save(model_path)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'horizon': horizon,
                'accuracy': accuracy,
                'trained_at': datetime.now().isoformat(),
                'sequence_length': self.sequence_length
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model saved: {symbol}_{interval}_{horizon}h (accuracy: {accuracy:.3f})")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, symbol: str, interval: str, horizon: int) -> Tuple[Optional[object], Optional[object], Optional[float]]:
        """Load trained model, scaler and metadata"""
        try:
            model_path = self.get_model_path(symbol, interval, horizon)
            scaler_path = self.get_scaler_path(symbol, interval, horizon)
            metadata_path = self.get_metadata_path(symbol, interval, horizon)
            
            # Check if all files exist
            if not all(os.path.exists(path) for path in [model_path, scaler_path, metadata_path]):
                return None, None, None
            
            # Load model
            model = load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            accuracy = metadata.get('accuracy', 0.0)
            trained_at = metadata.get('trained_at', 'Unknown')
            
            print(f"✅ Model loaded: {symbol}_{interval}_{horizon}h (accuracy: {accuracy:.3f}, trained: {trained_at})")
            return model, scaler, accuracy
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None, None
    
    def load_existing_models(self):
        """Load all existing models on startup"""
        if not os.path.exists(self.model_dir):
            return
        
        print("🔍 Scanning for existing trained models...")
        loaded_count = 0
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_metadata.json'):
                try:
                    # Parse filename to extract symbol, interval, horizon
                    name_parts = filename.replace('_metadata.json', '').split('_')
                    if len(name_parts) >= 3:
                        symbol = name_parts[0]
                        interval = name_parts[1]
                        horizon_str = name_parts[2].replace('h', '')
                        horizon = int(horizon_str)
                        
                        # Try to load the model
                        model, scaler, accuracy = self.load_model(symbol, interval, horizon)
                        if model is not None:
                            model_key = f"{symbol}_{interval}_{horizon}h"
                            self.models[model_key] = model
                            self.scalers[model_key] = scaler
                            self.model_accuracy[model_key] = accuracy
                            loaded_count += 1
                            
                except Exception as e:
                    print(f"⚠️ Could not load model {filename}: {e}")
        
        if loaded_count > 0:
            print(f"🎯 Loaded {loaded_count} pre-trained models from disk")
        else:
            print("📝 No existing models found - training will be required")
    
    def is_model_trained(self, symbol: str, interval: str, horizon: int) -> bool:
        """Check if a model is already trained and loaded"""
        model_key = f"{symbol}_{interval}_{horizon}h"
        return model_key in self.models and self.models[model_key] is not None
        
    def fetch_training_data(self, symbol: str, interval: str, days: int = 365) -> pd.DataFrame:
        """Fetch comprehensive training data"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            df_data = []
            for candle in data:
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0], unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'trades': int(candle[8])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"📈 Fetched {len(df)} candles for training")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for neural networks"""
        try:
            # Technical Indicators
            
            # RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            df['rsi'] = calculate_rsi(df['close'])
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Moving Averages
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Price Features
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(periods=2)
            df['price_change_5'] = df['close'].pct_change(periods=5)
            
            # Volume Features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()
            df['atr'] = (df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)).rolling(window=14).mean()
            
            # Market Structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            
            # Time Features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            
            # Lag Features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            # Future returns (targets)
            for horizon in self.prediction_horizon:
                df[f'future_return_{horizon}h'] = df['close'].shift(-horizon).pct_change()
                df[f'future_direction_{horizon}h'] = (df[f'future_return_{horizon}h'] > 0).astype(int)
            
            # Remove rows with NaN values
            df.dropna(inplace=True)
            
            print(f"🔧 Engineered {len(df.columns)} features from {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            return df
    
    def prepare_lstm_data(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        try:
            # Select feature columns (exclude target columns)
            feature_cols = [col for col in df.columns if not col.startswith('future_')]
            X_data = df[feature_cols].values
            y_data = df[target_column].values
            
            # Scale features
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_scaled = scaler_X.fit_transform(X_data)
            y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(self.sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-self.sequence_length:i])
                y_sequences.append(y_scaled[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Store scalers
            self.scalers[f'X_{target_column}'] = scaler_X
            self.scalers[f'y_{target_column}'] = scaler_y
            
            print(f"📊 Prepared LSTM data: {X_sequences.shape} -> {y_sequences.shape}")
            return X_sequences, y_sequences
            
        except Exception as e:
            print(f"❌ Error preparing LSTM data: {e}")
            return np.array([]), np.array([])
    
    def build_lstm_model(self, input_shape: Tuple, model_name: str = "default") -> Optional[object]:
        """Build advanced LSTM model"""
        if not self.tf_available:
            print("⚠️ TensorFlow not available - cannot build LSTM")
            return None
            
        try:
            model = Sequential([
                # First LSTM layer
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                
                # Second LSTM layer
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                
                # Third LSTM layer
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                
                # Dense layers
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            print(f"🏗️ Built LSTM model '{model_name}' with {model.count_params():,} parameters")
            return model
            
        except Exception as e:
            print(f"❌ Error building LSTM model: {e}")
            return None
    
    def train_lstm_model(self, symbol: str, interval: str, horizon: int) -> bool:
        """Train LSTM model for specific prediction horizon"""
        if not self.tf_available:
            print("⚠️ TensorFlow not available - cannot train LSTM")
            return False
            
        try:
            print(f"🎯 Training LSTM model for {horizon}h prediction")
            
            # Fetch and prepare data
            df = self.fetch_training_data(symbol, interval, days=500)
            if df.empty:
                return False
                
            df = self.engineer_features(df)
            if df.empty:
                return False
            
            # Prepare LSTM data
            target_col = f'future_return_{horizon}h'
            X, y = self.prepare_lstm_data(df, target_col)
            
            if len(X) == 0:
                return False
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            model_name = f"lstm_{symbol}_{interval}_{horizon}h"
            model = self.build_lstm_model((X.shape[1], X.shape[2]), model_name)
            
            if model is None:
                return False
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
            ]
            
            # Train model
            print(f"🚀 Training started...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            # Make predictions for accuracy
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform for real values
            scaler_y = self.scalers[f'y_{target_col}']
            y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_real = scaler_y.inverse_transform(y_pred).flatten()
            
            # Calculate accuracy metrics
            mse = mean_squared_error(y_test_real, y_pred_real)
            mae = mean_absolute_error(y_test_real, y_pred_real)
            
            # Direction accuracy
            direction_accuracy = np.mean((y_test_real > 0) == (y_pred_real > 0)) * 100
            
            # Store model and metrics
            self.models[model_name] = model
            self.model_accuracy[model_name] = {
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            
            # Get the scaler for saving
            scaler_X = self.scalers[f'X_{target_col}']
            
            # Save model with new persistence system
            success = self.save_model(symbol, interval, horizon, model, scaler_X, direction_accuracy/100)
            
            print(f"✅ LSTM training completed!")
            print(f"📊 Direction Accuracy: {direction_accuracy:.1f}%")
            print(f"📉 MAE: {mae:.6f}")
            if success:
                print(f"💾 Model persistently saved - no need to retrain!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training LSTM model: {e}")
            return False
    
    def build_ensemble_model(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        """Build ensemble of traditional ML models"""
        try:
            # Flatten X for traditional ML models
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            split_idx = int(len(X_flat) * 0.8)
            X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            models = {}
            
            # Random Forest
            print("🌲 Training Random Forest...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            models['random_forest'] = {
                'model': rf_model,
                'mse': mean_squared_error(y_test, rf_pred),
                'mae': mean_absolute_error(y_test, rf_pred),
                'direction_accuracy': np.mean((y_test > 0) == (rf_pred > 0)) * 100
            }
            
            # Gradient Boosting
            print("🚀 Training Gradient Boosting...")
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            models['gradient_boosting'] = {
                'model': gb_model,
                'mse': mean_squared_error(y_test, gb_pred),
                'mae': mean_absolute_error(y_test, gb_pred),
                'direction_accuracy': np.mean((y_test > 0) == (gb_pred > 0)) * 100
            }
            
            # Ensemble prediction (average)
            ensemble_pred = (rf_pred + gb_pred) / 2
            ensemble_accuracy = np.mean((y_test > 0) == (ensemble_pred > 0)) * 100
            
            models['ensemble'] = {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'direction_accuracy': ensemble_accuracy
            }
            
            print(f"🎯 Ensemble Models Trained:")
            for name, metrics in models.items():
                if 'direction_accuracy' in metrics:
                    print(f"  {name}: {metrics['direction_accuracy']:.1f}% direction accuracy")
            
            return models
            
        except Exception as e:
            print(f"❌ Error building ensemble models: {e}")
            return {}
    
    def train_all_models(self, symbol: str = "BTCUSDT", interval: str = "1h") -> Dict:
        """Train all models (LSTM + Ensemble) for all prediction horizons"""
        print(f"🚀 Training all models for {symbol} ({interval})")
        
        results = {}
        
        # Train LSTM models for each horizon
        for horizon in self.prediction_horizon:
            print(f"\n📈 Checking models for {horizon}h prediction horizon")
            
            # Check if LSTM model is already trained
            if self.is_model_trained(symbol, interval, horizon):
                print(f"✅ LSTM model for {horizon}h already trained - skipping!")
                results[f'lstm_{horizon}h'] = True
                
                # Still need to load ensemble models if available
                ensemble_key = f'ensemble_{horizon}h'
                if ensemble_key not in self.models:
                    print(f"🔍 Training ensemble model for {horizon}h...")
                    try:
                        df = self.fetch_training_data(symbol, interval, days=500)
                        df = self.engineer_features(df)
                        target_col = f'future_return_{horizon}h'
                        X, y = self.prepare_lstm_data(df, target_col)
                        
                        if len(X) > 0:
                            ensemble_models = self.build_ensemble_model(X, y, f"ensemble_{horizon}h")
                            self.models[ensemble_key] = ensemble_models
                            results[f'ensemble_{horizon}h'] = bool(ensemble_models)
                    except Exception as e:
                        print(f"❌ Error training ensemble for {horizon}h: {e}")
                        results[f'ensemble_{horizon}h'] = False
                else:
                    print(f"✅ Ensemble model for {horizon}h already available")
                    results[f'ensemble_{horizon}h'] = True
            else:
                print(f"🎯 Training new LSTM model for {horizon}h...")
                # LSTM training
                lstm_success = self.train_lstm_model(symbol, interval, horizon)
                results[f'lstm_{horizon}h'] = lstm_success
                
                # Ensemble training (using same data preparation)
                if lstm_success:
                    try:
                        df = self.fetch_training_data(symbol, interval, days=500)
                        df = self.engineer_features(df)
                        target_col = f'future_return_{horizon}h'
                        X, y = self.prepare_lstm_data(df, target_col)
                        
                        if len(X) > 0:
                            ensemble_models = self.build_ensemble_model(X, y, f"ensemble_{horizon}h")
                            self.models[f'ensemble_{horizon}h'] = ensemble_models
                            results[f'ensemble_{horizon}h'] = bool(ensemble_models)
                            
                    except Exception as e:
                        print(f"❌ Error training ensemble for {horizon}h: {e}")
                        results[f'ensemble_{horizon}h'] = False
                    results[f'ensemble_{horizon}h'] = False
        
        # Summary
        print(f"\n🎯 TRAINING SUMMARY:")
        print(f"=" * 50)
        for model_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{model_name}: {status}")
        
        return results
    
    def predict_with_ensemble(self, current_data: Dict, horizon: int = 1) -> Dict:
        """Make prediction using ensemble of models"""
        try:
            model_name_lstm = f"lstm_BTCUSDT_1h_{horizon}h"
            model_name_ensemble = f"ensemble_{horizon}h"
            
            predictions = {}
            confidences = {}
            
            # LSTM prediction
            if model_name_lstm in self.models and self.tf_available:
                # This would require current market data preparation
                # For now, return a placeholder
                predictions['lstm'] = 0.02  # 2% predicted return
                confidences['lstm'] = 0.75
            
            # Ensemble prediction
            if model_name_ensemble in self.models:
                predictions['ensemble'] = 0.015  # 1.5% predicted return
                confidences['ensemble'] = 0.70
            
            # Combined prediction
            if predictions:
                weights = [confidences[model] for model in predictions.keys()]
                weighted_pred = sum(pred * weight for pred, weight in zip(predictions.values(), weights)) / sum(weights)
                avg_confidence = sum(confidences.values()) / len(confidences)
                
                return {
                    'horizon_hours': horizon,
                    'predicted_return': weighted_pred,
                    'confidence': avg_confidence,
                    'direction': 'BUY' if weighted_pred > 0.005 else 'SELL' if weighted_pred < -0.005 else 'HOLD',
                    'individual_predictions': predictions,
                    'model_confidences': confidences
                }
            else:
                return {
                    'horizon_hours': horizon,
                    'predicted_return': 0,
                    'confidence': 0.5,
                    'direction': 'HOLD',
                    'error': 'No trained models available'
                }
                
        except Exception as e:
            print(f"❌ Error in ensemble prediction: {e}")
            return {
                'horizon_hours': horizon,
                'predicted_return': 0,
                'confidence': 0.5,
                'direction': 'HOLD',
                'error': str(e)
            }

if __name__ == "__main__":
    # Test the enhanced neural network
    engine = EnhancedNeuralEngine()
    
    print("🤖 ENHANCED NEURAL NETWORK TEST")
    print("=" * 50)
    
    # Train models (this will take several minutes)
    results = engine.train_all_models("BTCUSDT", "1h")
    
    print(f"\n🎯 MODEL ACCURACY SUMMARY:")
    for model_name, metrics in engine.model_accuracy.items():
        print(f"\n{model_name}:")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"  MAE: {metrics['mae']:.6f}")
    
    # Test predictions
    print(f"\n🔮 PREDICTION TEST:")
    for horizon in [1, 4, 24]:
        pred = engine.predict_with_ensemble({}, horizon)
        print(f"  {horizon}h: {pred['direction']} ({pred['confidence']:.1%} confidence)")
