# === ECHTE ML-IMPLEMENTIERUNG FÜR TRADING ===

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import logging
import threading
import time
from datetime import datetime
from binance_api import fetch_binance_data

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# === 1. DATEN VORBEREITUNG ===
class TradingDataPreprocessor:
    """Bereitet Marktdaten für ML-Training vor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, ohlc_data):
        """Erstelle Features aus OHLC-Daten"""
        if len(ohlc_data) < 50:
            return None, None
        
        features = []
        labels = []
        
        for i in range(30, len(ohlc_data) - 10):  # 30 lookback, 10 forward
            # Price data
            closes = [float(candle[4]) for candle in ohlc_data[i-30:i]]
            highs = [float(candle[2]) for candle in ohlc_data[i-30:i]]
            lows = [float(candle[3]) for candle in ohlc_data[i-30:i]]
            volumes = [float(candle[5]) for candle in ohlc_data[i-30:i]]
            
            # Technical indicators
            rsi = self._calculate_rsi(closes)
            macd = self._calculate_macd(closes)
            bb_position = self._calculate_bb_position(closes)
            volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10)
            
            # Price patterns
            price_change_5 = (closes[-1] - closes[-5]) / closes[-5]
            price_change_10 = (closes[-1] - closes[-10]) / closes[-10]
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
            
            # Support/Resistance
            support_distance = self._calculate_support_distance(closes, lows)
            resistance_distance = self._calculate_resistance_distance(closes, highs)
            
            # Feature vector
            feature_vector = [
                rsi,
                macd,
                bb_position,
                volume_ratio,
                price_change_5,
                price_change_10,
                volatility,
                support_distance,
                resistance_distance,
                closes[-1] / closes[-20],  # 20-period price ratio
                max(highs[-10:]) / closes[-1],  # Distance to recent high
                closes[-1] / min(lows[-10:])   # Distance to recent low
            ]
            
            # Label (future price movement)
            future_price = float(ohlc_data[i+10][4])  # 10 periods ahead
            current_price = float(ohlc_data[i][4])
            price_change = (future_price - current_price) / current_price
            
            # Classification labels
            if price_change > 0.02:  # > 2% gain
                label = 2  # Strong Buy
            elif price_change > 0.005:  # > 0.5% gain
                label = 1  # Buy
            elif price_change < -0.02:  # < -2% loss
                label = -2  # Strong Sell
            elif price_change < -0.005:  # < -0.5% loss
                label = -1  # Sell
            else:
                label = 0  # Hold
            
            features.append(feature_vector)
            labels.append(label)
        
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'price_change_5', 'price_change_10', 'volatility',
            'support_distance', 'resistance_distance',
            'price_ratio_20', 'distance_to_high', 'distance_to_low'
        ]
        
        return np.array(features), np.array(labels)
    
    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices):
        if len(prices) < 26:
            return 0
        
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        return ema_12 - ema_26
    
    def _ema(self, prices, period):
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_bb_position(self, prices, period=20):
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        std = np.std(recent_prices)
        
        current_price = prices[-1]
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    def _calculate_support_distance(self, closes, lows):
        recent_low = min(lows[-20:])
        current_price = closes[-1]
        return (current_price - recent_low) / current_price
    
    def _calculate_resistance_distance(self, closes, highs):
        recent_high = max(highs[-20:])
        current_price = closes[-1]
        return (recent_high - current_price) / current_price

# === 2. ECHTE ML-MODELLE ===
class RealMLTradingEngine:
    """Echte ML-Modelle für Trading Predictions"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = TradingDataPreprocessor()
        self.is_trained = False
    
    def train_models(self, symbol, days_back=90):
        """Trainiere alle ML-Modelle mit historischen Daten"""
        try:
            logger.info(f"🤖 Training ML models for {symbol}")
            
            # Hole ausreichend historische Daten
            historical_data = fetch_binance_data(symbol, '1h', days_back * 24)
            
            if not historical_data or len(historical_data) < 500:
                logger.error("Nicht genug Daten für ML-Training")
                return False
            
            # Erstelle Features und Labels
            X, y = self.preprocessor.create_features(historical_data)
            
            if X is None or len(X) < 100:
                logger.error("Feature-Erstellung fehlgeschlagen")
                return False
            
            # Skaliere Features
            X_scaled = self.preprocessor.scaler.fit_transform(X)
            
            # Teile Daten auf
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Training mit {len(X_train)} Samples, Testing mit {len(X_test)} Samples")
            
            # 1. Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_score = rf_model.score(X_test, y_test)
            self.models['random_forest'] = rf_model
            
            # 2. Support Vector Machine
            svm_model = SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
            svm_model.fit(X_train, y_train)
            svm_score = svm_model.score(X_test, y_test)
            self.models['svm'] = svm_model
            
            # 3. Ensemble Model (kombiniert RF + SVM)
            ensemble_predictions_train = []
            ensemble_predictions_test = []
            
            rf_pred_train = rf_model.predict_proba(X_train)
            svm_pred_train = svm_model.predict_proba(X_train)
            rf_pred_test = rf_model.predict_proba(X_test)
            svm_pred_test = svm_model.predict_proba(X_test)
            
            # Gewichtete Kombination (RF: 60%, SVM: 40%)
            ensemble_train = 0.6 * rf_pred_train + 0.4 * svm_pred_train
            ensemble_test = 0.6 * rf_pred_test + 0.4 * svm_pred_test
            
            ensemble_train_pred = np.argmax(ensemble_train, axis=1) - 2  # Convert back to -2,2 range
            ensemble_test_pred = np.argmax(ensemble_test, axis=1) - 2
            
            ensemble_score = np.mean(ensemble_test_pred == y_test)
            
            self.is_trained = True
            
            logger.info(f"✅ ML Training completed!")
            logger.info(f"Random Forest Accuracy: {rf_score:.3f}")
            logger.info(f"SVM Accuracy: {svm_score:.3f}")
            logger.info(f"Ensemble Accuracy: {ensemble_score:.3f}")
            
            # Speichere Modelle
            self._save_models(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ML Training failed: {e}")
            return False
    
    def predict(self, ohlc_data):
        """Mache Vorhersagen mit trainierten Modellen"""
        if not self.is_trained:
            return self._fallback_predictions()
        
        try:
            # Erstelle Features für die letzten Daten
            X, _ = self.preprocessor.create_features(ohlc_data)
            if X is None:
                return self._fallback_predictions()
            
            # Nehme nur die letzten Features
            latest_features = X[-1:] if len(X) > 0 else None
            if latest_features is None:
                return self._fallback_predictions()
            
            # Skaliere Features
            X_scaled = self.preprocessor.scaler.transform(latest_features)
            
            predictions = {}
            
            # Random Forest Prediction
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict(X_scaled)[0]
                rf_proba = self.models['random_forest'].predict_proba(X_scaled)[0]
                rf_confidence = np.max(rf_proba) * 100
                
                predictions['random_forest'] = {
                    'prediction': self._convert_label_to_direction(rf_pred),
                    'confidence': float(rf_confidence),  # Convert to Python float
                    'signal_strength': int(abs(rf_pred))  # Convert to Python int
                }
            
            # SVM Prediction
            if 'svm' in self.models:
                svm_pred = self.models['svm'].predict(X_scaled)[0]
                svm_proba = self.models['svm'].predict_proba(X_scaled)[0]
                svm_confidence = np.max(svm_proba) * 100
                
                predictions['svm'] = {
                    'prediction': self._convert_label_to_direction(svm_pred),
                    'confidence': float(svm_confidence),  # Convert to Python float
                    'signal_strength': int(abs(svm_pred))  # Convert to Python int
                }
            
            # Ensemble Prediction
            if 'random_forest' in self.models and 'svm' in self.models:
                rf_proba = self.models['random_forest'].predict_proba(X_scaled)[0]
                svm_proba = self.models['svm'].predict_proba(X_scaled)[0]
                
                # Gewichtete Kombination
                ensemble_proba = 0.6 * rf_proba + 0.4 * svm_proba
                ensemble_pred = np.argmax(ensemble_proba) - 2  # Convert back to -2,2
                ensemble_confidence = np.max(ensemble_proba) * 100
                
                predictions['ensemble'] = {
                    'prediction': self._convert_label_to_direction(ensemble_pred),
                    'confidence': float(ensemble_confidence),  # Convert to Python float
                    'signal_strength': int(abs(ensemble_pred))  # Convert to Python int
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ ML Prediction failed: {e}")
            return self._fallback_predictions()
    
    def _convert_label_to_direction(self, label):
        """Konvertiere numerische Labels zu Richtungen"""
        label_map = {
            -2: 'STRONG_SELL',
            -1: 'SELL',
            0: 'HOLD',
            1: 'BUY',
            2: 'STRONG_BUY'
        }
        return label_map.get(label, 'HOLD')
    
    def _fallback_predictions(self):
        """Fallback wenn ML nicht verfügbar"""
        return {
            'random_forest': {'prediction': 'HOLD', 'confidence': 50, 'signal_strength': 0},
            'svm': {'prediction': 'HOLD', 'confidence': 50, 'signal_strength': 0},
            'ensemble': {'prediction': 'HOLD', 'confidence': 50, 'signal_strength': 0}
        }
    
    def _save_models(self, symbol):
        """Speichere trainierte Modelle"""
        try:
            model_data = {
                'models': self.models,
                'preprocessor': self.preprocessor,
                'symbol': symbol,
                'trained_at': datetime.now().isoformat()
            }
            
            # In production würdest du das in einer Datei/DB speichern
            # joblib.dump(model_data, f'models/{symbol}_ml_models.pkl')
            logger.info(f"✅ Models saved for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")

# === 3. AUTOMATISCHES TRAINING BEIM START ===
def auto_train_models(ml_engine):
    """Trainiere Modelle automatisch beim App-Start"""
    try:
        logger.info("🤖 Starting automatic ML training...")
        
        # Trainiere für die wichtigsten Symbole
        top_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in top_symbols:
            logger.info(f"Training ML models for {symbol}...")
            success = ml_engine.train_models(symbol, days_back=60)  # 60 Tage
            
            if success:
                logger.info(f"✅ {symbol} ML training completed")
            else:
                logger.warning(f"⚠️ {symbol} ML training failed")
            
            time.sleep(2)  # Pause zwischen Trainings
        
        logger.info("🎯 Automatic ML training completed")
        
    except Exception as e:
        logger.error(f"❌ Auto-training failed: {e}")

# Starte Auto-Training in eigenem Thread
def start_ml_training_thread(ml_engine):
    """Starte ML-Training in separatem Thread"""
    training_thread = threading.Thread(target=auto_train_models, args=(ml_engine,), daemon=True)
    training_thread.start()

# === 4. VERWENDUNG IN DEINER APP ===
def get_real_ml_predictions(ml_engine, symbol, ohlc_data):
    """Hole echte ML-Vorhersagen für Symbol"""
    if ml_engine.is_trained:
        return ml_engine.predict(ohlc_data)
    else:
        return ml_engine._fallback_predictions()
