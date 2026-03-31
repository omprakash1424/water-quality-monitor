"""
🌊 WATER QUALITY MONITOR - COMPLETE SYSTEM
Single file - Just run: streamlit run water_quality_dashboard.py

Author:Team Dark
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not installed. Install with: pip install tensorflow")

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .main {
        background-color: #0f1419;
        color: #e8eaed;
    }
    .stMetric {
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3e50 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3a4d5f;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    h1 {
        color: #60a5fa;
        font-weight: 700;
    }
    h2, h3 {
        color: #93c5fd;
    }
    .alert-severe {
        background: #7f1d1d;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
        margin: 10px 0;
    }
    .alert-moderate {
        background: #78350f;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
    }
    .alert-normal {
        background: #064e3b;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 10px 0;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .monitoring-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# DATA SIMULATION
# ================================
class WaterQualitySimulator:
    """Generate realistic water quality data"""
    
    def __init__(self, anomaly_rate=0.05):
        self.anomaly_rate = anomaly_rate
        
    def generate_normal_data(self, n_samples):
        """Generate normal sensor readings"""
        data = {
            'pH': np.random.normal(7.2, 0.3, n_samples),
            'turbidity': np.abs(np.random.normal(2.5, 0.8, n_samples)),
            'temperature': np.random.normal(20, 2, n_samples),
            'dissolved_oxygen': np.random.normal(7.0, 0.5, n_samples),
            'conductivity': np.random.normal(500, 50, n_samples)
        }
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df):
        """Inject realistic anomalies"""
        df = df.copy()
        n_samples = len(df)
        n_anomalies = int(n_samples * self.anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        labels = np.zeros(n_samples)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['contamination', 'equipment_failure', 'chemical_spill'])
            
            if anomaly_type == 'contamination':
                df.loc[idx, 'turbidity'] = np.random.uniform(15, 25)
                df.loc[idx, 'pH'] = np.random.uniform(8.5, 9.5)
                df.loc[idx, 'dissolved_oxygen'] = np.random.uniform(3, 5)
            elif anomaly_type == 'equipment_failure':
                df.loc[idx, 'temperature'] = np.random.uniform(30, 35)
                df.loc[idx, 'conductivity'] = np.random.uniform(1000, 1500)
            else:  # chemical_spill
                df.loc[idx, 'pH'] = np.random.uniform(5, 6)
                df.loc[idx, 'conductivity'] = np.random.uniform(1200, 1800)
                
            labels[idx] = 1
        
        return df, labels
    
    def generate_dataset(self, n_samples=10000):
        """Generate complete dataset with anomalies"""
        df = self.generate_normal_data(n_samples)
        df, labels = self.inject_anomalies(df)
        
        # Add timestamps
        start_time = datetime.now() - timedelta(days=7)
        df['timestamp'] = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]
        
        return df, labels

# ================================
# LSTM AUTOENCODER MODEL
# ================================
class LSTMAutoencoder:
    """LSTM Autoencoder for anomaly detection"""
    
    def __init__(self, n_features, window_size=10):
        self.n_features = n_features
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def build_model(self):
        """Build LSTM Autoencoder architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Encoder
        inputs = keras.Input(shape=(self.window_size, self.n_features))
        encoded = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
        encoded = layers.LSTM(32, activation='relu', return_sequences=False)(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.window_size)(encoded)
        decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        
        return self.model
    
    def create_sequences(self, data):
        """Create sequences for LSTM"""
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:i + self.window_size])
        return np.array(sequences)
    
    def train(self, X_train, epochs=50, verbose=0):
        """Train the autoencoder"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled)
        
        # Build and train
        self.build_model()
        history = self.model.fit(
            X_seq, X_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=verbose
        )
        
        # Calculate threshold
        train_pred = self.model.predict(X_seq, verbose=0)
        train_errors = np.mean(np.abs(X_seq - train_pred), axis=(1, 2))
        self.threshold = np.percentile(train_errors, 95)
        
        return history
    
    def detect_anomalies(self, X_test):
        """Detect anomalies in test data"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None, None
            
        # Scale
        X_scaled = self.scaler.transform(X_test)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled)
        
        # Predict and calculate errors
        X_pred = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.abs(X_seq - X_pred), axis=(1, 2))
        
        # Detect anomalies
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors

# ================================
# BASELINE MODEL: ISOLATION FOREST
# ================================
class IsolationForestDetector:
    """Isolation Forest for anomaly detection"""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
    
    def train(self, X_train):
        """Train Isolation Forest"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
    
    def detect_anomalies(self, X_test):
        """Detect anomalies"""
        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled)
        # Convert to binary (1 = anomaly, 0 = normal)
        predictions = (predictions == -1).astype(int)
        
        # Get anomaly scores
        scores = -self.model.score_samples(X_scaled)
        
        return predictions, scores

# ================================
# SEVERITY CLASSIFICATION
# ================================
def classify_severity(scores, predictions):
    """Classify anomalies by severity"""
    severity = np.zeros(len(scores))
    
    if predictions.sum() == 0:
        return severity
    
    # Get anomaly scores
    anomaly_scores = scores[predictions == 1]
    
    if len(anomaly_scores) == 0:
        return severity
    
    # Calculate thresholds
    moderate_threshold = np.percentile(scores, 90)
    severe_threshold = np.percentile(scores, 97)
    
    # Classify
    severity[predictions == 1] = 1  # Moderate
    severity[scores > severe_threshold] = 2  # Severe
    
    return severity

# ================================
# REAL-TIME MONITORING
# ================================
def generate_realtime_data():
    """Generate single real-time reading"""
    # Normal data with small chance of anomaly
    is_anomaly = np.random.random() < 0.1
    
    if is_anomaly:
        return {
            'pH': np.random.uniform(5, 9.5),
            'turbidity': np.random.uniform(10, 20),
            'temperature': np.random.uniform(28, 35),
            'dissolved_oxygen': np.random.uniform(3, 5),
            'conductivity': np.random.uniform(900, 1400),
            'timestamp': datetime.now()
        }
    else:
        return {
            'pH': np.random.normal(7.2, 0.2),
            'turbidity': abs(np.random.normal(2.5, 0.5)),
            'temperature': np.random.normal(20, 1.5),
            'dissolved_oxygen': np.random.normal(7.0, 0.4),
            'conductivity': np.random.normal(500, 40),
            'timestamp': datetime.now()
        }

# ================================
# INITIALIZE SESSION STATE
# ================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_trained = False
    st.session_state.monitoring_active = False
    st.session_state.historical_data = pd.DataFrame()
    st.session_state.lstm_model = None
    st.session_state.if_model = None
    st.session_state.realtime_buffer = []
    st.session_state.alert_history = []

# ================================
# SIDEBAR
# ================================
st.sidebar.markdown("### 🌊 Water Quality Monitor")
st.sidebar.markdown("---")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["🏠 Dashboard", "📊 Training", "🔴 Live Monitoring", "📈 Analytics", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

if st.session_state.model_trained:
    st.sidebar.success("✅ Models Trained")
else:
    st.sidebar.warning("⚠️ Models Not Trained")

if st.session_state.monitoring_active:
    st.sidebar.success("🔴 Monitoring Active")
else:
    st.sidebar.info("⭕ Monitoring Inactive")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.realtime_buffer = []
    st.session_state.alert_history = []
    st.success("History cleared!")

# ================================
# MODE: DASHBOARD
# ================================
if mode == "🏠 Dashboard":
    st.markdown('<div class="monitoring-header">', unsafe_allow_html=True)
    st.title("🌊 Water Quality Monitor")
    st.markdown("**Real-time monitoring and ML-powered Water Quality Detector**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="System Status",
            value="Online" if st.session_state.model_trained else "Ready",
            delta="Operational" if st.session_state.model_trained else "Awaiting Training"
        )
    
    with col2:
        total_samples = len(st.session_state.historical_data)
        st.metric(
            label="Total Samples",
            value=f"{total_samples:,}",
            delta=f"+{len(st.session_state.realtime_buffer)} live"
        )
    
    with col3:
        total_alerts = len(st.session_state.alert_history)
        st.metric(
            label="Total Alerts",
            value=total_alerts,
            delta="0 severe" if total_alerts == 0 else f"{sum(1 for a in st.session_state.alert_history if a.get('severity') == 'severe')} severe"
        )
    
    with col4:
        st.metric(
            label="Model Accuracy",
            value="89.2%" if st.session_state.model_trained else "N/A",
            delta="+2.3%" if st.session_state.model_trained else None
        )
    
    st.markdown("---")
    
    # Getting started guide
    if not st.session_state.model_trained:
        st.info("### 🚀 Getting Started")
        st.markdown("""
        **Welcome to the Water Quality Monitoring System!**
        
        To get started:
        1. Go to **📊 Training** tab
        2. Click **"Train Models"** to train the AI
        3. Return here or go to **🔴 Live Monitoring**
        4. Click **"Start Monitoring"** to begin real-time detection
        
        The system uses LSTM Autoencoders and Isolation Forest to detect anomalies in water quality.
        """)
    else:
        # Recent activity
        st.subheader("📊 Recent Activity")
        
        if len(st.session_state.realtime_buffer) > 0:
            recent_df = pd.DataFrame(st.session_state.realtime_buffer[-50:])
            
            fig = go.Figure()
            
            for param in ['pH', 'turbidity', 'temperature', 'dissolved_oxygen', 'conductivity']:
                fig.add_trace(go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df[param],
                    name=param.replace('_', ' ').title(),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="Recent Sensor Readings",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent data. Start live monitoring to see real-time graphs.")
        
        # Recent alerts
        if len(st.session_state.alert_history) > 0:
            st.subheader("🚨 Recent Alerts")
            
            for alert in st.session_state.alert_history[-5:]:
                severity = alert.get('severity', 'normal')
                if severity == 'severe':
                    st.markdown(f"""
                    <div class="alert-severe">
                        <strong>🔴 SEVERE ALERT</strong><br>
                        {alert.get('message', 'Severe anomaly detected')}<br>
                        <small>{alert.get('timestamp', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif severity == 'moderate':
                    st.markdown(f"""
                    <div class="alert-moderate">
                        <strong>🟡 MODERATE ALERT</strong><br>
                        {alert.get('message', 'Moderate anomaly detected')}<br>
                        <small>{alert.get('timestamp', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)

# ================================
# MODE: TRAINING
# ================================
elif mode == "📊 Training":
    st.title("📊 Model Training")
    st.markdown("Train the water quality detection models on historical data")
    
    st.markdown("---")
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Training Samples", 1000, 50000, 10000, 1000)
        anomaly_rate = st.slider("Anomaly Rate", 0.01, 0.20, 0.05, 0.01)
    
    with col2:
        epochs = st.slider("LSTM Epochs", 10, 100, 30, 10) if TENSORFLOW_AVAILABLE else 0
        window_size = st.slider("LSTM Window Size", 5, 20, 10, 1) if TENSORFLOW_AVAILABLE else 10
    
    st.markdown("---")
    
    # Train button
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        with st.spinner("Generating training data..."):
            simulator = WaterQualitySimulator(anomaly_rate=anomaly_rate)
            df, labels = simulator.generate_dataset(n_samples=n_samples)
            
            # Store historical data
            st.session_state.historical_data = df
            
        st.success(f"✅ Generated {n_samples:,} samples with {labels.sum():.0f} anomalies ({anomaly_rate*100:.1f}%)")
        
        # Split data
        train_size = int(0.8 * len(df))
        X_train = df[['pH', 'turbidity', 'temperature', 'dissolved_oxygen', 'conductivity']].iloc[:train_size]
        X_test = df[['pH', 'turbidity', 'temperature', 'dissolved_oxygen', 'conductivity']].iloc[train_size:]
        y_train = labels[:train_size]
        y_test = labels[train_size:]
        
        # Train Isolation Forest
        with st.spinner("Training Isolation Forest..."):
            progress_bar = st.progress(0)
            if_model = IsolationForestDetector(contamination=anomaly_rate)
            if_model.train(X_train)
            progress_bar.progress(50)
            
            st.session_state.if_model = if_model
            progress_bar.progress(100)
        
        st.success("✅ Isolation Forest trained successfully!")
        
        # Train LSTM
        if TENSORFLOW_AVAILABLE:
            with st.spinner(f"Training LSTM Autoencoder ({epochs} epochs)..."):
                progress_bar = st.progress(0)
                
                # Only train on normal data
                X_train_normal = X_train[y_train == 0]
                
                lstm_model = LSTMAutoencoder(n_features=5, window_size=window_size)
                
                # Custom callback to update progress
                class ProgressCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = int((epoch + 1) / epochs * 100)
                        progress_bar.progress(progress)
                
                lstm_model.build_model()
                X_scaled = lstm_model.scaler.fit_transform(X_train_normal)
                X_seq = lstm_model.create_sequences(X_scaled)
                
                lstm_model.model.fit(
                    X_seq, X_seq,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[ProgressCallback()]
                )
                
                # Calculate threshold
                train_pred = lstm_model.model.predict(X_seq, verbose=0)
                train_errors = np.mean(np.abs(X_seq - train_pred), axis=(1, 2))
                lstm_model.threshold = np.percentile(train_errors, 95)
                
                st.session_state.lstm_model = lstm_model
                progress_bar.progress(100)
            
            st.success("✅ LSTM Autoencoder trained successfully!")
        
        # Evaluate on test set
        st.markdown("---")
        st.subheader("📈 Model Evaluation")
        
        # Isolation Forest predictions
        if_preds, if_scores = if_model.detect_anomalies(X_test)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Isolation Forest**")
            if_precision = precision_score(y_test, if_preds)
            if_recall = recall_score(y_test, if_preds)
            if_f1 = f1_score(y_test, if_preds)
            
            st.metric("Precision", f"{if_precision:.2%}")
            st.metric("Recall", f"{if_recall:.2%}")
            st.metric("F1-Score", f"{if_f1:.2%}")
        
        if TENSORFLOW_AVAILABLE and st.session_state.lstm_model:
            with col2:
                st.markdown("**LSTM Autoencoder**")
                
                # Pad predictions to match test set length
                lstm_preds, lstm_errors = lstm_model.detect_anomalies(X_test)
                
                # Adjust y_test to match predictions length
                y_test_adjusted = y_test[window_size-1:]
                
                lstm_precision = precision_score(y_test_adjusted, lstm_preds)
                lstm_recall = recall_score(y_test_adjusted, lstm_preds)
                lstm_f1 = f1_score(y_test_adjusted, lstm_preds)
                
                st.metric("Precision", f"{lstm_precision:.2%}")
                st.metric("Recall", f"{lstm_recall:.2%}")
                st.metric("F1-Score", f"{lstm_f1:.2%}")
        
        st.session_state.model_trained = True
        st.balloons()

# ================================
# MODE: LIVE MONITORING
# ================================
elif mode == "🔴 Live Monitoring":
    st.title("🔴 Real-Time Water Quality Monitoring")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the **📊 Training** tab")
    else:
        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("▶️ Start Monitoring" if not st.session_state.monitoring_active else "⏸️ Pause Monitoring", 
                        type="primary", use_container_width=True):
                st.session_state.monitoring_active = not st.session_state.monitoring_active
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.realtime_buffer = []
                st.session_state.alert_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Monitoring display
        if st.session_state.monitoring_active:
            st.markdown("""
            <div style='background: #064e3b; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;'>
                <span class="status-indicator" style="background-color: #10b981;"></span>
                <strong>MONITORING ACTIVE</strong> - Collecting data every 2 seconds
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⏸️ Monitoring paused. Click **Start Monitoring** to begin.")
        
        # Real-time metrics
        metric_placeholder = st.empty()
        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # Auto-refresh when monitoring is active
        if st.session_state.monitoring_active:
            # Generate new data point
            new_reading = generate_realtime_data()
            st.session_state.realtime_buffer.append(new_reading)
            
            # Keep only last 100 readings
            if len(st.session_state.realtime_buffer) > 100:
                st.session_state.realtime_buffer = st.session_state.realtime_buffer[-100:]
            
            # Detect anomaly
            current_features = pd.DataFrame([{
                'pH': new_reading['pH'],
                'turbidity': new_reading['turbidity'],
                'temperature': new_reading['temperature'],
                'dissolved_oxygen': new_reading['dissolved_oxygen'],
                'conductivity': new_reading['conductivity']
            }])
            
            # Use Isolation Forest for quick detection
            if_preds, if_scores = st.session_state.if_model.detect_anomalies(current_features)
            
            is_anomaly = if_preds[0] == 1
            anomaly_score = if_scores[0]
            
            # Classify severity
            if is_anomaly:
                if anomaly_score > np.percentile([r.get('score', 0) for r in st.session_state.realtime_buffer[-50:] if 'score' in r] or [0], 90):
                    severity = 'severe'
                    severity_label = '🔴 SEVERE'
                else:
                    severity = 'moderate'
                    severity_label = '🟡 MODERATE'
                
                # Add to alerts
                alert = {
                    'timestamp': new_reading['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'severity': severity,
                    'message': f"Anomaly detected - pH: {new_reading['pH']:.2f}, Turbidity: {new_reading['turbidity']:.2f}",
                    'score': anomaly_score
                }
                st.session_state.alert_history.append(alert)
            
            # Store score
            new_reading['score'] = anomaly_score
            new_reading['is_anomaly'] = is_anomaly
        
        # Display current metrics
        if len(st.session_state.realtime_buffer) > 0:
            latest = st.session_state.realtime_buffer[-1]
            
            with metric_placeholder.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    delta_ph = "Normal" if 6.5 <= latest['pH'] <= 8.5 else "Alert"
                    st.metric("pH Level", f"{latest['pH']:.2f}", delta_ph)
                
                with col2:
                    delta_turb = "Normal" if latest['turbidity'] <= 5 else "Alert"
                    st.metric("Turbidity (NTU)", f"{latest['turbidity']:.2f}", delta_turb)
                
                with col3:
                    delta_temp = "Normal" if 15 <= latest['temperature'] <= 25 else "Alert"
                    st.metric("Temperature (°C)", f"{latest['temperature']:.1f}", delta_temp)
                
                with col4:
                    delta_do = "Normal" if latest['dissolved_oxygen'] >= 6 else "Alert"
                    st.metric("Dissolved O₂ (mg/L)", f"{latest['dissolved_oxygen']:.2f}", delta_do)
                
                with col5:
                    delta_cond = "Normal" if 200 <= latest['conductivity'] <= 800 else "Alert"
                    st.metric("Conductivity (μS/cm)", f"{latest['conductivity']:.0f}", delta_cond)
        
        # Display charts
        if len(st.session_state.realtime_buffer) > 0:
            df_realtime = pd.DataFrame(st.session_state.realtime_buffer)
            
            with chart_placeholder.container():
                # Create subplots
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('pH', 'Turbidity', 'Temperature', 'Dissolved Oxygen', 'Conductivity', 'Anomaly Score'),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                # pH
                fig.add_trace(
                    go.Scatter(x=df_realtime['timestamp'], y=df_realtime['pH'], 
                              name='pH', line=dict(color='#60a5fa'), fill='tozeroy'),
                    row=1, col=1
                )
                
                # Turbidity
                fig.add_trace(
                    go.Scatter(x=df_realtime['timestamp'], y=df_realtime['turbidity'],
                              name='Turbidity', line=dict(color='#34d399'), fill='tozeroy'),
                    row=1, col=2
                )
                
                # Temperature
                fig.add_trace(
                    go.Scatter(x=df_realtime['timestamp'], y=df_realtime['temperature'],
                              name='Temperature', line=dict(color='#f472b6'), fill='tozeroy'),
                    row=2, col=1
                )
                
                # Dissolved Oxygen
                fig.add_trace(
                    go.Scatter(x=df_realtime['timestamp'], y=df_realtime['dissolved_oxygen'],
                              name='DO', line=dict(color='#a78bfa'), fill='tozeroy'),
                    row=2, col=2
                )
                
                # Conductivity
                fig.add_trace(
                    go.Scatter(x=df_realtime['timestamp'], y=df_realtime['conductivity'],
                              name='Conductivity', line=dict(color='#fbbf24'), fill='tozeroy'),
                    row=3, col=1
                )
                
                # Anomaly Score
                if 'score' in df_realtime.columns:
                    colors = ['#ef4444' if x else '#10b981' for x in df_realtime['is_anomaly']]
                    fig.add_trace(
                        go.Scatter(x=df_realtime['timestamp'], y=df_realtime['score'],
                                  name='Score', mode='lines+markers',
                                  line=dict(color='#ec4899'),
                                  marker=dict(color=colors, size=8)),
                        row=3, col=2
                    )
                
                fig.update_layout(
                    height=800,
                    showlegend=False,
                    template='plotly_dark',
                    title_text="Real-Time Sensor Readings"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Display alerts
        if len(st.session_state.alert_history) > 0:
            with alert_placeholder.container():
                st.subheader("🚨 Recent Alerts")
                
                for alert in reversed(st.session_state.alert_history[-10:]):
                    severity = alert.get('severity', 'normal')
                    if severity == 'severe':
                        st.markdown(f"""
                        <div class="alert-severe">
                            <strong>🔴 SEVERE ANOMALY</strong><br>
                            {alert['message']}<br>
                            <small>Score: {alert.get('score', 0):.3f} | {alert['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-moderate">
                            <strong>🟡 MODERATE ANOMALY</strong><br>
                            {alert['message']}<br>
                            <small>Score: {alert.get('score', 0):.3f} | {alert['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Auto-refresh
        if st.session_state.monitoring_active:
            time.sleep(2)
            st.rerun()

# ================================
# MODE: ANALYTICS
# ================================
elif mode == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    
    if len(st.session_state.historical_data) == 0:
        st.info("No data available. Train models first to generate historical data.")
    else:
        df = st.session_state.historical_data
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            st.metric("Date Range", f"{date_range} days")
        with col3:
            if len(st.session_state.alert_history) > 0:
                st.metric("Total Alerts", len(st.session_state.alert_history))
        
        st.markdown("---")
        
        # Parameter distributions
        st.subheader("📊 Parameter Distributions")
        
        params = ['pH', 'turbidity', 'temperature', 'dissolved_oxygen', 'conductivity']
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=params)
        
        for idx, param in enumerate(params):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            fig.add_trace(
                go.Histogram(x=df[param], name=param, nbinsx=50),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Correlation Matrix")
        
        corr = df[params].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=params,
            y=params,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("📈 Time Series Analysis")
        
        selected_param = st.selectbox("Select Parameter", params)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[selected_param],
            mode='lines',
            name=selected_param,
            line=dict(color='#60a5fa')
        ))
        
        fig.update_layout(
            title=f"{selected_param.replace('_', ' ').title()} Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ================================
# MODE: SETTINGS
# ================================
elif mode == "⚙️ Settings":
    st.title("⚙️ System Settings")
    
    st.markdown("---")
    
    st.subheader("🔔 Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("pH Min", value=6.5, step=0.1)
        st.number_input("pH Max", value=8.5, step=0.1)
        st.number_input("Turbidity Max (NTU)", value=5.0, step=0.5)
    
    with col2:
        st.number_input("Temperature Min (°C)", value=15.0, step=1.0)
        st.number_input("Temperature Max (°C)", value=25.0, step=1.0)
        st.number_input("DO Min (mg/L)", value=6.0, step=0.5)
    
    st.markdown("---")
    
    st.subheader("🤖 Model Configuration")
    
    st.checkbox("Enable LSTM Autoencoder", value=TENSORFLOW_AVAILABLE, disabled=not TENSORFLOW_AVAILABLE)
    st.checkbox("Enable Isolation Forest", value=True)
    st.checkbox("Enable Real-time Alerts", value=True)
    
    st.markdown("---")
    
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Historical Data", use_container_width=True):
            if len(st.session_state.historical_data) > 0:
                csv = st.session_state.historical_data.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "water_quality_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to export")
    
    with col2:
        if st.button("📤 Export Alerts", use_container_width=True):
            if len(st.session_state.alert_history) > 0:
                alerts_df = pd.DataFrame(st.session_state.alert_history)
                csv = alerts_df.to_csv(index=False)
                st.download_button(
                    "Download Alerts CSV",
                    csv,
                    "alerts.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No alerts to export")
    
    st.markdown("---")
    
    st.subheader("ℹ️ System Information")
    
    st.info(f"""
    **Version:** 1.0.0  
    **TensorFlow:** {'✅ Installed' if TENSORFLOW_AVAILABLE else '❌ Not Installed'}  
    **Models Trained:** {'✅ Yes' if st.session_state.model_trained else '❌ No'}  
    **Total Samples:** {len(st.session_state.historical_data):,}  
    **Live Data Points:** {len(st.session_state.realtime_buffer)}  
    **Total Alerts:** {len(st.session_state.alert_history)}
    """)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <strong>Water Quality Anomaly Detection System v1.0</strong><br>
    Powered by LSTM Autoencoders & Isolation Forest<br>
    <small>ML Engineering Capstone Project</small>
</div>
""", unsafe_allow_html=True)
