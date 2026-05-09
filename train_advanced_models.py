import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Bidirectional, Concatenate, Layer
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Custom Bahdanau Attention Layer
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden=None):
        if hidden is None:
            hidden = tf.reduce_mean(features, axis=1)
            
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

# Base Columns
BASE_FEATURES = [
    'altitude_ft', 'speed_knots', 'pitch_deg', 'vertical_speed', 'engine_thrust',
    'wind_speed_knots', 'crosswind_component', 'visibility_km', 'temperature_celsius',
    'humidity_percent', 'weather', 'time_of_day', 'runway_condition', 'pilot_error',
    'pilot_experience', 'aircraft_weight', 'flap_setting', 'engine_status', 'fuel_level'
]

# Engineered Columns
ENGINEERED_FEATURES = ['delta_vs', 'energy_state', 'glide_deviation']

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

def feature_engineering(df):
    # Delta Vertical Speed (Rate of descent change)
    df['delta_vs'] = df.groupby('flight_id')['vertical_speed'].diff().fillna(0)
    
    # Energy State Indicator (Simplified: Kinetic + Potential)
    # KE ~ v^2, PE ~ h
    df['energy_state'] = (df['speed_knots'] ** 2) * 0.5 + (df['altitude_ft'] * 32.2)
    
    # Glide Path Deviation (Simplified absolute deviation from a standard 3 degree slope)
    # Using altitude and speed as a proxy for approach stability
    df['glide_deviation'] = abs(df['altitude_ft'] - (df['speed_knots'] * 15)) # proxy
    
    return df

def encode_categorical(df):
    weather_map = {'Clear': 0, 'Rain': 1, 'Storm': 2, 'Fog': 3}
    time_map = {'Day': 0, 'Night': 1}
    runway_map = {'Dry': 0, 'Wet': 1, 'Icy': 2}
    engine_map = {'Normal': 1, 'Failur': 0}
    
    df['weather'] = df['weather'].replace(weather_map)
    df['time_of_day'] = df['time_of_day'].replace(time_map)
    df['runway_condition'] = df['runway_condition'].replace(runway_map)
    df['engine_status'] = df['engine_status'].replace(engine_map)
    
    return df

def map_severity(accident_type):
    if 'Safe' in accident_type:
        return 'Low'
    elif 'Hard' in accident_type or 'Incident' in accident_type:
        return 'Medium'
    else:
        return 'High'

def map_outcome(accident_type):
    if 'Safe' in accident_type:
        return 'Safe'
    elif 'Hard' in accident_type:
        return 'Minor Incident'
    else:
        return 'Hull Loss'

def prepare_multi_task_data(df, phase_filter, target_col):
    df = df[df['phase'] == phase_filter].copy()
    df = df.dropna(subset=BASE_FEATURES + [target_col])
    df = encode_categorical(df)
    df = feature_engineering(df)
    
    # Map Multi-task Targets
    df['Severity'] = df[target_col].apply(map_severity)
    df['Outcome'] = df[target_col].apply(map_outcome)
    
    # Encoders
    le_type = LabelEncoder()
    le_sev = LabelEncoder()
    le_out = LabelEncoder()
    
    df['y_type'] = le_type.fit_transform(df[target_col])
    df['y_sev'] = le_sev.fit_transform(df['Severity'])
    df['y_out'] = le_out.fit_transform(df['Outcome'])
    
    return df, le_type, le_sev, le_out

# 1. Load Data
df_app_raw = pd.read_csv('approch_dataset_v4.csv')
df_land_raw = pd.read_csv('landing_dataset_v4.csv')

df_app, le_type_app, le_sev_app, le_out_app = prepare_multi_task_data(df_app_raw, 'Approach', 'approach_accident_type')
df_land, le_type_land, le_sev_land, le_out_land = prepare_multi_task_data(df_land_raw, 'Landing', 'landing_accident_type')

# 2. Scale Features (Separately for each phase)
scaler_app = MinMaxScaler()
df_app.loc[:, ALL_FEATURES] = scaler_app.fit_transform(df_app[ALL_FEATURES])

scaler_land = MinMaxScaler()
df_land.loc[:, ALL_FEATURES] = scaler_land.fit_transform(df_land[ALL_FEATURES])

# Save Preprocessors
with open('scaler_app.pkl', 'wb') as f: pickle.dump(scaler_app, f)
with open('le_type_app.pkl', 'wb') as f: pickle.dump(le_type_app, f)
with open('le_sev_app.pkl', 'wb') as f: pickle.dump(le_sev_app, f)
with open('le_out_app.pkl', 'wb') as f: pickle.dump(le_out_app, f)

with open('scaler_land.pkl', 'wb') as f: pickle.dump(scaler_land, f)
with open('le_type_land.pkl', 'wb') as f: pickle.dump(le_type_land, f)
with open('le_sev_land.pkl', 'wb') as f: pickle.dump(le_sev_land, f)
with open('le_out_land.pkl', 'wb') as f: pickle.dump(le_out_land, f)

# 3. Create Sequences
def extract_sequences(df):
    sequences = {}
    targets = {}
    for flight_id, group in df.groupby('flight_id'):
        group = group.sort_values(by='timestep')
        seq = group[ALL_FEATURES].values
        if len(seq) == 4:
            sequences[flight_id] = seq
            # First row dictates the flight's overall outcome
            targets[flight_id] = {
                'type': group['y_type'].iloc[0],
                'sev': group['y_sev'].iloc[0],
                'out': group['y_out'].iloc[0]
            }
    return sequences, targets

app_seqs, app_targets = extract_sequences(df_app)
land_seqs, land_targets = extract_sequences(df_land)

# Intersect flight IDs
common_flights = list(set(app_seqs.keys()).intersection(set(land_seqs.keys())))
print(f"Total flights with full trajectories (Approach -> Landing): {len(common_flights)}")

X_app = np.array([app_seqs[fid] for fid in common_flights])
y_type_app = np.array([app_targets[fid]['type'] for fid in common_flights])
y_sev_app = np.array([app_targets[fid]['sev'] for fid in common_flights])
y_out_app = np.array([app_targets[fid]['out'] for fid in common_flights])

X_land = np.array([land_seqs[fid] for fid in common_flights])
y_type_land = np.array([land_targets[fid]['type'] for fid in common_flights])
y_sev_land = np.array([land_targets[fid]['sev'] for fid in common_flights])
y_out_land = np.array([land_targets[fid]['out'] for fid in common_flights])

# Train/Test Split
X_app_train, X_app_test, y_type_app_train, y_type_app_test, y_sev_app_train, y_sev_app_test, y_out_app_train, y_out_app_test, \
X_land_train, X_land_test, y_type_land_train, y_type_land_test, y_sev_land_train, y_sev_land_test, y_out_land_train, y_out_land_test = train_test_split(
    X_app, y_type_app, y_sev_app, y_out_app,
    X_land, y_type_land, y_sev_land, y_out_land,
    test_size=0.2, random_state=42
)

# 4. Build the Unified Multi-Task Context-Passing Architecture
seq_len = 4
num_features = len(ALL_FEATURES)

# --- STAGE 1: APPROACH MODEL ---
app_input = Input(shape=(seq_len, num_features), name='approach_input')
app_gru = Bidirectional(GRU(128, return_sequences=True))(app_input)
app_context = BahdanauAttention(64, name='approach_context')(app_gru)

app_dense = Dense(64, activation='relu')(app_context)
app_dense = Dropout(0.4)(app_dense)

# Approach Multi-Task Heads
out_app_type = Dense(len(le_type_app.classes_), activation='softmax', name='app_type_output')(app_dense)
out_app_sev = Dense(len(le_sev_app.classes_), activation='softmax', name='app_sev_output')(app_dense)
out_app_out = Dense(len(le_out_app.classes_), activation='softmax', name='app_out_output')(app_dense)

approach_model = Model(inputs=app_input, outputs=[out_app_type, out_app_sev, out_app_out, app_context], name='Approach_Risk_Model')

# --- STAGE 2: LANDING MODEL ---
land_input = Input(shape=(seq_len, num_features), name='landing_input')
land_gru = Bidirectional(GRU(128, return_sequences=True))(land_input)
land_attention = BahdanauAttention(64, name='landing_attention')(land_gru)

# Context Passing Bridge: Concatenate Approach Context with Landing Attention
# The landing model needs an input layer for the approach context when deployed standalone
app_context_input = Input(shape=(app_context.shape[1],), name='app_context_input')

land_combined = Concatenate()([land_attention, app_context_input])

land_dense = Dense(64, activation='relu')(land_combined)
land_dense = Dropout(0.4)(land_dense)

# Landing Multi-Task Heads
out_land_type = Dense(len(le_type_land.classes_), activation='softmax', name='land_type_output')(land_dense)
out_land_sev = Dense(len(le_sev_land.classes_), activation='softmax', name='land_sev_output')(land_dense)
out_land_out = Dense(len(le_out_land.classes_), activation='softmax', name='land_out_output')(land_dense)

landing_model = Model(inputs=[land_input, app_context_input], outputs=[out_land_type, out_land_sev, out_land_out], name='Landing_Risk_Model')

# --- UNIFIED TRAINING GRAPH ---
# Connect the Approach Context output into the Landing Model input
land_preds = landing_model([land_input, app_context])

unified_model = Model(
    inputs=[app_input, land_input], 
    outputs=[out_app_type, out_app_sev, out_app_out, land_preds[0], land_preds[1], land_preds[2]]
)

unified_model.compile(
    optimizer='adam',
    loss={
        'app_type_output': 'sparse_categorical_crossentropy',
        'app_sev_output': 'sparse_categorical_crossentropy',
        'app_out_output': 'sparse_categorical_crossentropy',
        'Landing_Risk_Model': 'sparse_categorical_crossentropy' # Handles all 3 landing heads if passed as list? Wait.
    },
    loss_weights=[1.0, 0.5, 0.5, 1.0, 0.5, 0.5]
)

# Keras multi-output loss format correction:
unified_model.compile(
    optimizer='adam',
    loss=['sparse_categorical_crossentropy'] * 6,
    loss_weights=[1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
    metrics=['accuracy']
)

print("Training Unified Architecture...")
unified_model.fit(
    [X_app_train, X_land_train],
    [y_type_app_train, y_sev_app_train, y_out_app_train, y_type_land_train, y_sev_land_train, y_out_land_train],
    validation_data=(
        [X_app_test, X_land_test],
        [y_type_app_test, y_sev_app_test, y_out_app_test, y_type_land_test, y_sev_land_test, y_out_land_test]
    ),
    epochs=12, batch_size=32, verbose=1
)

# Save the individual inference models!
# We can use the standalone approach and landing models for deployment.
approach_model.save('advanced_approach_model.keras')
landing_model.save('advanced_landing_model.keras')

print("Advanced Unified Models Trained and Saved Successfully!")

# --- EVALUATION METRICS ---
print("\n" + "="*50)
print("--- GENERATING EVALUATION METRICS ON TEST SET ---")
print("="*50 + "\n")

# Predict on test set
preds = unified_model.predict([X_app_test, X_land_test])

# The model outputs 6 arrays corresponding to:
# [app_type, app_sev, app_out, land_type, land_sev, land_out]
y_true_list = [y_type_app_test, y_sev_app_test, y_out_app_test, y_type_land_test, y_sev_land_test, y_out_land_test]
le_list = [le_type_app, le_sev_app, le_out_app, le_type_land, le_sev_land, le_out_land]
task_names = [
    "Approach Accident Type", 
    "Approach Severity Level", 
    "Approach Accident Outcome", 
    "Landing Accident Type", 
    "Landing Severity Level", 
    "Landing Accident Outcome"
]

accuracies = []

for i in range(6):
    print(f"--- Metrics for: {task_names[i]} ---")
    
    # Get highest probability index for each sample
    y_pred_indices = np.argmax(preds[i], axis=1)
    y_true_indices = y_true_list[i]
    
    # Generate classification report
    # We use le.classes_ to map the indices back to human-readable string names
    target_names = [str(cls) for cls in le_list[i].classes_]
    
    report = classification_report(y_true_indices, y_pred_indices, target_names=target_names, zero_division=0)
    acc = accuracy_score(y_true_indices, y_pred_indices)
    accuracies.append(acc)
    
    print(f"** {task_names[i]} Accuracy: {acc*100:.2f}% **\n")
    print(report)
    print("-" * 40 + "\n")

overall_system_accuracy = sum(accuracies) / len(accuracies)
print("="*50)
print(f"★ OVERALL SYSTEM PIPELINE ACCURACY: {overall_system_accuracy*100:.2f}% ★")
print("="*50 + "\n")

print("Evaluation Complete. You can now use these metrics for your IEEE paper!")
