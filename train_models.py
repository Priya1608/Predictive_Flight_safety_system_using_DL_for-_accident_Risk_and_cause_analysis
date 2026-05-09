import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
import pickle

# Define the exact columns that app.py uses during inference.
FEATURE_COLUMNS = [
    'altitude_ft', 'speed_knots', 'pitch_deg', 'vertical_speed', 'engine_thrust',
    'wind_speed_knots', 'crosswind_component', 'visibility_km', 'temperature_celsius',
    'humidity_percent', 'weather', 'time_of_day', 'runway_condition', 'pilot_error',
    'pilot_experience', 'aircraft_weight', 'flap_setting', 'engine_status', 'fuel_level'
]

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

def train_phase_model(csv_path, phase_filter, target_column, model_save_name, scaler_save_name, le_save_name):
    print(f"--- Starting Training for {model_save_name} ---")
    
    # 1. Load the specific phase dataset
    df = pd.read_csv(csv_path)
    
    # Ensure there are no missing values in features and target
    df = df.dropna(subset=FEATURE_COLUMNS + [target_column, 'flight_id', 'timestep', 'phase'])
    # Encode categoricals before they reach MinMaxScaler
    df = encode_categorical(df)
    # Ensure they are numeric
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    
    # Filter by specific phase (Approach or Landing)
    df = df[df['phase'] == phase_filter]
    
    # Sort by flight_id and timestep to guarantee order
    df = df.sort_values(by=['flight_id', 'timestep'])
    
    # 2. Fit specific scaler and encoder
    scaler = MinMaxScaler()
    df.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    
    le = LabelEncoder()
    le.fit(df[target_column])
    
    # Save the scaler and encoder so app.py can load them!
    with open(scaler_save_name, 'wb') as f: 
        pickle.dump(scaler, f)
    with open(le_save_name, 'wb') as f: 
        pickle.dump(le, f)
        
    print(f"Saved {scaler_save_name} and {le_save_name}")
    
    # 3. Create Sequences grouped by flight_id
    sequences = []
    labels = []
    
    # Group by flight to ensure we don't leak frames between train/test
    # and to build sequences of timesteps
    for flight_id, flight_data in df.groupby('flight_id'):
        seq = flight_data[FEATURE_COLUMNS].values
        # Only accept complete sequences of length 4
        if len(seq) == 4:
            sequences.append(seq)
            # The label is static for the given phase of the flight
            label = flight_data[target_column].iloc[0]
            labels.append(label)
    
    X_reshaped = np.array(sequences)
    y_encoded = le.transform(labels)
    
    print(f"Extracted {len(X_reshaped)} flight sequences of shape {X_reshaped.shape[1:]}")
    
    # 4. Train/Test Split by flight sequence (preventing data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)
    
    # 5. Build and Train GRU
    num_classes = len(np.unique(y_encoded))
    
    model = Sequential()
    # Now GRU properly receives 3D data: (timesteps, features) -> (4, 19)
    model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print("Training GRU Neural Network...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # 6. Save Model
    model.save(model_save_name)
    print(f"Saved {model_save_name} successfully!\n")

if __name__ == '__main__':
    # 1. Train the Approach Model
    train_phase_model(
        csv_path='approch_dataset_v3.csv', 
        phase_filter='Approach',
        target_column='approach_accident_type', 
        model_save_name='approach_model.keras', 
        scaler_save_name='scaler_app.pkl', 
        le_save_name='le_app.pkl'
    )
    
    # 2. Train the Landing Model
    train_phase_model(
        csv_path='landing_dataset_v3.csv', 
        phase_filter='Landing',
        target_column='landing_accident_type', 
        model_save_name='landing_model.keras', 
        scaler_save_name='scaler_land.pkl', 
        le_save_name='le_land.pkl'
    )
    
    print("ALL DONE! Both models and preprocessors have been successfully generated.")
