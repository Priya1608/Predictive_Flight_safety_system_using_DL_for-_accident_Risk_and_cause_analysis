import os
os.environ['KERAS_BACKEND'] = 'torch'
from flask import Flask, request, render_template

app = Flask(__name__)

# Try loading Keras models gracefully
ADVANCED_ML_AVAILABLE = False
ML_AVAILABLE = False

try:
    import pickle
    import numpy as np
    import keras
    from keras.models import load_model
    from keras.layers import Layer, Dense
    import keras.ops as ops
    
    @keras.saving.register_keras_serializable()
    class BahdanauAttention(Layer):
        def __init__(self, units, **kwargs):
            super(BahdanauAttention, self).__init__(**kwargs)
            self.units = units
            self.W1 = Dense(units)
            self.W2 = Dense(units)
            self.V = Dense(1)

        def call(self, features, hidden=None):
            if hidden is None:
                hidden = ops.mean(features, axis=1)
            hidden_with_time_axis = ops.expand_dims(hidden, 1)
            score = self.V(ops.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
            attention_weights = ops.softmax(score, axis=1)
            context_vector = attention_weights * features
            context_vector = ops.sum(context_vector, axis=1)
            return context_vector

        def get_config(self):
            config = super(BahdanauAttention, self).get_config()
            config.update({'units': self.units})
            return config

    # Load original scalers
    with open('scaler_app.pkl', 'rb') as f: scaler_app = pickle.load(f)
    with open('scaler_land.pkl', 'rb') as f: scaler_land = pickle.load(f)
    with open('le_app.pkl', 'rb') as f: le_app = pickle.load(f)
    with open('le_land.pkl', 'rb') as f: le_land = pickle.load(f)
    
    # Try advanced models first
    if os.path.exists('advanced_approach_model.keras') and os.path.exists('advanced_landing_model.keras'):
        app_model = load_model('advanced_approach_model.keras')
        land_model = load_model('advanced_landing_model.keras')
        with open('le_type_app.pkl', 'rb') as f: le_type_app = pickle.load(f)
        with open('le_sev_app.pkl', 'rb') as f: le_sev_app = pickle.load(f)
        with open('le_out_app.pkl', 'rb') as f: le_out_app = pickle.load(f)
        with open('le_type_land.pkl', 'rb') as f: le_type_land = pickle.load(f)
        with open('le_sev_land.pkl', 'rb') as f: le_sev_land = pickle.load(f)
        with open('le_out_land.pkl', 'rb') as f: le_out_land = pickle.load(f)
        with open('ascaler_app.pkl', 'rb') as f: scaler_app = pickle.load(f)
        with open('ascaler_land.pkl', 'rb') as f: scaler_land = pickle.load(f)
        ADVANCED_ML_AVAILABLE = True
    elif os.path.exists('approach_model.keras') and os.path.exists('landing_model.keras'):
        app_model = load_model('approach_model.keras')
        land_model = load_model('landing_model.keras')
    ML_AVAILABLE = True
except Exception as e:
    print("INFO: Could not load TF/Keras models or dependencies. Using fallback logic. Reason:", e)

FEATURE_COLUMNS = [
    'altitude_ft', 'speed_knots', 'pitch_deg', 'vertical_speed', 'engine_thrust',
    'wind_speed_knots', 'crosswind_component', 'visibility_km', 'temperature_celsius',
    'humidity_percent', 'weather', 'time_of_day', 'runway_condition', 'pilot_error',
    'pilot_experience', 'aircraft_weight', 'flap_setting', 'engine_status', 'fuel_level'
]

def extrapolate_history(single_frame, phase):
    sequence = []
    for i in range(3, -1, -1):
        frame = single_frame.copy()
        alt_modifier = 600 if phase == 'approach' else 250
        speed_modifier = 10 if phase == 'approach' else 5
        frame[0] = max(0, frame[0] + (i * alt_modifier))
        frame[1] = frame[1] + (i * speed_modifier)
        frame[3] = frame[3] * (1.0 - (i * 0.05))
        sequence.append(frame)
    import numpy as np
    return np.array(sequence)

def process_flight_features(features, phase):
    alt, speed, pitch, vs, thrust, wind, cross, vis, temp, hum, weather, time_val, runway, error, exp, weight, flap, engine, fuel = features
    prediction_label = "Safe Approach" if phase == 'approach' else "Safe Landing"
    severity = "Low"
    description = ""
    causes_text = ""
    causes_points = []
    prevention_text = ""
    prevention_points = []
    exact_factors = []
    outcome = "Safe"
    
    if ML_AVAILABLE:
        try:
            import numpy as np
            sequence_arr = extrapolate_history(features, phase)
            
            if ADVANCED_ML_AVAILABLE:
                adv_sequence = []
                for idx, row in enumerate(sequence_arr):
                    _alt, spd, ptch, _vs = row[0], row[1], row[2], row[3]
                    delta_vs = _vs - sequence_arr[idx-1][3] if idx > 0 else 0
                    energy_state = (spd ** 2) * 0.5 + (_alt * 32.2)
                    glide_dev = abs(_alt - (spd * 15))
                    adv_sequence.append(list(row) + [delta_vs, energy_state, glide_dev])
                adv_sequence = np.array(adv_sequence)
                
                if phase == 'approach':
                    feat_scaled = scaler_app.transform(adv_sequence)
                    feat_scaled = np.expand_dims(feat_scaled, axis=0)
                    preds = app_model.predict(feat_scaled)
                    prediction_label = le_type_app.inverse_transform([np.argmax(preds[0])])[0]
                    severity = le_sev_app.inverse_transform([np.argmax(preds[1])])[0]
                    outcome = le_out_app.inverse_transform([np.argmax(preds[2])])[0]
                else:
                    feat_scaled = scaler_land.transform(adv_sequence)
                    feat_scaled = np.expand_dims(feat_scaled, axis=0)
                    dummy_app = np.zeros((1, 4, 22))
                    app_context = app_model.predict(dummy_app)[3]
                    preds = land_model.predict([feat_scaled, app_context])
                    prediction_label = le_type_land.inverse_transform([np.argmax(preds[0])])[0]
                    severity = le_sev_land.inverse_transform([np.argmax(preds[1])])[0]
                    outcome = le_out_land.inverse_transform([np.argmax(preds[2])])[0]
            else:
                if phase == 'approach':
                    feat_scaled = scaler_app.transform(sequence_arr)
                    feat_scaled = np.expand_dims(feat_scaled, axis=0) 
                    pred_probs = app_model.predict(feat_scaled)
                    pred_idx = np.argmax(pred_probs, axis=1)[0]
                    prediction_label = le_app.inverse_transform([pred_idx])[0]
                else:
                    feat_scaled = scaler_land.transform(sequence_arr)
                    feat_scaled = np.expand_dims(feat_scaled, axis=0)
                    pred_probs = land_model.predict(feat_scaled)
                    pred_idx = np.argmax(pred_probs, axis=1)[0]
                    prediction_label = le_land.inverse_transform([pred_idx])[0]
                
            if engine == 0 or thrust < 20:
                prediction_label = "Engine Failure"
                if not ADVANCED_ML_AVAILABLE: severity = "Critical"
                if engine == 0 and "Engine Status is failed (0)." not in exact_factors:
                    exact_factors.append("Engine Status is failed (0).")
                if thrust < 20:
                    exact_factors.append(f"Engine thrust is critically low: {thrust}%.")
            elif phase == 'landing' and cross >= 25 and runway > 0:
                prediction_label = "Runway Excursion"
                if "Crosswind and contaminated runway exceed absolute safety limits." not in exact_factors:
                    exact_factors.append("Crosswind and contaminated runway exceed absolute safety limits.")
            elif phase == 'approach' and speed < 120:
                prediction_label = "Stall During Approach"
                if "Airspeed violating stall margin." not in exact_factors:
                    exact_factors.append("Airspeed violating stall margin.")
            elif phase == 'landing' and engine == 1 and runway == 0 and cross < 15 and speed < 145:
                prediction_label = "Safe Landing"
                exact_factors = [] 

            if prediction_label != "Safe Approach" and prediction_label != "Safe Landing":
                recent_frame_scaled = feat_scaled[0, 3]
                for idx, val in enumerate(recent_frame_scaled):
                    col_name = (FEATURE_COLUMNS + ['delta_vs', 'energy_state', 'glide_dev'])[idx] if ADVANCED_ML_AVAILABLE else FEATURE_COLUMNS[idx]
                    clean_name = col_name.replace('_', ' ').title()
                    if val > 0.95:
                        exact_factors.append(f"Critical high-limit exceedance detected in {clean_name}.")
                    elif val < 0.05:
                        exact_factors.append(f"Dangerous deviation (low margin) observed in {clean_name}.")
                
                if engine == 0 and "Dangerous deviation (low margin) observed in Engine Status." not in exact_factors and "Engine Status is failed (0)." not in exact_factors: 
                    exact_factors.append("Engine Status is failed (0).")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("Inference error:\n", error_details)
            prediction_label = "Inference Error"
            description = f"Exception details: {e}"
    else:
        if engine == 0 or thrust < 20:
             prediction_label = "Engine Failure"
             if engine == 0: exact_factors.append("Reported Engine Status is Failed.")
             if thrust < 20: exact_factors.append(f"Engine thrust is critically low: {thrust}%.")
        elif speed > 160 and phase == 'landing':
             prediction_label = "Hard Landing"
             exact_factors.append(f"Aircraft landing speed is excessively high: {speed} knots (limit is ~160).")
        elif phase == 'approach' and pitch < -5:
             prediction_label = "CFIT"
             exact_factors.append(f"Nose pitch angle is dangerously steep: {pitch} degrees.")
             if alt < 1000: exact_factors.append(f"Altitude is severely degraded at {alt} ft given the pitch.")
        elif speed < 120 and phase == 'approach':
             prediction_label = "Stall During Approach"
             exact_factors.append(f"Airspeed has dropped to {speed} knots, violating the stall margin.")
        elif phase == 'landing' and wind > 25:
             prediction_label = "Runway Excursion"
             exact_factors.append(f"Wind speed ({wind} knots) exceeds maximum safe landing limits.")
             if runway > 0: exact_factors.append("Wet or contaminated runway condition exacerbates poor braking.")
        else:
             prediction_label = "Safe Approach" if phase == 'approach' else "Safe Landing"

    if prediction_label == "Engine Failure":
        severity = "Critical"
        description = "Loss of thrust in one or more engines. Requires immediate pilot action to maintain controlled flight."
        causes_text = "Multiple factors can lead to engine failure."
        causes_points = ["Fuel starvation", "Bird strike processing", "Mechanical malfunction", "Severe icing"]
        prevention_text = "Recommended Pilot Actions:"
        prevention_points = ["Maintain safe airspeed (Vglide)", "Execute engine restart checklist", "Declare emergency (Mayday)", "Prepare for nearest appropriate landing zone"]
    elif prediction_label == "Inference Error":
        severity = "Unknown"
        description = description if description.startswith("Exception details") else "An unexpected error occurred during prediction. Please verify system dependencies and loaded models."
        causes_text = "Error Logs:"
        causes_points = ["Backend Exception caught during Keras feedforward pass."]
        prevention_text = "Administrator Action:"
        prevention_points = ["Check server logs", "Verify tf/keras environments"]
    elif prediction_label == "Hard Landing":
        severity = "High"
        description = "Aircraft contacts the ground with excessive vertical speed or force, risking structural damage."
        causes_text = "Typically results from unstable approaches."
        causes_points = ["High vertical descent rate", "Improper flare execution", "Wind shear or severe downdrafts"]
        prevention_text = "To minimize risk:"
        prevention_points = ["Execute a go-around if approach is unstable", "Maintain proper approach speed (Vref)", "Anticipate crosswinds and wind gusts"]
    elif prediction_label == "Runway Excursion":
        severity = "High"
        description = "Aircraft veers off or overruns the runway surface."
        causes_text = "Often caused by a combination of weather and aircraft state."
        causes_points = ["Wet or icy runway conditions", "Excessive landing speed", "Late touchdown point", "Brake or thrust reverser failure"]
        prevention_text = "Safety measures:"
        prevention_points = ["Ensure touchdown within the targeted zone", "Apply maximum correct braking and reverse thrust", "If deep landing is inevitable, execute an early go-around"]
    elif prediction_label == "CFIT":
        severity = "Critical"
        description = "Controlled Flight Into Terrain. An airworthy aircraft is unintentionally flown into the ground, a mountain, or an obstacle."
        causes_text = "Usually due to loss of situational awareness."
        causes_points = ["Poor visibility / IMC conditions", "Incorrect altimeter setting", "Ignoring Ground Proximity Warning System (GPWS)"]
        prevention_text = "Immediate corrections needed:"
        prevention_points = ["Pull up immediately if GPWS sounds", "Verify altitude against minimum safe altitudes", "Maintain instrument cross-check"]
    elif prediction_label == "Stall During Approach":
        severity = "Critical"
        description = "Aircraft exceeds critical angle of attack resulting in loss of lift during a critical phase of flight."
        causes_text = "Airspeed and pitch management failures."
        causes_points = ["Airspeed dropping below stall speed (Vs)", "Aggressive maneuvering at low speeds", "Icing accumulation altering airflow"]
        prevention_text = "Recovery steps:"
        prevention_points = ["Reduce angle of attack (pitch down)", "Apply maximum thrust", "Recover altitude once airspeed increases"]
    else:
        severity = "Low"
        description = "All parameters nominal. Flight is proceeding safely according to standard operating procedures."
        causes_text = ""
        causes_points = []
        prevention_text = ""
        prevention_points = []
        
    return {
        "prediction": prediction_label,
        "severity": severity,
        "outcome": outcome,
        "description": description,
        "causes_text": causes_text,
        "causes_points": causes_points,
        "prevention_text": prevention_text,
        "prevention_points": prevention_points,
        "exact_factors": exact_factors,
        "phase": phase
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    phase = request.form.get('phase', 'approach')
    def parse_float(val, default_val=0.0):
        try: return float(val)
        except (TypeError, ValueError): return default_val

    alt = parse_float(request.form.get('alt'))
    speed = parse_float(request.form.get('speed'))
    pitch = parse_float(request.form.get('pitch'))
    vs = parse_float(request.form.get('vs'))
    thrust = parse_float(request.form.get('thrust'))
    wind = parse_float(request.form.get('wind'))
    cross = parse_float(request.form.get('cross'))
    vis = parse_float(request.form.get('vis'))
    temp = parse_float(request.form.get('temp'))
    hum = parse_float(request.form.get('hum'))
    weather = parse_float(request.form.get('weather'))
    time_val = parse_float(request.form.get('time'))
    runway = parse_float(request.form.get('runway'))
    error = parse_float(request.form.get('error'))
    exp = parse_float(request.form.get('exp'))
    weight = parse_float(request.form.get('weight'))
    flap = parse_float(request.form.get('flap'))
    engine = parse_float(request.form.get('engine'))
    fuel = parse_float(request.form.get('fuel'))
    
    features = [alt, speed, pitch, vs, thrust, wind, cross, vis, temp, hum, weather, time_val, runway, error, exp, weight, flap, engine, fuel]
    res = process_flight_features(features, phase)
    
    return render_template('result.html', 
                            prediction=res['prediction'], 
                            severity=res['severity'], 
                            outcome=res['outcome'],
                            description=res['description'],
                            causes_text=res['causes_text'],
                            causes_points=res['causes_points'],
                            prevention_text=res['prevention_text'],
                            prevention_points=res['prevention_points'],
                            exact_factors=res['exact_factors'],
                            phase=res['phase'],
                            advanced_ml=ADVANCED_ML_AVAILABLE)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        import pandas as pd
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error reading CSV: {e}", 400
        
        results = []
        for index, row in df.iterrows():
            phase = str(row.get('phase', 'approach')).lower() if 'phase' in row else 'approach'
            if phase not in ['approach', 'landing']:
                phase = 'approach'
                
            def parse_row(col_name, default=0.0):
                return float(row.get(col_name, default)) if col_name in row else default

            alt = parse_row('altitude_ft', parse_row('Altitude', 0.0))
            speed = parse_row('speed_knots', parse_row('Speed', 0.0))
            pitch = parse_row('pitch_deg', parse_row('Pitch', 0.0))
            vs = parse_row('vertical_speed', parse_row('Vertical Speed', 0.0))
            thrust = parse_row('engine_thrust', parse_row('Thrust', 0.0))
            wind = parse_row('wind_speed_knots', parse_row('Wind Speed', 0.0))
            cross = parse_row('crosswind_component', parse_row('Crosswind', 0.0))
            vis = parse_row('visibility_km', parse_row('Visibility', 0.0))
            temp = parse_row('temperature_celsius', parse_row('Temperature', 0.0))
            hum = parse_row('humidity_percent', parse_row('Humidity', 0.0))
            weather = parse_row('weather', parse_row('Weather', 0.0))
            time_val = parse_row('time_of_day', parse_row('Time', 0.0))
            runway = parse_row('runway_condition', parse_row('Runway Condition', 0.0))
            error = parse_row('pilot_error', parse_row('Pilot Error', 0.0))
            exp = parse_row('pilot_experience', parse_row('Pilot Experience', 0.0))
            weight = parse_row('aircraft_weight', parse_row('Weight', 0.0))
            flap = parse_row('flap_setting', parse_row('Flaps', 0.0))
            engine = parse_row('engine_status', parse_row('Engine Status', 1.0))
            fuel = parse_row('fuel_level', parse_row('Fuel', 100.0))

            features_row = [alt, speed, pitch, vs, thrust, wind, cross, vis, temp, hum, weather, time_val, runway, error, exp, weight, flap, engine, fuel]
            
            res = process_flight_features(features_row, phase)
            results.append(res)
            
        return render_template('batch_result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
