import socket
import numpy as np
import onnxruntime as ort

# Load the ONNX model
onnx_session = ort.InferenceSession(r"C:\Users\pc\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\pilot_fish_model.onnx")

# Socket server setup
HOST = '127.0.0.1'  # Localhost
PORT = 8466         # Port to listen on

def verify_features(features, feature_names=None):
    """Verify that the features are valid (finite and within expected ranges)."""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[1])]

    is_valid = True
    for i, feature_value in enumerate(features[0]):
        if not np.isfinite(feature_value):
            print(f"{feature_names[i]} is not finite: {feature_value}")
            is_valid = False
        elif feature_value < -1e6 or feature_value > 1e6:  # Example range
            print(f"{feature_names[i]} is out of range: {feature_value}")
            is_valid = False
        else:
            print(f"{feature_names[i]} is valid: {feature_value}")

    return is_valid

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen()
    print(f"Server listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            if not data:
                break

            # Debug raw data
            print(f"Raw data received: {data}")
            print(f"Raw data length: {len(data)}")
            print(f"Raw data as hex: {data.hex()}")

            # Ensure that only 40 bytes are considered
            data = data[:40]  # Trim the data to 40 bytes if it's longer than expected

            # Validate data length
            expected_length = 5 * 8  # 5 features * 8 bytes per float64
            if len(data) != expected_length:
                print(f"Data length mismatch: expected {expected_length}, got {len(data)}")
                continue

            try:
                # Parse the incoming data
                features = np.frombuffer(data, dtype=np.float64, count=5).reshape(1, -1)
                print(f"Parsed features: {features}")
                print(f"Shape of parsed features: {features.shape}")

                # Validate the features
                feature_names = ["HMA_250", "HMA_25", "Price_Diff", "HMA_Diff", "Close_vs_Open"]
                if not verify_features(features, feature_names):
                    print("Feature verification failed, skipping...")
                    continue

            except Exception as e:
                print(f"Error parsing features: {e}")
                continue

            # Model input debugging
            input_name = onnx_session.get_inputs()[0].name
            print(f"Model input name: {input_name}")
            print(f"Expected input shape: {onnx_session.get_inputs()[0].shape}")

            # Make prediction
            pred = onnx_session.run(None, {input_name: features.astype(np.float32)})[0]

            # Print prediction
            print(f"Prediction: {pred}")
            print(f"Prediction shape: {pred.shape}")

            # Handle the prediction output
            if pred.ndim == 1:
                prediction_value = pred[0]
            else:
                prediction_value = pred[0][0]
            
            print(f"Prediction value to send: {prediction_value}")

            # Send the prediction back
            conn.sendall(str(prediction_value).encode())
def validate_features(features):
    """Check if the feature values are within a reasonable range."""
    for i, feature in enumerate(features[0]):
        if np.abs(feature) < 1e-6:  # Consider values too close to zero as invalid
            print(f"Warning: Feature {i} is too small: {feature}")
            features[0, i] = np.nan  # Set invalid values to NaN

    # Check if all features are valid
    if np.any(np.isnan(features)):
        print("Error: Invalid features detected.")
        return False
    return True

# Model inference and prediction
if validate_features(features):
    # Make prediction only if features are valid
    pred = onnx_session.run(None, {input_name: features.astype(np.float32)})[0]
    print(f"Prediction: {pred}")
    print(f"Prediction value to send: {pred[0]}")
    conn.sendall(str(pred[0]).encode())
else:
    print("Skipping prediction due to invalid features.")
    conn.sendall("Error: Invalid features".encode())
