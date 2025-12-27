import tensorflow as tf
import tf2onnx

keras_model_path = "ea_stop_model.h5"
onnx_model_path = "newmodels.onnx"

# Charger le modèle sans compilation
model = tf.keras.models.load_model(keras_model_path, compile=False)

# Forcer la présence d'un attribut output_names (contournement bug tf2onnx)
model.output_names = ["output"]

# Spécifier la signature d'entrée
spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)

# Convertir en ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Sauvegarder le modèle ONNX
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Modèle ONNX sauvegardé dans {onnx_model_path}")
