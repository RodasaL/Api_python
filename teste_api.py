# -*- coding: utf-8 -*-
"""
API para Classificação de Imagens com TensorFlow Lite
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.lite import Interpreter

# Configurações do TensorFlow Lite
MODEL_PATH = "animal_classifier_model.tflite"
NUM_CLASSES = 6
TARGET_SIZE = (96, 96)

# Mapear os índices das classes para os seus rótulos reais
cifar_labels = [
    "passaro", "gato", "viado",
    "cao", "sapo", "cavalo"
]

# Otimizações para reduzir logs e uso de GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Apenas erros críticos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativar operações oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forçar uso apenas de CPU

# Inicializar Flask
app = Flask(__name__)

# Carregar o modelo TensorFlow Lite
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print(f"Modelo TensorFlow Lite carregado com sucesso de {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo TensorFlow Lite: {e}")
    interpreter = None


def predict_with_tflite(image):
    """Faz a previsão usando o modelo TensorFlow Lite."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Configurar entrada
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obter resultados
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions


@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo foi fornecido"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        # Abrir a imagem enviada e pré-processá-la
        image = Image.open(file).resize(TARGET_SIZE)
        image = np.array(image) / 255.0  # Normalizar
        image = image.reshape(1, *TARGET_SIZE, 3)  # Adicionar dimensão do batch

        # Fazer a previsão
        prediction = predict_with_tflite(image)
        class_id = np.argmax(prediction)  # Obter o índice da classe com maior probabilidade
        confidence = float(np.max(prediction))  # Confiança da previsão
        class_label = cifar_labels[class_id] if class_id < len(cifar_labels) else "Desconhecida"

        return jsonify({
            "class_id": int(class_id),
            "class_label": class_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
