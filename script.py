"""
Servidor intermediário:
HTML -> Python -> ESP32-S3
Modelo: 96x96 RGB (INT8)
"""

import serial
import time
import struct
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import re

# ===================== CONFIGURAÇÕES =====================
SERIAL_PORT = "COM5"
BAUDRATE = 115200
IMAGE_W = 96
IMAGE_H = 96
CHANNELS = 3
IMAGE_SIZE = IMAGE_W * IMAGE_H * CHANNELS
ser = None
# =========================================================

app = Flask(__name__)

# =========================================================
# Inicialização da serial
# =========================================================
def init_serial():
    global ser
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Aguarda a inicialização da ESP
    print("Conectado à ESP32 na porta", SERIAL_PORT)

# =========================================================
# Comunicação com ESP32
# =========================================================
def send_image_to_esp(image: np.ndarray) -> str:
    global ser
    
    if ser is None or not ser.is_open:
        raise RuntimeError("Serial não inicializada")
    
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.05) # Pequena pausa antes de enviar

    header = struct.pack("<HHB", IMAGE_W, IMAGE_H, CHANNELS)
    ser.write(header)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    image = np.ascontiguousarray(image)
    img_bytes = image.tobytes(order="C")
    total_bytes = len(img_bytes)
    print(f"Enviando {total_bytes} bytes...")
        
    chunk_size = 1024
    for i in range(0, total_bytes, chunk_size):
        chunk = img_bytes[i:i+chunk_size]
        print(f"Enviado: {i+len(chunk)}/{total_bytes} bytes", end='\r')
        ser.write(chunk)
    ser.flush()
    
    start_time = time.time()
    last_line = ""
    while time.time() - start_time < 5:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        last_line = line
        if line.startswith("I (") or "Cabeçalho" in line:
            continue

        if re.search(r"\bM:(\d+)\b", line) and re.search(r"\bS:(\d+)\b", line):
            return line

    raise TimeoutError(f"ESP não respondeu com M:/S: (última linha: {last_line})")

# =========================================================
# ROTAS
# =========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "Imagem não enviada"}), 400

    file = request.files["image"]
    raw_result = ""
    print("Recebendo imagem para classificação...")

    m, s = 0, 0
    confidence = 0.0

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        print(img.format, img.size, img.mode)

        image_np = np.array(img, dtype=np.uint8)

        print("Enviando imagem para ESP...")
        raw_result = send_image_to_esp(image_np)
        print("Resposta da ESP:", raw_result)
        # Esperado: "M:180 S:60"

        m_match = re.search(r'M:(\d+)', raw_result)
        s_match = re.search(r'S:(\d+)', raw_result)

        if not m_match or not s_match:
                raise ValueError(f"Resposta inválida da ESP: {raw_result}")

        m = int(m_match.group(1))
        s = int(s_match.group(1))

        print("Probabilidades - Masked:", m, "Unmasked:", s)
        
        mask = m > s
        confidence = (max(m, s) / (m + s)) if (m + s) > 0 else 0.0

        print(mask, confidence)

        return jsonify({
            "mask": mask,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        print("Erro ao processar a imagem ou comunicar com a ESP:", str(e))
        return jsonify({
            "error": "Falha ao processar resposta da ESP",
            "raw_response": raw_result,
            "exception": str(e)
        }), 500


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Servidor rodando em http://localhost:5000")
    print("Conectando à ESP em", SERIAL_PORT)
    init_serial()
    app.run(host="0.0.0.0", port=5000, debug=False)
