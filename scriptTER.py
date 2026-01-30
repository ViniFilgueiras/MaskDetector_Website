"""
Script para enviar imagem 96x96 RGB para ESP32-S3 via USB
"""
import serial
import numpy as np
import struct
import argparse
import time

def create_test_image():
    """Cria uma imagem de teste 96x96x3"""
    img = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    # Opcional: criar padrão específico
    # img[:32, :, 0] = 255  # Metade superior vermelha
    # img[32:, :, 2] = 255  # Metade inferior azul
    return img

def send_image(port, baudrate, image):
    """Envia imagem para ESP32 e recebe resultado"""
    try:
        ser = serial.Serial(port, baudrate, timeout=5)
        time.sleep(2)  # Aguarda ESP inicializar
        
        print(f"Conectado em {port}")
        print(f"Tamanho da imagem: {image.shape}")
        
        # Envia cabeçalho com dimensões
        header = struct.pack('<HHB', 96, 96, 3)  # width, height, channels
        ser.write(header)
        
        # Envia dados da imagem
        img_bytes = image.tobytes()
        total_bytes = len(img_bytes)
        print(f"Enviando {total_bytes} bytes...")
        
        chunk_size = 1024
        for i in range(0, total_bytes, chunk_size):
            chunk = img_bytes[i:i+chunk_size]
            ser.write(chunk)
            print(f"Enviado: {i+len(chunk)}/{total_bytes} bytes", end='\r')
        
        print(f"\nImagem enviada com sucesso!")
        
# --- BLOCO NOVO: Leitura Inteligente ---
        print("Aguardando classificação (ignorando logs do sistema)...")
        
        start_time = time.time()
        while (time.time() - start_time) < 20:  # Tenta ler por até 5 segundos
            line = ser.readline().decode('utf-8').strip()
            
            if not line:
                continue
                
            # Se for log de sistema (começa com "I ("), apenas mostra mas continua esperando
            if line.startswith("I (") or "Cabeçalho" in line:
                print(f"[Log do ESP]: {line}")
                continue
            
            # Se chegamos aqui, provavelmente é a resposta real!
            print(f"\n--- Resultado da ESP32 ---")
            print(f"Classificação: {line}")
            print(f"-------------------------")
            break
        # ---------------------------------------
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"Erro de comunicação serial: {e}")
    except Exception as e:
        print(f"Erro: {e}")

def load_image(image_path):
    """Carrega e redimensiona imagem para 96x96"""
    try:
        from PIL import Image
        print(f"Carregando imagem: {image_path}")
        img = Image.open(image_path)
        print(f"Tamanho original: {img.size}, Modo: {img.mode}")

        # Redimensiona para 96x96 e converte para RGB
        img = img.resize((96, 96), Image.Resampling.LANCZOS).convert('RGB')
        image = np.array(img)
        print(f"Imagem processada: {image.shape}")
        return image
        
    except ImportError:
        print("ERRO: PIL/Pillow não está instalado!")
        print("Instale com: pip install pillow")
        return None
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{image_path}' não encontrado!")
        return None
    except Exception as e:
        print(f"ERRO ao carregar imagem: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Envia imagem 96x96 para ESP32-S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s -p /dev/ttyUSB0 -i foto.jpg
  %(prog)s -p COM3 -i C:\\Users\\foto.png
  %(prog)s -p /dev/ttyUSB0  (usa imagem de teste aleatória)
        """)
    
    parser.add_argument('-p', '--port', default='/dev/ttyUSB0', 
                       help='Porta serial (ex: /dev/ttyUSB0 ou COM3)')
    parser.add_argument('-b', '--baudrate', type=int, default=115200,
                       help='Baudrate (padrão: 115200)')
    parser.add_argument('-i', '--image', 
                       help='Caminho para imagem (JPG, PNG, etc.)')
    
    args = parser.parse_args()
    
    if args.image:
        image = load_image(args.image)
        if image is None:
            print("\nUsando imagem de teste aleatória como fallback.")
            image = create_test_image()
    else:
        print("Nenhuma imagem especificada. Gerando imagem de teste aleatória.")
        image = create_test_image()
    
    send_image(args.port, args.baudrate, image)

if __name__ == '__main__':
    main()
