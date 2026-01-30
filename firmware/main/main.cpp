#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/usb_serial_jtag.h"
#include "NeuralNetwork.h"
#include "esp_heap_caps.h"

static const char *TAG = "MASK_DETECTOR";

// Função para enviar resposta via USB
void send_usb_response(const char* message) {
    usb_serial_jtag_write_bytes((const int8_t*)message, strlen(message), pdMS_TO_TICKS(100));
    usb_serial_jtag_write_bytes((const int8_t*)"\n", 1, pdMS_TO_TICKS(100)); // Nova linha
}

// Função para ler dados do USB
bool read_usb_data(int8_t *buffer, int needed_bytes) {
    int total_read = 0;
    while (total_read < needed_bytes) {
        int len = usb_serial_jtag_read_bytes(buffer + total_read, needed_bytes - total_read, pdMS_TO_TICKS(50));
        if (len > 0) {
            total_read += len;
        } else {
            vTaskDelay(pdMS_TO_TICKS(10)); // Evita travar a CPU esperando
        }
    }
    return true;
}

extern "C" void app_main(void) {
    
    // 1. Inicializa USB Serial JTAG
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&usb_serial_jtag_config);
    
    ESP_LOGI(TAG, "USB Inicializado. Aguardando conexão...");

    // 2. Inicializa a Rede Neural
    NeuralNetwork nn;
    
    // Buffer para o cabeçalho
    int8_t header[5]; 
    const int IMAGE_SIZE = 96 * 96 * 3;

    // Buffer temporário para receber dados crus do USB (0-255)
    uint8_t* temp_usb_buffer = (uint8_t*) heap_caps_malloc(
    IMAGE_SIZE,
    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );


    if (temp_usb_buffer == NULL) {
        ESP_LOGE(TAG, "Falha ao alocar buffer temporário!");
        return;
    }

    while (true) {
        // 1. Lê o cabeçalho
        if (usb_serial_jtag_read_bytes(header, 5, pdMS_TO_TICKS(100)) == 5) {
            
            ESP_LOGI(TAG, "Cabeçalho recebido! Lendo imagem...");
            
            // 2. Lê a imagem para o buffer TEMPORÁRIO (fazendo cast para int8_t* só para a função aceitar)
            if (read_usb_data((int8_t*)temp_usb_buffer, IMAGE_SIZE)) {
                
                // 3. O GRANDE FIX: Normalização
                int8_t* input_buffer = nn.getInputBuffer();
                
                for (int i = 0; i < IMAGE_SIZE; i++) {
                    int pixel_cru = temp_usb_buffer[i]; // 0 a 255
                    
                    // Subtrai 128 (vira -128 a 127)
                    input_buffer[i] = (int8_t)(pixel_cru - 128);
                }

                // 4. Roda a predição
                Prediction p = nn.predict();
                
                // 5. Envia resposta
                char response[100];
                snprintf(response, sizeof(response), "M:%d S:%d", p.score_with_mask, p.score_without_mask);
                send_usb_response(response);
                
                ESP_LOGI(TAG, "Predição: %s", response);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    // free(temp_usb_buffer); // Inacessível, mas boa prática manter comentado
}