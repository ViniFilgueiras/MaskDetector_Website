#include "NeuralNetwork.h"
#include "model_data.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

static const char* TAG = "NeuralNetwork";

NeuralNetwork::NeuralNetwork()
{
    // Inicializa o TFLM
    tflite::InitializeTarget();

    // Carrega o modelo
    model = tflite::GetModel(mask_detector_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Erro de versão do schema: %d", model->version());
        return;
    }

    // Resolver das operações usadas
    static tflite::MicroMutableOpResolver<20> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddMean();

    tensor_arena = (uint8_t*) heap_caps_malloc(
        kArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );


    if (!tensor_arena) {
        ESP_LOGE(TAG, "Falha ao alocar tensor arena (%d bytes)", kArenaSize);
        return;
    }

    ESP_LOGI(TAG, "Tensor arena alocada em %p (%d bytes)", tensor_arena, kArenaSize);

    // Interpreter precisa ser estático (exigência do TFLM)
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kArenaSize
    );
    interpreter = &static_interpreter;

    // Aloca tensores
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() falhou!");
        return;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Modelo carregado! Input bytes: %d", input->bytes);
}

NeuralNetwork::~NeuralNetwork()
{
    if (tensor_arena) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
}

int8_t* NeuralNetwork::getInputBuffer()
{
    return input->data.int8;
}

Prediction NeuralNetwork::predict()
{
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Erro ao invocar o modelo!");
        return {0, 0};
    }

    // Saída int8 (-128 a 127)
    int raw_mask   = output->data.int8[0];
    int raw_nomask = output->data.int8[1];

    // Converte para 0–255
    Prediction result;
    result.score_with_mask    = (uint8_t)(raw_mask + 128);
    result.score_without_mask = (uint8_t)(raw_nomask + 128);

    return result;
}
