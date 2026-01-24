#include "NeuralNetwork.h"
#include "model_data.h" 
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Usamos o Mutable (mais leve e compatível)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

NeuralNetwork::NeuralNetwork()
{
    // Carrega o modelo
    model = tflite::GetModel(mask_detector_int8_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Erro de versão do Schema: %d!", model->version());
        return;
    }

    // Adicionamos manualmente as camadas
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

    // Aloca memória (tensor_arena) como uint8_t (Obrigatório do TFLite)
    tensor_arena = (uint8_t *)malloc(kArenaSize);
    if (!tensor_arena) {
        MicroPrintf("Falha ao alocar arena de memória!");
        return;
    }

    // Cria o interpretador
    static tflite::MicroInterpreter static_interpreter(
        model, 
        resolver, 
        tensor_arena, 
        kArenaSize
    );
    interpreter = &static_interpreter;

    // Aloca os tensores internos
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() falhou!");
        return;
    }

    // Conecta os ponteiros
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    MicroPrintf("Modelo carregado! Input bytes: %d", input->bytes);
}

// Retorna ponteiro int8 (assinado) para preencher com (pixel - 128)
int8_t* NeuralNetwork::getInputBuffer() {
    return input->data.int8;
}

Prediction NeuralNetwork::predict()
{
    // Roda a rede neural
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Erro ao invocar o modelo!");
        return {0, 0};
    }

    Prediction result;
    
    // CORREÇÃO: Conversão de int8 (-128 a 127) para uint8 (0 a 255)
    int raw_mask = output->data.int8[0];
    int raw_nomask = output->data.int8[1];

    // Somamos 128 para tirar o negativo
    result.score_with_mask = (uint8_t)(raw_mask + 128);
    result.score_without_mask = (uint8_t)(raw_nomask + 128);
    
    return result;
}