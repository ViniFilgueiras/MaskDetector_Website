#pragma once
#include <cstdint>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// CORREÇÃO 1: Usamos uint8_t para a pontuação ser positiva (0 a 255)
struct Prediction {
    uint8_t score_with_mask;
    uint8_t score_without_mask;
};

class NeuralNetwork {
public:
    NeuralNetwork();
    // O destrutor não é estritamente necessário se não dermos delete, mas ok
    ~NeuralNetwork() {} 

    // O input continua sendo int8 (assinado)
    int8_t* getInputBuffer();
    
    Prediction predict();

private:
    // 250KB de arena
    static constexpr int kArenaSize = 250 * 1024;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    uint8_t* tensor_arena = nullptr; // Arena tem que ser uint8 (bytes crus)
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
};