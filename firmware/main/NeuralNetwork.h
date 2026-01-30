#pragma once

#include <cstdint>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Scores finais (0 a 255)
struct Prediction {
    uint8_t score_with_mask;
    uint8_t score_without_mask;
};

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    // Buffer de entrada normalizado (-128 a 127)
    int8_t* getInputBuffer();

    Prediction predict();

private:
    // 320 KB de arena
    static constexpr int kArenaSize = 320 * 1024;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;

    uint8_t* tensor_arena = nullptr;

    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
};
