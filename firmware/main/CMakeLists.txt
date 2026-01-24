idf_component_register(
    SRCS "main.cpp" "NeuralNetwork.cpp"  # Adicionado NeuralNetwork e o seu modelo novo
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro driver  # "esp-tflite-micro" é o nome oficial no IDF
)