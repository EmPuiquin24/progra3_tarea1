#include "../include/Tensor.h"
#include <iostream>

using namespace std;

int main() {
    srand(time(nullptr));

    // 1. Crear un tensor de entrada de dimensiones 1000 × 20 × 20.
    Tensor T1 = Tensor::random({1000, 20, 20}, 9, 25);
    
    // 2. Transformarlo a 1000 × 400 usando view.
    Tensor T2 = T1.view({1000, 400});
    
    // 3. Multiplicarlo por una matriz 400 × 100.
    Tensor W1 = Tensor::random({400, 100}, 2, 30);
    Tensor T3 = matmul(T2, W1);
    
    // 4. Sumar una matriz 1 × 100.
    Tensor T5 = Tensor::random({1000, 100}, 0, 8);
    T3 = T3 + T5;
    
    // 5. Aplicar la función ReLU.
    ReLU relu;
    Tensor T6 = T3.apply(relu);
    
    // 6. Multiplicar por una matriz 100 × 10
    Tensor W2 = Tensor::random({100, 10}, 121, 204);
    Tensor T7 = matmul(T6, W2);
    
    // 7. Sumar una matriz 1 × 10 (bias)
    Tensor T8 = Tensor::random({1000, 10}, 123, 777);
    T7 = T7 + T8;
    
    // 8. Aplicar la función Sigmoid
    Sigmoid sigmoid;
    Tensor T9 = T7.apply(sigmoid);
    
    // Imprimir dimensiones finales :o
    T9.print_dimensions();
    
    return 0;
}