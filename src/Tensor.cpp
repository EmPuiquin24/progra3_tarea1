#include "../include/Tensor.h"

Tensor::Tensor(double* data, int* ref_count, const std::vector<size_t>& shape)
    : data(data), ref_count(ref_count), shape(shape) {
    (*ref_count)++;
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) 
    : shape(shape), values(values) {
        
        // Validación de tamaño
		if (shape.size() > 3 || shape.size() < 1) {
			throw std::invalid_argument("El Tensor debe tener 1, 2 o 3 dimesiones");
		}

        // Validación de la cantidad de valores
		int expectedValues= 1;
		for (size_t n : shape) {
			expectedValues = expectedValues * n;	
		}
		if (expectedValues != values.size()) {
			throw std::logic_error("La cantidad de valores otorgados no coincide con el tamaño del tensor");
		}


        this->data = new double[expectedValues];
        for (size_t i = 0; i < expectedValues; i++) {
            this->data[i] = values[i];
        }
        
        this->ref_count = new int(1);
}

Tensor::Tensor(Tensor&& otro) noexcept {

    this->data = otro.data;
    this->ref_count = otro.ref_count;
    this->shape = otro.shape;
    this->values = otro.values;

    otro.data = nullptr;
    otro.ref_count = nullptr;
}

Tensor::Tensor(const Tensor& otro) 
    : data(otro.data), ref_count(otro.ref_count), shape(otro.shape), values(otro.values) {
    if (ref_count) {
        (*ref_count)++;
    }
}

Tensor& Tensor::operator=(const Tensor& otro) {
    if (this != &otro) {
        if (ref_count) {
            (*ref_count)--;
            if (*ref_count == 0) {
                delete[] data;
                delete ref_count;
            }
        }
        
        data = otro.data;
        ref_count = otro.ref_count;
        shape = otro.shape;
        values = otro.values;
        
        if (ref_count) {
            (*ref_count)++;
        }
    }
    return *this;
}

Tensor::~Tensor() {
    if (ref_count) {
        (*ref_count)--;
        if (*ref_count == 0) {
            delete[] data;
            delete ref_count;
        }
    }
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    
    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n; 
    }

    std::vector<double> values(amountOfValues, 0.0);
    return Tensor(shape, values);
}


Tensor Tensor::ones(const std::vector<size_t>& shape) {
    
    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n;
    }

    std::vector<double> values(amountOfValues, 1.0);
    return Tensor(shape, values);
}


Tensor Tensor::random(const std::vector<size_t>& shape, const int& min, const int& max) {
    
    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n;
    }

    // Creación de el vector values con cantidades aleatorias
    
    std::vector<double> values; values.reserve(amountOfValues);
    size_t i = 0;
    while (i < amountOfValues) {
        values.push_back(rand() % max + min);
        i++;
    }

    return Tensor(shape, values);
}


Tensor Tensor::arange(const int& start, const int& end) {

    size_t amountOfValues = end - start - 1;

    // Creación del vector values con valores secuenciales
    std::vector<double> values; values.reserve(amountOfValues); 
    double i = start;
    while (i < end) {
        values.push_back(i);
        i++;
    } 
    
    return Tensor({amountOfValues}, values);
}


Tensor Tensor::operator+(const Tensor& otro) {

    if (this->shape != otro.shape) {
        throw std::logic_error("Las dimensiones no coinciden");
    }

    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n; 
    }

    std::vector<double> new_values; new_values.reserve(amountOfValues);
    for (size_t i = 0; i < amountOfValues; i++) {
        new_values.push_back(this->data[i] + otro.data[i]);
    }

    return Tensor(this->shape, new_values);
}


Tensor Tensor::operator-(const Tensor& otro) {

    if (this->shape != otro.shape) {
        throw std::logic_error("las dimensiones no coinciden");
    }

    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n; 
    }

    std::vector<double> new_values; new_values.reserve(amountOfValues);
    for (size_t i = 0; i < amountOfValues; i++) {
        new_values.push_back(this->data[i] - otro.data[i]);
    }

    return Tensor(this->shape, new_values);
}


Tensor Tensor::operator*(const Tensor& otro) {

    if (this->shape != otro.shape) {
        throw std::logic_error("las dimensiones no coinciden");
    }

    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n; 
    }

    std::vector<double> new_values; new_values.reserve(amountOfValues);
    for (size_t i = 0; i < amountOfValues; i++) {
        new_values.push_back(this->data[i] * otro.data[i]);
    }

    return Tensor(this->shape, new_values);
}


Tensor Tensor::operator*(const double& n) {

    size_t amountOfValues = 1;
    for (size_t n : shape) {
        amountOfValues *= n; 
    }

    std::vector<double> new_values; new_values.reserve(amountOfValues);
    for (size_t i = 0; i < amountOfValues; i++) {
        new_values.push_back(this->data[i] * n);
    }

    return Tensor(this->shape, new_values);
}


Tensor Tensor::view(const std::vector<size_t>& new_shape) {

    if (new_shape.size() > 3 || new_shape.size() < 1) {
			throw std::invalid_argument("El Tensor debe tener 1, 2 o 3 dimesiones");
	}
	
	size_t amountOfAValues = 1;
	for (size_t n : shape) {
		amountOfAValues *= n;
	}

	size_t amountOfBValues = 1;
	for (size_t n : new_shape) {
		amountOfBValues *= n;
	}

	if (amountOfAValues != amountOfBValues) {
		throw std::invalid_argument("El número total de elementos debe mantenerse constante");
	}
	
	return Tensor(this->data, this->ref_count, new_shape);
}


Tensor Tensor::unsqueeze(const size_t& position) {
	
	if (position > shape.size() || position < 0) {
        throw std::invalid_argument("posición inválida"); 
    }
    
    if (shape.size() >= 3) {
		throw std::invalid_argument("no puede exceder 3 dimensiones");
	}
	
	std::vector<size_t> new_shape; new_shape.reserve(shape.size() + 1);
    
    for (size_t i = 0; i < position; i++) {
		new_shape.push_back(shape[i]);
	}
	new_shape.push_back(1);
	for (size_t i = position; i < shape.size(); i++) {
		new_shape.push_back(shape[i]);
	}
	
	return Tensor(this->data, this->ref_count, new_shape);
}


Tensor Tensor::concat(const std::vector<Tensor>& tensors, const size_t& dimension) {
    // Debido a que esta parte del código es algo grande, añadiré comentarios puntuales :'v
	if (tensors.empty()) {
		throw std::invalid_argument("no se pueden concatenar 0 tensores");
	}
	
	// Validaciones
	size_t num_dims = tensors[0].shape.size();
	for (size_t i = 0; i < tensors.size(); i++) {
		if (tensors[i].shape.size() != num_dims) {
			throw std::invalid_argument("Todos los tensores deben tener el mismo número de dimensiones");
		}
	}
	
	// Validar que la dimensión de concatenación sea válida
	if (dimension >= num_dims) {
		throw std::invalid_argument("Dimensión inválida para concatenación");
	}
	
	// Validar compatibilidad que todas las dimensiones coincidan excepto en la dimensión de concatenación 
    for (size_t i = 0; i < num_dims; i++) {
		if (i == dimension) continue;
		size_t expected_size = tensors[0].shape[i];
		for (const auto& tensor : tensors) {
			if (tensor.shape[i] != expected_size) {
				throw std::invalid_argument("Las dimensiones deben coincidir excepto en la dimensión de concatenación");
			}
		}
	}
	
	// Calcular el nuevo shape
    std::vector<size_t> new_shape = tensors[0].shape;
	for (size_t i = 1; i < tensors.size(); i++) {
		new_shape[dimension] += tensors[i].shape[dimension];
	}
	
	// Calcular el tamaño total del tensor resultante
	size_t amountOfValues = 1;
	for (size_t n : new_shape) {
		amountOfValues *= n;
	}
	
	// Crear el vector para los nuevos valores 
	std::vector<double> new_values; new_values.reserve(amountOfValues);
	
	// Concatenar según el número de dimensiones
	if (num_dims == 1) {
		// Caso 1D: simplemente copiar todos los elementos secuencialmente
		for (const auto& tensor : tensors) {
			size_t tensorAmountOfValues = tensor.shape[0];
			for (size_t i = 0; i < tensorAmountOfValues; i++) {
				new_values.push_back(tensor.data[i]);
			}
		}
	} else if (num_dims == 2) {
		// Caso 2D
		if (dimension == 0) {
			// Concatenar por filas
			for (const auto& tensor : tensors) {
				size_t tensorAmountOfValues = tensor.shape[0] * tensor.shape[1];
				for (size_t i = 0; i < tensorAmountOfValues; i++) {
					new_values.push_back(tensor.data[i]);
				}
			}
		} else {
			// Concatenar por columnas
			size_t rows = tensors[0].shape[0];
			for (size_t r = 0; r < rows; r++) {
				for (const auto& tensor : tensors) {
					size_t cols = tensor.shape[1];
					for (size_t c = 0; c < cols; c++) {
						new_values.push_back(tensor.data[r * cols + c]);
					}
				}
			}
		}
	} else if (num_dims == 3) {
		// Caso 3D :'v
		size_t dim1 = tensors[0].shape[1]; 
		size_t dim2 = tensors[0].shape[2];
		
		if (dimension == 0) {
			// Concatenar en la primera dimensión
			for (const auto& tensor : tensors) {
				size_t tensorAmountOfValues = tensor.shape[0] * tensor.shape[1] * tensor.shape[2];
				for (size_t i = 0; i < tensorAmountOfValues; i++) {
					new_values.push_back(tensor.data[i]);
				}
			}
		} else if (dimension == 1) {
			// Concatenar en la segunda dimensión
			size_t dim0 = tensors[0].shape[0];
			for (size_t i = 0; i < dim0; i++) {
				for (const auto& tensor : tensors) {
					size_t d1 = tensor.shape[1];
					for (size_t j = 0; j < d1; j++) {
						for (size_t k = 0; k < dim2; k++) {
							new_values.push_back(tensor.data[i * d1 * dim2 + j * dim2 + k]);
						}
					}
				}
			}
		} else {
			// Concatenar en la tercera dimensión
			size_t dim0 = tensors[0].shape[0];
			for (size_t i = 0; i < dim0; i++) {
				for (size_t j = 0; j < dim1; j++) {
					for (const auto& tensor : tensors) {
						size_t d2 = tensor.shape[2];
						for (size_t k = 0; k < d2; k++) {
							new_values.push_back(tensor.data[i * dim1 * d2 + j * d2 + k]);
						}
					}
				}
			}
		}
	}
    
	return Tensor(new_shape, std::move(new_values));
}


Tensor dot(const Tensor& a, const Tensor& b) {
	
	if (a.shape != b.shape) {
		throw std::invalid_argument("las dimensiones no coinciden");
	}
	
	size_t amountOfValues = 1;
	for (size_t n : a.shape) {
		amountOfValues *= n;
	}
	
	double result = 0.0;
	for (size_t i = 0; i < amountOfValues; i++) {
		result += a.data[i] * b.data[i];
	}
	
	std::vector<double> values = {result};
	return Tensor({1}, std::move(values));
}


Tensor matmul(const Tensor& a, const Tensor& b) {
	
	if (a.shape.size() != 2 || b.shape.size() != 2) {
		throw std::invalid_argument("los tensores deben ser bidimensionales");
	}
	
	size_t rows_a = a.shape[0]; size_t cols_a = a.shape[1];
	size_t rows_b = b.shape[0]; size_t cols_b = b.shape[1];
	
	if (cols_a != rows_b) {
		throw std::invalid_argument("dimensiones incompatibles");
	}
	
	std::vector<size_t> new_shape = {rows_a, cols_b};
	size_t amountOfValues = rows_a * cols_b;
	
	std::vector<double> values; values.reserve(amountOfValues);
	
	for (size_t i = 0; i < rows_a; i++) {
		for (size_t j = 0; j < cols_b; j++) {
			double sum = 0.0;
			for (size_t k = 0; k < cols_a; k++) {
				sum += a.data[i * cols_a + k] * b.data[k * cols_b + j];
			}
			values.push_back(sum);
		}
	}
	return Tensor(new_shape, std::move(values));    
}


Tensor Tensor::apply(const TensorTransform& tensorTransform) {
    
	size_t amountOfValues = 1;
	for (size_t n : shape) {
		amountOfValues *= n;
	}
	
	std::vector<double> new_values; new_values.reserve(amountOfValues);
	
	for (size_t i = 0; i < amountOfValues; i++) {
		new_values.push_back(tensorTransform.apply(this->data[i]));
	}
	return Tensor(this->shape, std::move(new_values));
}


void Tensor::print_dimensions() {
	for (size_t i = 0; i < shape.size(); i++) {
		std::cout << shape[i];
		if (i < shape.size() - 1) std::cout << " x ";
	}
	std::cout << std::endl;
}



