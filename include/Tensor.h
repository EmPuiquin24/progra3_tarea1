#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cmath>
#include <vector>

// TensorTrasform: Polimorfismo con métodos virtuales
class TensorTransform {
public:
	virtual double apply(double x) const = 0;
	virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
	double apply(double x) const override {
		return x > 0 ? x : 0;
	}
};

class Sigmoid : public TensorTransform {
public:
	double apply(double x) const override {
			return 1 / (1 + std::exp(-x));
	}
};

// Clase Tensor
class Tensor { 
private:
	double* data;
	int* ref_count;
	std::vector<size_t> shape;
	std::vector<double> values;
	

public:
	// Constructores
	Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);
	Tensor(double* data, int* ref_count, const std::vector<size_t>& shape);
	
	Tensor(Tensor&& otro) noexcept;
	Tensor(const Tensor& otro);	
	Tensor& operator=(const Tensor& otro);
	
	// Destructor >:(
	~Tensor();

	// Métodos estáticos
	static Tensor zeros(const std::vector<size_t>& shape);
	static Tensor ones(const std::vector<size_t>& shape);
	static Tensor random(const std::vector<size_t>& shape, const int& min, const int& max);
	static Tensor arange(const int& start, const int& end);

	// Sobrecarga de operadores
	Tensor operator+(const Tensor& otro);
	Tensor operator-(const Tensor& otro);
	Tensor operator*(const Tensor& otro);
	Tensor operator*(const double& n);

	// Método view, unsqueeze y concat
	Tensor view(const std::vector<size_t>& new_shape);
	Tensor unsqueeze(const size_t& position);
	static Tensor concat(const std::vector<Tensor>& tensors, const size_t& dimension);	

	// Funciones amigas
	friend Tensor dot(const Tensor& a, const Tensor& b);
	friend Tensor matmul(const Tensor& a, const Tensor& b);

	// Método apply
	Tensor apply(const TensorTransform& tensorTransform);
		
	// Imprimir dimensiones
	void print_dimensions();
};


#endif // TENSOR_H
