#pragma once
#include "Matrix.h"
#include <iostream>
#include <iomanip>

namespace ANN {

	template<typename T>
	class Weights;

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Weights<T> &mat);

	template<typename T>
	std::istream &operator>>(std::istream &in, Weights<T> &mat);

	template<typename T>
	class Weights : public Matrix<T> {
	public:
		// Конструкторы ----------------------------------------------------------
		Weights(); // По умолчанию
		Weights(const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор (нулевая матрица)
		Weights(T **arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
		Weights(T *arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
		Weights(const Weights<T> &copy); // Копирования
		Weights(const Weights<T> &&copy); // Копирования

		// Методы класса ---------------------------------------------------------
		// Вывод весов на консоль в красивом виде
		void Out();

		// Получение доступа к d
		T &GetD() { return d; };

		const T &GetD() const { return d; };

		// Получение доступа к wbisas
		T &GetWBias() { return wbias; };

		const T &GetWBias() const { return wbias; };

		// Перегрузки операторов ------------------------
		Weights<T> &operator=(const Weights<T> &copy); // Оператор присваивания
		friend std::ostream &operator<<<>(std::ostream &out, const Weights<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>><>(std::istream &in, Weights<T> &mat); // Оператор чтение матрицы из потока


		// Деструктор ------------------------------------------------------------
		~Weights();

	protected:
		// Величина производной функции ошибки
		T d;

		// Вес нейрона сдвига
		T wbias;

	};

	template<typename T>
	Weights<T>::Weights() : Matrix<T>(), d(0), wbias(0) {
	}

	template<typename T>
	Weights<T>::Weights(const int &i_, const int &j_, const int &wbisas_) : Matrix<T>(i_, j_), d(0), wbias(wbisas_) {
	}

	template<typename T>
	Weights<T>::Weights(T **arr_, const int &i_, const int &j_, const int &wbisas_) : Matrix<T>(arr_, i_, j_), d(0),
																					  wbias(wbisas_) {
	}

	template<typename T>
	Weights<T>::Weights(const Weights<T> &copy) : Matrix<T>(copy), d(copy.GetD()), wbias(copy.wbias) {
	}

	template<typename T>
	inline void Weights<T>::Out() {
		for (int i = 0; i < this->n; i++) {
			for (int j = 0; j < this->m; j++) {
				std::cout << this->arr[i][j] << " ";
			}
			std::cout << std::endl;
		}
		if (wbias != 0) {
			std::cout << wbias << std::endl;
		}
	}

// inline void Weights<int>::Out()
// {
// 	using std::cout;
// 	int max = this->Max();
// 	int k = 2;
// 	while (max > 0) {
// 		k++;
// 		max = max / 10;
// 	}
// 	for (int i = 0; i < this->n; i++) {
// 		for (int j = 0; j < this->m; j++) {
// 			cout << std::setw(k) << this->arr[i][j];
// 		}
// 		cout << std::endl;
// 	}
// 	if (wbias != 0) {
// 		std::cout << std::setw(k) << wbias << std::endl;
// 	}
// }

// inline void Weights<double>::Out()
// {
// 	using std::cout;
// 	int max = (int)this->Max();
// 	int k = 2;
// 	while (max > 0) {
// 		k++;
// 		max = max / 10;
// 	}
// 	for (int i = 0; i < this->n; i++) {
// 		for (int j = 0; j < this->m; j++) {
// 			cout << std::setw(k+5) << std::setprecision(2)<< this->arr[i][j];
// 		}
// 		cout << std::endl;
// 	}
// 	if (wbias != 0) {
// 		std::cout << std::setw(k + 5) << std::setprecision(4) << wbias << std::endl;
// 	}
// }

	template<typename T>
	inline Weights<T> &Weights<T>::operator=(const Weights<T> &copy) {
        if (this == &copy) {
            return *this;
        }

        if ((copy.n > this->n) || (copy.m > this->m)) {
            if (this->n == 0 && this->m == 0) {
				this->n = copy.n;
				this->m = copy.m;
				this->initMat();
				d = copy.d;
				wbias = copy.wbias;
            } else {
				this->deinitMat();
				this->n = copy.n;
				this->m = copy.m;
				this->initMat();
				d = copy.d;
				wbias = copy.wbias;
            }
        } else {
			this->n = copy.n;
			this->m = copy.m;
			d = copy.d;
			wbias = copy.wbias;
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
				this->arr[i][j] = copy.arr[i][j];
            }
        }
		return *this;
	}

	template<typename T>
	Weights<T>::~Weights() {
	}

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Weights<T> &mat) {
		out << (Matrix<T>) mat;
		out << mat.d << ' ' << mat.wbias << std::endl;
		return out;
	}

	template<typename T>
	std::istream &operator>>(std::istream &in, Weights<T> &mat) {
		in >> ((Matrix<T> &) mat);
		in >> mat.d;
		in >> mat.wbias;
		return in;
	}

	template<typename T>
	Weights<T>::Weights(T *arr_, const int &i_, const int &j_, const int &wbisas_) : Matrix<T>(arr_, i_, j_),
																					 wbias(wbisas_), d(0) {

	}

	template<typename T>
	Weights<T>::Weights(const Weights<T> &&copy) : Matrix<T>(copy), wbias(copy.wbias), d(0) {

	}

}