#pragma once
#include "Matrix.h"
#include "Filter.h"

template<typename T>
class Base_Cnn
{
public:
	// Конструкторы ----------------------------------------------------------
	Base_Cnn();
	Base_Cnn(const Base_Cnn& copy) = delete; // Запрет копирования

	// Методы класса ---------------------------------------------------------
	 // Добавление "полей" к матрице
	virtual void Padding(Matrix<T>&);

	// Операция "Макс пулинга"
	virtual Matrix<T> Pooling(const Matrix<T>&, const int&, const int&);

	// Операция свертки над матрицей значений
	virtual Matrix<T> Svertka(const Matrix<T>&, const Matrix<T>&) = 0;

	// Перегрузка операторов -------------------------------------------------
	Base_Cnn<T>& operator= (const Base_Cnn<T>& copy) = delete; // Запрет копирования
	
	// Деструктор ------------------------------------------------------------
	virtual ~Base_Cnn();

	// Класс исключения ------------------------------------------------------
	class Base_CnnExeption : public std::runtime_error {
	public:
		Base_CnnExeption(std::string str) : std::runtime_error(str) {};
		~Base_CnnExeption() {};
	};
};

template<typename T>
Base_Cnn<T>::Base_Cnn()
{
}

template<typename T>
Base_Cnn<T>::~Base_Cnn()
{
}

template<typename T>
Matrix<T> Base_Cnn<T>::Pooling(const Matrix<T>& a, const int& n_, const int& m_)
{
	if ((n_ < 0) || (m_ < 0) || (n_ > a.getN()) || (m_ > a.getM())) {
		throw Base_Cnn<T>::Base_CnnExeption("Неверный размер ядра!");
	}

	Matrix<T> copy(a.getN() / n_, a.getM() / m_);

	for (int i = 0; i < copy.getN(); i++) {
		for (int j = 0; j < copy.getM(); j++) {
			copy[i][j] = a.getPodmatrix(i*n_, j*m_, n_, m_).Max();
		}
	}
	return copy;
}


template<typename T>
void Base_Cnn<T>::Padding(Matrix<T>& a)
{
	Matrix<T> copy(a.getN() + 2, a.getM() + 2);

	for (int i = 0; i < copy.getN(); i++) {
		for (int j = 0; j < copy.getM(); j++) {
			if ((i == 0) || (j == 0) || (j == (copy.getM() - 1)) || (i == (copy.getN() - 1))) {
				copy[i][j] = 0;
			}
			else {
				copy[i][j] = a[i - 1][j - 1];
			}
		}
	}

	a = copy;
}
