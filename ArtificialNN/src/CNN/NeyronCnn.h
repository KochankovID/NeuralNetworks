#ifndef BASE_CNN
#define BASE_CNN

#include "Base_Cnn.h"
#include <string>


template <typename T>
class NeyronCnn : public Base_Cnn<T>
{
public:
	// Конструкторы ----------------------------------------------------------
	NeyronCnn();
	explicit NeyronCnn(const int &step_);
	NeyronCnn(const NeyronCnn<T> &copy) = delete; // Запрет копирования

	// Методы класса ---------------------------------------------------------
	// Операция свертки над матрицей значений
	Matrix<T> Svertka(const Matrix<T> &F, const Matrix<T> &a);

	// Получение доступа к шагу свертки
	int &GetStep() { return step; }

	// Перегрузка операторов -------------------------------------------------
	NeyronCnn &operator=(const NeyronCnn<T> &copy) = delete; // Запрет копирования

	// Деструктор ------------------------------------------------------------
	~NeyronCnn();

	// Класс исключения ------------------------------------------------------
	class NeyronCnnExeption : public Base_Cnn<T>::Base_CnnExeption
	{
	public:
		NeyronCnnExeption(std::string str) : Base_Cnn<T>::Base_CnnExeption(str){};
		~NeyronCnnExeption(){};
	};

private:
	// Поля класса -----------------------------------------------------------
	int step; // Шаг свертки
};

template <typename T>
NeyronCnn<T>::NeyronCnn() : Base_Cnn<T>(), step(1)
{
}

template <typename T>
NeyronCnn<T>::NeyronCnn(const int &step_) : Base_Cnn<T>(), step(step_)
{
}

template <typename T>
Matrix<T> NeyronCnn<T>::Svertka(const Matrix<T> &F, const Matrix<T> &a)
{

	if ((step > a.getN()) || (step > a.getM()) || (step < 1))
	{
		throw NeyronCnnExeption("Задан невозможный шаг свертки!");
	}

	Matrix<T> rez((a.getN() - F.getN()) / step + 1, (a.getM() - F.getM()) / step + 1);

	double sum;
	Matrix<T> fokus;
	for (int i = 0; i < rez.getN(); i++)
	{
		for (int j = 0; j < rez.getM(); j++)
		{
			sum = 0;
			fokus = a.getPodmatrix(i * step, j * step, F.getN(), F.getM());
#pragma omp parallel for reduction(+:sum)
			for (int ii = 0; ii < F.getN(); ii++)
			{
				for (int jj = 0; jj < F.getM(); jj++)
				{
					sum += fokus[ii][jj] * F[ii][jj];
				}
			}
			rez[i][j] = sum;
		}
	}
	return rez;
}

template <typename T>
NeyronCnn<T>::~NeyronCnn()
{
}
#endif