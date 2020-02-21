#pragma once
#include "Weights.h"
#include "Filter.h"
#include "NeyronCnn.h"
#include <cstdlib>

template<typename T>
class CNNLearning
{
public:
	// Конструкторы ----------------------------------------------------------
	CNNLearning(const int& s_ = 1, const double& E_ = 1);

	// Методы класса ---------------------------------------------------------
	// Метод обратного распространения ошибки
	Matrix<T> ReversConvolution(const Matrix<T>& D, const Filter<T>& f);

	// Метод градиентного спуска
	void GradDes(const Matrix<T>& X, const Matrix<T>& D, Filter<T>& F);

	// Получение доступа к шагу свертки
	int& getStep() { return s; }

	// Метод получения доступа к кофиценту обучения
	double& getE() { return E; };

	// Операция обратного распространение ошибки c перцептрона на подвыборочный слой
	void Revers_Perceptron_to_CNN(Matrix<T>& a, const Weights<T>& w);

	// Операция обратного распространение ошибки на слое "Макс пулинга"
	Matrix<T> ReversPooling(const Matrix<T>& a, const int& n_, const int& m_);

	// Класс исключения ------------------------------------------------------
	class CNNLearningExeption : public std::runtime_error {
	public:
		CNNLearningExeption(std::string str) : std::runtime_error(str) {};
		~CNNLearningExeption() {};
	};

	// Деструктор ------------------------------------------------------------
	~CNNLearning();
private:
	NeyronCnn<T> neyron;
	int s;
	double E;
};

template<typename T>
inline CNNLearning<T>::CNNLearning(const int& s_, const double& E_):neyron(), s(s_), E(E_)
{
}

template<typename T>
inline Matrix<T> CNNLearning<T>::ReversConvolution(const Matrix<T>& D, const Filter<T>& f)
{
	if (s < 1) {
			throw CNNLearning<T>::CNNLearningExeption("Задан невозможный шаг свертки!");
	}
	auto F = f.roate_180();
	Matrix<T> O((D.getN() - 1) / s + f.getN(), (D.getM() - 1) / s + f.getM());
	if (s != 1) {
		int stepJ = 0, stepI = 0;
		int ii = 0, jj = 0;
		for (int i = 0; i < O.getN(); i++) {
			stepJ = 0;
			jj = 0;
			if (stepI) {
				for (int j = 0; j < O.getM(); j++) {
					O[i][j] = 0;
				}
				stepI--;
			}
			else {
				for (int j = 0; j < O.getM(); j++) {
					if (stepJ) {
						stepJ--;
						O[i][j] = 0;
					}
					else {
						O[i][j] = D[ii][jj++];
						stepJ = s;
					}
				}
				stepI = s;
			}
			ii++;
		}
	}
	else {
		O = D;
	}
	for (int i = 0; i < f.getN()-1; i++) {
		neyron.Padding(O);
	}
	return neyron.Svertka(F, O);
}

template<typename T>
void CNNLearning<T>::GradDes(const Matrix<T>& X, const Matrix<T>& D, Filter<T>& F) {
	Matrix<T> Delta = neyron.Svertka(D, X);
	if ((Delta.getN() != F.getN()) || (Delta.getM() != F.getM())) {
		throw typename CNNLearning<T>::CNNLearningExeption("Задана неверная размерность! После свертки размеры матрицы фильтра и матрицы ошибки не совпадают!");
	}
	T delt;
	for (int i = 0; i < Delta.getN(); i++) {
		for (int j = 0; j < Delta.getM(); j++) {
			delt = E * Delta[i][j];
			if (delt > 1000) {
				throw typename CNNLearning<T>::CNNLearningExeption("Слишком большая производная!");
			}
			F[i][j]-= delt;
		}
	}
}

template<typename T>
inline void CNNLearning<T>::Revers_Perceptron_to_CNN(Matrix<T>& a, const Weights<T>& w)
{
	if ((a.getN() < 0) || (a.getM() < 0)||(a.getN() != w.getN()) || (a.getM() != w.getM())) {
		throw typename Base_Cnn<T>::Base_CnnExeption("Неверный размер матрицы ошибки!");
	}
	for (int i = 0; i < w.getN(); i++) {
		for (int j = 0; j < w.getM(); j++) {
			a[i][j] += w.GetD() * w[i][j];
		}
	}
}

template<typename T>
inline Matrix<T> CNNLearning<T>::ReversPooling(const Matrix<T>& D, const int & n_, const int & m_)
{
	if ((n_ < 0) || (m_ < 0) || (n_ > D.getN()) || (m_ > D.getM())) {
		throw typename Base_Cnn<T>::Base_CnnExeption("Неверный размер ядра!");
	}

	Matrix<T> copy(D.getN() * n_, D.getM() * m_);

	for (int i = 0; i < D.getN(); i++) {
		for (int j = 0; j < D.getM(); j++) {
			for (int ii = i * n_; ii < i*n_ + n_; ii++) {
				for (int jj = j * m_; jj < j*m_ + m_; jj++) {
					copy[ii][jj] = D[i][j];
				}
			}
		}
	}
	return copy;
}


template<typename T>
inline CNNLearning<T>::~CNNLearning()
{
}
