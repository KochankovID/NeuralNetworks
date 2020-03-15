#pragma once
#include "Matrix.h"
#include "opencv2/opencv.hpp"
#include <iomanip>

namespace ANN {

	template<typename T>
	class Filter;

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Filter<T> &mat);

	template<typename T>
	std::istream &operator>>(std::istream &in, Filter<T> &mat);

	template<typename T>
	class Filter : public Matrix<T> {
	public:
		// Конструкторы ----------------------------------------------------------
		Filter(); // По умолчанию
		Filter(const int &i_, const int &j_); // Инициализатор (нулевая матрица)
		Filter(T **arr_, const int &i_, const int &j_); // Инициализатор
		Filter(T *arr_, const int &i_, const int &j_); // Инициализатор
		Filter(const Filter<T> &copy); // Копирования
		Filter(const Filter<T> &&copy); // Move

		// Методы класса ---------------------------------------------------------
		static Matrix<T> Padding(const Matrix <T> &a, size_t nums);
		static Matrix <T> Pooling(const Matrix <T> &a, int n_, int m_);
		static Matrix<T> Svertka(const Matrix<T> &a, const Matrix<T>& in, int step);
        Matrix<T> Svertka(const Matrix<T> &a, int step);
		// Поворот фильтра на 180
		Filter<T> roate_180() const;

		// Вывод фильтра на консоль в красивом виде
		void Out() const;

		// Перегрузки операторов ------------------------
		Filter<T> &operator=(const Filter<T> &copy); // Оператор присваивания
		friend std::ostream &operator<<<>(std::ostream &out, const Filter<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>><>(std::istream &in, Filter<T> &mat); // Оператор чтение матрицы из потока

		// Деструктор ------------------------------------------------------------
		~Filter<T>();

        // Класс исключения ------------------------------------------------------
        class Filter_Exeption : public std::logic_error {
        public:
            Filter_Exeption(std::string str) : std::logic_error(str) {};

            ~Filter_Exeption() {};
        };
	};

	template<typename T>
	Filter<T>::Filter() : Matrix<T>() {
	}

	template<typename T>
	Filter<T>::Filter(const int &i_, const int &j_) : Matrix<T>(i_, j_) {
	}

	template<typename T>
	Filter<T>::Filter(T **arr_, const int &i_, const int &j_) : Matrix<T>(arr_, i_, j_) {
	}

	template<typename T>
	Filter<T>::Filter(const Filter<T> &copy) : Matrix<T>(copy) {
	}

	template<typename T>
	inline Filter<T> Filter<T>::roate_180() const {
		Filter<T> F(this->n, this->m);
		for (int i = this->n - 1; i >= 0; i--) {
			for (int j = this->m - 1; j >= 0; j--) {
				F[i][j] = this->arr[this->n - 1 - i][this->m - 1 - j];
			}
		}
		return F;
	}

	template<typename T>
	inline Filter<T> &Filter<T>::operator=(const Filter<T> &copy) {
		if (this == &copy) {
			return *this;
		}
		if ((copy.n > this->n) || (copy.m > this->m)) {
			for (int i = 0; i < this->n; i++) {
				delete[] this->arr[i];
			}
			delete[] this->arr;
			this->n = copy.n;
			this->m = copy.m;
			this->initMat();
		} else {
			this->n = copy.n;
			this->m = copy.m;
		}

		for (int i = 0; i < this->n; i++) {
			for (int j = 0; j < this->m; j++) {
				this->arr[i][j] = copy.arr[i][j];
			}
		}
		return *this;
	}

	template<typename T>
	Filter<T>::~Filter() {
	}

	template<typename T>
	inline void Filter<T>::Out() const {
		for (int i = 0; i < this->n; i++) {
			for (int j = 0; j < this->m; j++) {
				std::cout << this->arr[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	template<typename T>
	Filter<T>::Filter(const Filter<T> &&copy) : Matrix<T>(copy) {}

	template<typename T>
	Filter<T>::Filter(T *arr_, const int &i_, const int &j_) : Matrix<T>(arr_, i_, j_) {

	}

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Filter<T> &mat) {
		out << (Matrix<T>) mat;
		return out;
	}

	template<typename T>
	std::istream &operator>>(std::istream &in, Filter<T> &mat) {
		in >> ((Matrix<T> &) mat);
		return in;
	}

    template<typename T>
    Matrix<T> Filter<T>::Padding(const Matrix <T> &a, size_t nums) {
        // Создаем результирующую матрицу
        Matrix<T> copy(a.getN() + 2*nums, a.getM() + 2*nums);

        // Если мы на границе - то заполняем элемент матрицы нулем, если нет, то копируем матрицу
        for (int i = 0; i < copy.getN(); i++) {
            for (int j = 0; j < copy.getM(); j++) {
                if ((i < nums) || (j < nums)||(i>=a.getN()+nums)||(j>=a.getM()+nums)) {
                    copy[i][j] = 0;
                } else {
                    copy[i][j] = a[i - nums][j - nums];
                }
            }
        }

        // Возвращаем результирующую матрицу
        return copy;
    }

    template<typename T>
    Matrix <T> Filter<T>::Pooling(const Matrix <T> &a, int n_, int m_) {
        // Проверяем размер ядра
        if ((n_ <= 0) || (m_ <= 0) || (n_ > a.getN()) || (m_ > a.getM())) {
            throw Filter<T>::Filter_Exeption("Неверный размер ядра!");
        }

        // Создаем результирующую матрицу
        Matrix<T> copy(a.getN() / n_, a.getM() / m_);

        // Выбираем максимальный элемент из полученной подматрицы
        for (int i = 0; i < copy.getN(); i++) {
            for (int j = 0; j < copy.getM(); j++) {
                copy[i][j] = a.getPodmatrix(i * n_, j * m_, n_, m_).Max();
            }
        }

        // Возвращаем результирующую матрицу
        return copy;
    }

    template<typename T>
    Matrix<T> Filter<T>::Svertka(const Matrix <T> &a, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getN()) || (step > a.getM()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }

        // Создание результирующей матрицы
        Matrix<T> rez((a.getN() - this->n) / step + 1, (a.getM() - this->m) / step + 1);

        cv::parallel_for_(cv::Range(0, rez.getN()), [&](const cv::Range &range) {
            for (int i = range.start; i < range.end; i++) {
                for (int j = 0; j < rez.getM(); j++) {

                    // Переменная в которой хранится текущая сумма свертки
                    double sum;

                    // Начало поэлементного умножения
                    sum = 0;

                    // Вычисление суммы
                    for (int ii = 0; ii < this->n; ii++) {
                        for (int jj = 0; jj < this->m; jj++) {
                            sum += a[i * step + ii][j * step + jj] * (*this)[ii][jj];
                        }
                    }
                    rez[i][j] = sum;
                }
            }
        });

        return rez;
    }

    template<typename T>
    Matrix<T> Filter<T>::Svertka(const Matrix<T> &a, const Matrix<T>& in, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getN()) || (step > a.getM()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((in.getN() < a.getN())||(in.getM() < a.getN())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }

        // Создание результирующей матрицы
        Matrix<T> rez((a.getN() - in.getN()) / step + 1, (a.getM() - in.getM()) / step + 1);

        cv::parallel_for_(cv::Range(0, rez.getN()), [&](const cv::Range &range) {
            for (int i = range.start; i < range.end; i++) {
                for (int j = 0; j < rez.getM(); j++) {

                    // Переменная в которой хранится текущая сумма свертки
                    double sum;

                    // Начало поэлементного умножения
                    sum = 0;

                    // Вычисление суммы
                    for (int ii = 0; ii < in.getN(); ii++) {
                        for (int jj = 0; jj < in.getM(); jj++) {
                            sum += a[i * step + ii][j * step + jj] * in[ii][jj];
                        }
                    }
                    rez[i][j] = sum;
                }
            }
        });

        return rez;
    }

}