#pragma once
#include "Matrix.h"
#include "opencv2/opencv.hpp"
#include <iomanip>
#include <Tensor.h>


namespace ANN {

	template<typename T>
	class Filter;

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Filter<T> &mat);

	template<typename T>
	std::istream &operator>>(std::istream &in, Filter<T> &mat);

	template<typename T>
	class Filter : public Matrix<Matrix<T> > {
	public:
		// Конструкторы ----------------------------------------------------------
		Filter(); // По умолчанию
		Filter(int height, int wight, int depth = 1); // Инициализатор (нулевая матрица)
		Filter(const Filter<T> &copy); // Копирования
		Filter(const Filter<T> &&copy); // Move

		// Методы класса ---------------------------------------------------------
		static Matrix<T> Padding(const Matrix <T> &a, size_t nums);
		static Matrix<T> Pooling(const Matrix <T> &a, int n_, int m_);
		static Matrix<Matrix<T>> Svertka(const Matrix<Matrix<T>> &a, const Matrix<Matrix<T>>& filter, int step);
        Matrix<Matrix<T>> Svertka(const Matrix<Matrix<T>> &a, int step);
		// Поворот фильтра на 180
		Filter<T> roate_180() const;

		// Вывод фильтра на консоль в красивом виде
		void Out() const;

		// Перегрузки операторов ------------------------
		Filter<T> &operator=(const Filter<T> &copy); // Оператор присваивания
		friend std::ostream &operator<<<>(std::ostream &out, const Filter<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>><>(std::istream &in, Filter<T> &mat); // Оператор чтение матрицы из потока
        Matrix<T>& operator[](int index); // Оператор индексации
        const Matrix<T>& operator[](int index) const; // Оператор индексации константы
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
	Filter<T>::Filter() : Matrix<Matrix<T>>() {
	}

	template<typename T>
	Filter<T>::Filter(int height, int wight, int depth) : Matrix<Matrix<T> >(1, depth) {
	    if((height < 0) || (wight < 0) || (depth < 0)){
	        throw Filter_Exeption("Wrong shape of filter!");
	    }
	    for(size_t i = 0; i < depth; i++){
            this->arr[0][i] = Matrix<T>(height, wight);
	    }
	}

	template<typename T>
	Filter<T>::Filter(const Filter<T> &copy) : Matrix<Matrix<T> >(copy) {
	}

	template<typename T>
	inline Filter<T> Filter<T>::roate_180() const {
		Filter<T> F(this->arr[0][0].getN(), this->arr[0][0].getM(), this->getM());
		for(size_t k = 0; k < this->getM(); k++) {
            for (int i = (*this)[k].getN() - 1; i >= 0; i--) {
                for (int j = (*this)[k].getM() - 1; j >= 0; j--) {
                    F[k][i][j] = (*this)[k][(*this)[k].getN() - 1 - i][(*this)[k].getM() - 1 - j];
                }
            }
        }
		return F;
	}

	template<typename T>
	inline Filter<T> &Filter<T>::operator=(const Filter<T> &copy) {
		if (this == &copy) {
			return *this;
		}
        for (int i = 0; i < this->n; i++) {
            delete[] this->arr[i];
        }
        delete[] this->arr;
        this->n = copy.n;
        this->m = copy.m;
        this->initMat();

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
	Filter<T>::Filter(const Filter<T> &&copy) : Matrix<Matrix<T>>(copy) {}

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Filter<T> &mat) {
		out << (Matrix<Matrix<T>>) mat;
		return out;
	}

	template<typename T>
	std::istream &operator>>(std::istream &in, Filter<T> &mat) {
		in >> ((Matrix<Matrix<T>> &) mat);
		return in;
	}
    template<typename T>
    const Matrix <T>& Filter<T>::operator[](int index) const {
        this->isInRange(index);
        return this->arr[0][index];
    }

    template<typename T>
    Matrix<T>& Filter<T>::operator[](int index) {
        this->isInRange(index);
        return this->arr[0][index];
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
    Matrix<Matrix<T>> Filter<T>::Svertka(const Matrix<Matrix<T>> &a, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a[0][0].getN()) || (step > a[0][0].getM()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((this->arr[0][0].getN() > a[0][0].getN())||(this->arr[0][0].getM() > a[0][0].getN())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }
        T n = this->arr[0][0].getN();
        T m = this->arr[0][0].getM();
        // Создание результирующей матрицы
        Matrix<Matrix<T> > rez(1, this->m);
        for(size_t i = 0; i < this->m; i++){
            rez[0][i] = Matrix<T>((a[0][0].getN() - (*this)[i].getN()) / step + 1,
                                  (a[0][0].getM() - (*this)[i].getM()) / step + 1);
        }
        for(size_t d = 0;  d < this->m; d++){
            for(size_t h = 0; h < rez[0][d].getN(); h++){
                for(size_t w = 0; w < rez[0][d].getM(); w++){
                    // Переменная в которой хранится текущая сумма свертки
                    T sum = 0;

                    // Вычисление суммы
                    for (size_t x = 0; x < n; x++) {
                        for (size_t y = 0; y < m; y++) {
                            sum += a[0][d][h * step + x][w * step + y] * (*this)[d][x][y];
                        }
                    }
                    rez[0][d][h][w] = sum;
                }
            }
        }
//        cv::parallel_for_(cv::Range(0, rez.getN()), [&](const cv::Range &range) {
//            for (int i = range.start; i < range.end; i++) {
//                for (int j = 0; j < rez.getM(); j++) {
//
//                    // Переменная в которой хранится текущая сумма свертки
//                    double sum;
//
//                    // Начало поэлементного умножения
//                    sum = 0;
//
//                    // Вычисление суммы
//                    for (int ii = 0; ii < this->n; ii++) {
//                        for (int jj = 0; jj < this->m; jj++) {
//                            sum += a[i * step + ii][j * step + jj] * (*this)[ii][jj];
//                        }
//                    }
//                    rez[i][j] = sum;
//                }
//            }
//        });

        return rez;
    }

    template<typename T>
    Matrix<Matrix<T>> Filter<T>::Svertka(const Matrix<Matrix<T>> &a, const Matrix<Matrix<T>>& filter, int step) {
        if(a.getM() != filter.getM()){
            throw Filter<T>::Filter_Exeption("Depth matrix and filter is different!");

        }
        // Проверка правильности задания шага свертки
        if ((step > a[0][0].getN()) || (step > a[0][0].getM()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((filter[0][0].getN() > a.getN())||(filter[0][0].getM() > a.getN())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }

        T n = filter[0][0].getN();
        T m = filter[0][0].getM();

        // Создание результирующей матрицы
        Matrix<Matrix<T> > rez(1, filter.getM());
        for(size_t i = 0; i < filter.getM(); i++){
            rez[0][i] = Matrix<T>((a[0][0].getN() - filter[0][i].getN()) / step + 1,
                    (a[0][0].getM() - filter[0][i].getM()) / step + 1);
        }

        for(size_t d = 0;  d < filter.getM(); d++){
            for(size_t h = 0; h < rez[0][d].getN(); h++){
                for(size_t w = 0; w < rez[0][d].getM(); w++){
                    // Переменная в которой хранится текущая сумма свертки
                    T sum = 0;

                    // Вычисление суммы
                    for (size_t x = 0; x < n; x++) {
                        for (size_t y = 0; y < m; y++) {
                            sum += a[0][d][h * step + x][w * step + y] * filter[0][d][x][y];
                        }
                    }

                    rez[0][d][h][w] = sum;
                }
            }
        }

//        cv::parallel_for_(cv::Range(0, rez.getN()), [&](const cv::Range &range) {
//            for (int i = range.start; i < range.end; i++) {
//                for (int j = 0; j < rez.getM(); j++) {
//
//                    // Переменная в которой хранится текущая сумма свертки
//                    double sum;
//
//                    // Начало поэлементного умножения
//                    sum = 0;
//
//                    // Вычисление суммы
//                    for (int ii = 0; ii < filter.getN(); ii++) {
//                        for (int jj = 0; jj < filter.getM(); jj++) {
//                            sum += a[i * step + ii][j * step + jj] * filter[ii][jj];
//                        }
//                    }
//                    rez[i][j] = sum;
//                }
//            }
//        });

        return rez;
    }

}