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
	class Filter : public Tensor<T> {
	public:
		// Конструкторы ----------------------------------------------------------
		Filter(); // По умолчанию
		Filter(int height, int wight, int depth = 1); // Инициализатор (нулевая матрица)
		Filter(const Filter<T> &copy); // Копирования
		Filter(const Filter<T> &&copy); // Move

		// Методы класса ---------------------------------------------------------
		static Matrix<T> Padding(const Matrix <T> &a, size_t nums);
		static Matrix<T> Pooling(const Matrix <T> &a, int n_, int m_);

        static Tensor<T> Padding(const Tensor <T> &a, size_t nums);
        static Tensor<T> Pooling(const Tensor <T> &a, int n_, int m_);

		static Tensor<T> Svertka(const Tensor<T> &a, const Tensor<T>& filter, int step);
        Tensor<T> Svertka(const Tensor<T> &a, int step);
		// Поворот фильтра на 180
		Filter<T> roate_180() const;

		// Вывод фильтра на консоль в красивом виде
		void Out() const;

		// Перегрузки операторов ------------------------
		Filter<T> &operator=(const Filter<T> &copy); // Оператор присваивания
		friend std::ostream &operator<< <>(std::ostream &out, const Filter<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>> <>(std::istream &in, Filter<T> &mat); // Оператор чтение матрицы из потока
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
	Filter<T>::Filter() : Tensor<T>() {
	}

	template<typename T>
	Filter<T>::Filter(int height, int wight, int depth) : Tensor<T>(height, wight, depth) {
	}

	template<typename T>
	Filter<T>::Filter(const Filter<T> &copy) : Tensor<T>(copy) {
	}

	template<typename T>
	inline Filter<T> Filter<T>::roate_180() const {
		Filter<T> F(this->arr[0][0].getN(), this->arr[0][0].getM(), this->getDepth());

		for(size_t k = 0; k < this->getDepth(); k++) {
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
	Filter<T>::Filter(const Filter<T> &&copy) : Tensor<T>(copy) {}

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
    Tensor<T> Filter<T>::Svertka(const Tensor<T> &a, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getHeight()) || (step > a.getWight()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((this->getHeight() > a.getHeight())||(this->getWight() > a.getWight())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }
        T n = this->getHeight();
        T m = this->getWight();
        // Создание результирующей матрицы
        Tensor<T> rez((a.getHeight() - n) / step + 1, (a.getWight() - m) / step + 1, this->getDepth());
        for(size_t i = 0; i < this->getDepth(); i++){
            rez[i] = Matrix<T>((a[0].getN() - (*this)[i].getN()) / step + 1,
                                  (a[0].getM() - (*this)[i].getM()) / step + 1);
        }
        for(size_t d = 0;  d < this->getDepth(); d++){
            for(size_t h = 0; h < rez[d].getN(); h++){
                for(size_t w = 0; w < rez[d].getM(); w++){
                    // Переменная в которой хранится текущая сумма свертки
                    T sum = 0;

                    // Вычисление суммы
                    for (size_t x = 0; x < n; x++) {
                        for (size_t y = 0; y < m; y++) {
                            sum += a[d][h * step + x][w * step + y] * (*this)[d][x][y];
                        }
                    }
                    rez[d][h][w] = sum;
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
    Tensor<T> Filter<T>::Svertka(const Tensor<T> &a, const Tensor<T>& filter, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getHeight()) || (step > a.getWight()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((filter.getHeight() > a.getHeight())||(filter.getWight() > a.getWight())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }
        T n = filter.getHeight();
        T m = filter.getWight();
        // Создание результирующей матрицы
        Tensor<T> rez((a.getHeight() - n) / step + 1, (a.getWight() - m) / step + 1, filter.getDepth());
        for(size_t i = 0; i < rez.getDepth(); i++){
            rez[i] = Matrix<T>((a[0].getN() - filter[i].getN()) / step + 1,
                                  (a[0].getM() - filter[i].getM()) / step + 1);
        }
        for(size_t d = 0;  d < filter.getDepth(); d++){
            for(size_t h = 0; h < rez[d].getN(); h++){
                for(size_t w = 0; w < rez[d].getM(); w++){
                    // Переменная в которой хранится текущая сумма свертки
                    T sum = 0;

                    // Вычисление суммы
                    for (size_t x = 0; x < n; x++) {
                        for (size_t y = 0; y < m; y++) {
                            sum += a[d][h * step + x][w * step + y] * filter[d][x][y];
                        }
                    }
                    rez[d][h][w] = sum;
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
    Tensor<T> Filter<T>::Padding(const Tensor<T> &a, size_t nums) {
        Tensor<T> rez(a.getN() + 2*nums, a.getM() + 2*nums, a.getDepth());

        for(size_t i = 0; i< a.getDepth(); i++){
            rez[i] = Filter<T>::Padding(a[i], nums);
        }

        return rez;
    }

    template<typename T>
    Tensor<T> Filter<T>::Pooling(const Tensor<T> &a, int n_, int m_) {
        Tensor<T> rez(a.getN() / n_, a.getM() / m_, a.getDepth());

        for(size_t i = 0; i < a.getDepth(); i++){
            rez[i] = Filter<T>::Pooling(rez[i], n_, m_);
        }

        return rez;
    }

}