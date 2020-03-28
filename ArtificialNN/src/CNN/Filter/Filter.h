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

		static Matrix<T> Svertka(const Tensor<T> &a, const Tensor<T>& filter, int step);
        Matrix<T> Svertka(const Tensor<T> &a, int step);

		// Поворот фильтра на 180
		Filter<T> roate_180() const;

		// Вывод фильтра на консоль в красивом виде
		void Out() const;

        static std::pair<size_t, size_t> convolution_result_creation(size_t mat_h, size_t mat_w,
                                                                     size_t filter_h, size_t filter_w, size_t step){
            return std::make_pair(((mat_h - filter_h) / step + 1), ((mat_w - filter_w) / step + 1));
        }

		// Перегрузки операторов ------------------------
		Filter<T> &operator=(const Filter<T> &copy); // Оператор присваивания
		friend std::ostream &operator<< <>(std::ostream &out, const Filter<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>> <>(std::istream &in, Filter<T> &mat); // Оператор чтение матрицы из потока
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

		this->Tensor<T>::operator=(copy);
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
    Matrix<T> Filter<T>::Svertka(const Tensor<T> &a, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getHeight()) || (step > a.getWidth()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((this->getHeight() > a.getHeight())||(this->getWidth() > a.getWidth())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }
        // Создание результирующей матрицы
        auto size = convolution_result_creation(a.getHeight(), a.getWidth(),
                                                this->getHeight(), this->getWidth(), step);
        Matrix<T> rez(size.first, size.second);

        for(size_t h = 0; h < rez.getN(); h++){
            for(size_t w = 0; w < rez.getM(); w++){
                // Переменная в которой хранится текущая сумма свертки
                T sum = 0;

                // Вычисление суммы
                for(size_t z = 0; z < this->getDepth(); z++) {
                    for (size_t x = 0; x < this->getHeight(); x++) {
                        for (size_t y = 0; y < this->getWidth(); y++) {
                            sum += a[z][h * step + x][w * step + y] * (*this)[z][x][y];
                        }
                    }
                }
                rez[h][w] = sum;

            }
        }

        return rez;
    }

    template<typename T>
    Matrix<T> Filter<T>::Svertka(const Tensor<T> &a, const Tensor<T>& filter, int step) {
        // Проверка правильности задания шага свертки
        if ((step > a.getHeight()) || (step > a.getWidth()) || (step < 1)) {
            throw Filter<T>::Filter_Exeption("Задан невозможный шаг свертки!");
        }
        if((filter.getHeight() > a.getHeight())||(filter.getWidth() > a.getWidth())){
            throw Filter<T>::Filter_Exeption("Сворачиваемая матрица меньше ядра свертки!");
        }
        // Создание результирующей матрицы
        auto size = convolution_result_creation(a.getHeight(), a.getWidth(),
                                                filter.getHeight(), filter.getWidth(), step);
        Matrix<T> rez(size.first, size.second);

        for(size_t h = 0; h < rez.getN(); h++){
            for(size_t w = 0; w < rez.getM(); w++){
                // Переменная в которой хранится текущая сумма свертки
                T sum = 0;

                // Вычисление суммы
                for(size_t z = 0; z < filter.getDepth(); z++) {
                    for (size_t x = 0; x < filter.getHeight(); x++) {
                        for (size_t y = 0; y < filter.getWidth(); y++) {
                            sum += a[z][h * step + x][w * step + y] * filter[z][x][y];
                        }
                    }
                }
                rez[h][w] = sum;

            }
        }
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