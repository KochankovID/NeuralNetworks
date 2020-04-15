#pragma once
#include <string>
#include <memory>
#include <iostream>
#include <math.h>
#include <algorithm>

namespace ANN {

	template<typename T>
	class Matrix;

	template<typename T>
	Matrix<T> operator*(const int k, const Matrix<T> &mat);

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Matrix<T> &mat);

	template<typename T>
	std::istream &operator>>(std::istream &out, Matrix<T> &mat);

	template<typename T>
	class Matrix {
	public:
		// Конструкторы ---------------------------------
		Matrix(); // Конструктор по умолчанию -----------
		Matrix(T **arr_, const int &i, const int &j); // Конструктор инициализатор
		Matrix(T *arr_, const int &i, const int &j); // Конструктор инициализатор
		Matrix(const int &i,
			   const int &j); // Конструктор инициализатор (создает матрицу заданного размера заполненную 0)
		Matrix(const Matrix<T> &copy); // Конструктор копирования
		Matrix(Matrix<T> &&copy); // Конструктор move

		// Методы класса --------------------------------
		// Получение количества строк
		int getN() const {
			return n;
		}

		// Получение колисчества столбцов
		int getM() const {
			return m;
		}

		// Поиск максимума в матрице
		T Max() const;

		// Получение копии матрицы в виде массива
		T **getCopy();

		// Получение среднего значения элементов матрицы
		T mean();

		// Масштабирование матрицы
		Matrix<T> zoom(int place) const;

		// Заполнение матрицы заданным значением
		void Fill(const T &a);

		// Получение подматрицы
		Matrix<T> getPodmatrix(const int &poz_n_, const int &poz_m_, const int &n_, const int &m_) const;

		// Перегрузки операторов ------------------------
		Matrix<T> &operator=(const Matrix<T> &copy); // Оператор присваивания
		Matrix<T> &operator+=(const Matrix<T> &mat); // Оператор присваивания
		Matrix<T> operator+(const Matrix<T> &mat) const; // Оператор суммы
		friend Matrix operator*<>(const int k, const Matrix<T> &mat); // Оператор произведения на число
		Matrix<T> operator*(const Matrix<T> &mat) const; // Оператор произведения
		Matrix<T> operator*(const int k) const; // Оператор произведения на число
		friend std::ostream &operator<<<>(std::ostream &out, const Matrix<T> &mat); // Оператор вывод матрицы в поток
		friend std::istream &operator>><>(std::istream &out, Matrix<T> &mat); // Оператор чтение матрицы из потока
		T* operator[] (int index); // Оператор индексации
		const T *operator[](int index) const; // Оператор индексации константы
		bool operator==(const Matrix<T> &mat) const; // Оператор сравнения матриц


		// Деструктор -----------------------------------
		virtual ~Matrix();

		// Класс исключений ----------------------------
		class MatrixExeption : public std::runtime_error {
		public:
			MatrixExeption(std::string s) : std::runtime_error(s) {}

			~MatrixExeption() {}
		};

	protected:

		// Поля класса ----------------------------------
		int n, // Количество строк в матрице
		m; // Количество столбцов с матрице
		T **arr; // Матрица

		// Скрытые матоды класса ------------------------
		void initMat(); // Выделение памяти для матрицы
		void deinitMat(); // Удаление памяти матрицы
		void isInRange(int index) const; // Проверяет, находится ли индекс в допустимых границах
	};

	template <typename T>
	int getIndexOfMaxElem(T* first, T* last);

// Реализация ---------------------------------------
	template<typename T>
	Matrix<T>::Matrix() : n(0), m(0) {
		arr = nullptr;
	}

	template<typename T>
	Matrix<T>::Matrix(T **arr_, const int &i, const int &j) : n(i), m(j) {
		if ((n < 0) || (m < 0)) {
			throw Matrix::MatrixExeption("Неверный размер матрицы!");
		}
		initMat();
		try {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					arr[i][j] = arr_[i][j];
				}
			}
		} catch (...) {
			deinitMat();
			throw std::logic_error("Error while initialize matrix!");
		}
	}

	template<typename T>
	Matrix<T>::Matrix(T *arr_, const int &i, const int &j) : n(i), m(j) {
		if ((n < 0) || (m < 0)) {
			throw Matrix::MatrixExeption("Wrong size of matrix");
		}
		initMat();
		try {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					arr[i][j] = arr_[i * m + j];
				}
			}
		} catch (...) {
			deinitMat();
			throw std::logic_error("Error while initialize matrix!");
		}
	}

	template<typename T>
	Matrix<T>::Matrix(const int &i, const int &j) : n(i), m(j) {
		if ((n < 0) || (m < 0)) {
			throw Matrix::MatrixExeption("Неверный размер матрицы!");
		}
		initMat();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				arr[i][j] = T();
			}
		}
	}

	template<typename T>
	Matrix<T>::Matrix(const Matrix<T> &copy) : n(copy.n), m(copy.m) {
		initMat();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				arr[i][j] = copy.arr[i][j];
			}
		}
	}

	template<typename T>
	Matrix<T> Matrix<T>::getPodmatrix(const int &poz_n_, const int &poz_m_, const int &n_, const int &m_) const {
		if ((poz_n_ < 0) || (poz_m_ < 0) || (poz_n_ >= n) || (poz_m_ >= m)) {
			throw Matrix::MatrixExeption("Неверная позиция верхнего левого элемента подматрицы!");
		}
		if (((poz_n_ + n_) > n) || ((poz_m_ + m_) > m)) {
			throw Matrix::MatrixExeption("Подматрица выходит за границы матрицы!");
		}
		if ((n_ < 0) || (m_ < 0)) {
			throw Matrix::MatrixExeption("Подматрица выходит за границы матрицы!");
		}
		if ((n_ == 0) || (m_ == 0)) {
			Matrix<T> rez(0, 0);
			return rez;
		}

		Matrix<T> rez(n_, m_);

		for (int i = 0; i < n_; i++) {
			for (int j = 0; j < m_; j++) {
				rez[i][j] = arr[poz_n_ + i][poz_m_ + j];
			}
		}
		return rez;
	}

	template<typename T>
	T Matrix<T>::Max() const {
		T max = arr[0][0];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (arr[i][j] > max) {
					max = arr[i][j];
				}
			}
		}
		return max;
	}

	template<typename T>
	T **Matrix<T>::getCopy() {
		T **copy;
		copy = new T *[n];
		for (int i = 0; i < n; i++) {
			copy[i] = new T[m];
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				copy[i][j] = arr[i][j];
			}
		}
		return copy;
	}

	template<typename T>
	void Matrix<T>::Fill(const T &a) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				arr[i][j] = a;
			}
		}
	}

	template<typename T>
	Matrix<T> &Matrix<T>::operator=(const Matrix<T> &copy) {
		if (this == &copy) {
			return *this;
		}

        if (n == 0 && m == 0) {
            n = copy.n;
            m = copy.m;
            initMat();
        } else {
            deinitMat();
            n = copy.n;
            m = copy.m;
            initMat();
        }
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				arr[i][j] = copy.arr[i][j];
			}
		}
		return *this;
	}

	template<typename T>
	Matrix<T> Matrix<T>::operator+(const Matrix<T> &mat) const {
		Matrix<T> tmp(*this);
		if ((n != mat.n) || (m != mat.m)) {
			throw MatrixExeption("Невозможно выполнить сложение матриц разного размера");
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp[i][j] += mat.arr[i][j];
			}
		}
		return tmp;
	}

	template<typename T>
	Matrix<T> Matrix<T>::operator*(const Matrix<T> &mat) const {
		if (m != mat.n) {
			throw MatrixExeption(
					"Невозможно выполнить умножение матриц с несовпадающим количеством столбцов в первой и строк во второй");
		}
		Matrix<T> tmp(n, mat.m);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < mat.m; j++) {
				for (int o = 0; o < m; o++) {
					tmp[i][j] += (arr[i][o] * mat.arr[o][j]);
				}
			}
		}
		return tmp;
	}

	template<typename T>
	Matrix<T> Matrix<T>::operator*(const int k) const {
		Matrix<T> tmp(*this);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp[i][j] *= k;
			}
		}
		return tmp;
	}

	template<typename T>
    T* Matrix<T>::operator[] (int index) {
		isInRange(index);
		return arr[index];
	}

	template<typename T>
	const T *Matrix<T>::operator[](int index) const {
		isInRange(index);
		return arr[index];
	}

	template<typename T>
	bool Matrix<T>::operator==(const Matrix<T> &mat) const {
		if ((n != mat.n) || (m != mat.m)) {
			return false;
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (arr[i][j] != mat.arr[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	template<typename T>
	Matrix<T>::~Matrix() {
		if (n == 0 && m == 0) {
			return;
		} else {
			deinitMat();
		}
	}

	template<typename T>
	void Matrix<T>::initMat() {
		arr = new T *[n];
		for (int i = 0; i < n; i++) {
			arr[i] = new T[m];
		}
	}

	template<typename T>
	void Matrix<T>::deinitMat() {
		for (int i = 0; i < n; i++) {
			delete[] arr[i];
		}
		delete[] arr;
	}

	template<typename T>
	void Matrix<T>::isInRange(int index) const {
		if ((index >= n) || (index < 0)) {
			throw MatrixExeption("Индекс выходит за размер матрицы!");
		}
	}

	template<typename T>
	Matrix<T> operator*(const int k, const Matrix<T> &mat) {
		Matrix<T> tmp(mat);
		for (int i = 0; i < mat.n; i++) {
			for (int j = 0; j < mat.m; j++) {
				tmp[i][j] *= k;
			}
		}
		return tmp;
	}

	template<typename T>
	std::ostream &operator<<(std::ostream &out, const Matrix<T> &mat) {
		out << mat.n << ' ' << mat.m << std::endl; // Для совместимости с вводом из файла

		for (int i = 0; i < mat.n; i++) {
			for (int j = 0; j < mat.m; j++) {
				out << mat.arr[i][j] << ' ';
			}
			out << std::endl;
		}
		return out;
	}

	template<typename T>
	std::istream &operator>>(std::istream &in, Matrix<T> &mat) {
	    mat.deinitMat();
		in >> mat.n;
		in >> mat.m;
		if ((mat.n < 0) || (mat.m < 0)) {
			throw typename Matrix<T>::MatrixExeption("Неверный размер матрицы!");
		}
		mat.initMat();
		for (int i = 0; i < mat.n; i++) {
			for (int j = 0; j < mat.m; j++) {
				in >> mat.arr[i][j];
			}
		}
		return in;
	}

	template<typename T>
	Matrix<T>::Matrix(Matrix<T> &&copy) : n(copy.n), m(copy.m) {
		arr = copy.arr;
		copy.arr = nullptr;
		copy.n = 0;
		copy.m = 0;
	}

	template <typename T>
	T Matrix<T>::mean(){
		T mean = T();
		for(size_t i = 0; i < this->n; i++){
			for(size_t j = 0; j < this->m; j++){
				mean += this->arr[i][j];
			}
		}
		return mean / (n*m);
	}

	template<typename T>
	Matrix<T> Matrix<T>::zoom(int place) const {
		if(place <= 0){
            throw Matrix<T>::MatrixExeption("Неверный размер свободного пространства!");
		}
		Matrix<T> tmp(this->n + (n-1) * place, this->m + (m-1) * place);
		tmp.Fill(0);
		for(size_t i = 0; i< tmp.getN(); i+=place+1){
			for(size_t j = 0; j < tmp.getM(); j+=place+1){
				tmp[i][j] = this->arr[i/(2*place)][j/(2*place)];
			}
		}
		return tmp;
	}

    template<typename T>
    Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &copy) {
		(*this) = (*this) + copy;
		return *this;
    }

    template <typename T>
    int getIndexOfMaxElem(T* first, T* last){
        return std::max_element(first, last)-first;
    }
}