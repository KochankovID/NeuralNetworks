#ifndef NEURALNETWORKS_VECTOR_H
#define NEURALNETWORKS_VECTOR_H

#include "Matrix.h"

namespace NN {

    template<typename T>
    class Vector : public Matrix<T> {
    public:
        // Конструкторы ---------------------------------
        Vector(); // Конструктор по умолчанию -----------
        Vector(T *arr_, const int &j); // Конструктор инициализатор
        Vector(const int &j); // Конструктор инициализатор (создает матрицу заданного размера заполненную 0)
        Vector(const Ndarray<T>& ndarray);
        Vector(const Vector<T> &copy); // Конструктор копирования
        Vector(Vector<T> &&copy); // Конструктор move


        // Методы класса
        T* Data(){ return Matrix<T>::operator[](0); }

        // Перегрузки операторов ------------------------
        Vector<T> &operator=(const Vector<T> &copy); // Оператор присваивания
        T &operator[](int index); // Оператор индексации
        const T &operator[](int index) const; // Оператор индексации константы

        // Деструктор -----------------------------------
        ~Vector();
    };

#define D_Vector Vector<double>
#define F_Vector Vector<float>
#define I_Vector Vector<int>

    template<typename T>
    Vector<T>::Vector() : Matrix<T>() {

    }

    template<typename T>
    Vector<T>::Vector(T *arr_, const int &j) : Matrix<T>(arr_, 1, j) {

    }

    template<typename T>
    Vector<T>::Vector(const int &j) : Matrix<T>(1, j) {

    }

    template<typename T>
    Vector<T>::Vector(const Vector<T> &copy) : Matrix<T>(copy) {

    }

    template<typename T>
    Vector<T>::Vector(Vector<T> &&copy) : Matrix<T>(copy) {

    }

    template<typename T>
    Vector<T> &Vector<T>::operator=(const Vector<T> &copy) {
        *dynamic_cast<Matrix<T> *>(this) = copy;
        return this;
    }

    template<typename T>
    T &Vector<T>::operator[](int index) {
        return Matrix<T>::operator[](0)[index];
    }

    template<typename T>
    const T &Vector<T>::operator[](int index) const {
        return Matrix<T>::operator[](0)[index];
    }

    template<typename T>
    Vector<T>::~Vector() {

    }

    template<typename T>
    Vector<T>::Vector(const Ndarray <T> &ndarray){
        if(ndarray.shape().size() > 1){
            throw std::logic_error("Wrong shape!");
        }
        Matrix<T>(*this) = Matrix<T>(ndarray);
    }

#define D_Vector Vector<double>
#define F_Vector Vector<float>
#define I_Vector Vector<int>

}
#endif //NEURALNETWORKS_VECTOR_H
