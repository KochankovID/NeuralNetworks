#ifndef ARTIFICIALNN_TENSOR_H
#define ARTIFICIALNN_TENSOR_H

#include <string>
#include <memory>
#include <iostream>
#include <math.h>
#include "Matrix.h"

namespace ANN {

    template<typename T>
    class Tensor;

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Tensor<T> &mat);

    template<typename T>
    std::istream &operator>>(std::istream &out, Tensor<T> &mat);

    template<typename T>
    class Tensor : public Matrix<Matrix<T>>{
    public:
        // Конструкторы ---------------------------------
        Tensor(); // Конструктор по умолчанию -----------
        Tensor(int height, int width, int depth); // Конструктор инициализатор (создает матрицу заданного размера заполненную 0)
        Tensor(const Matrix<T>& elem); // Конструктор инициализатор (создает матрицу заданного размера заполненную 0)
        Tensor(const Tensor<T> &copy); // Конструктор копирования
        Tensor(Tensor<T> &&copy); // Конструктор move

        // Методы класса --------------------------------
        // Получение количества строк
        int getHeight() const {
            return this->arr[0][0].getN();
        }

        // Получение колисчества столбцов
        int getWidth() const {
            return this->arr[0][0].getM();
        }

        // Получение глубины тензора
        int getDepth() const {
            return this->getM()-1;
        }

        // Масштабирование тензора
        Tensor<T> zoom(int place) const;

        // Заполнение тензора заданным значением
        void Fill(const T &a);

        // Перегрузки операторов ------------------------
        Tensor<T> &operator=(const Tensor<T> &copy); // Оператор присваивания
        Tensor<T> &operator+=(const Tensor<T> &mat); // Оператор присваивания
        Tensor<T> operator+(const Tensor<T> &mat) const; // Оператор суммы
        friend std::ostream &operator<<<>(std::ostream &out, const Tensor<T> &mat); // Оператор вывод тензора в поток
        friend std::istream &operator>><>(std::istream &out, Tensor<T> &mat); // Оператор чтение тензора из потока
        Matrix<T>& operator[](int index); // Оператор индексации
        const Matrix<T>& operator[](int index) const; // Оператор индексации константы
        bool operator==(const Tensor<T> &mat) const; // Оператор сравнения тензоров


        // Деструктор -----------------------------------
        virtual ~Tensor();

        // Класс исключений ----------------------------
        class TensorExeption : public std::runtime_error {
        public:
            TensorExeption(std::string s) : std::runtime_error(s) {}

            ~TensorExeption() {}
        };

    protected:
        void isInRange(int index) const; // Проверяет, находится ли индекс в допустимых границах
    };

#define D_Tensor Tensor<double>
#define F_Tensor Tensor<float>
#define I_Tensor Tensor<int>

    template<typename T>
    Tensor<T>::Tensor() : Matrix<Matrix<T> >(1, 1){
    }

    template<typename T>
    Tensor<T>::Tensor(int height, int width, int depth) : Matrix<Matrix<T> >(1, depth+1){
        if((height < 0)||(width < 0)||(depth < 0)){
            throw TensorExeption("Wrong shape!");
        }

        for(size_t i = 0; i < this->m; i++){
            this->arr[0][i] = Matrix<T>(height, width);
        }
    }

    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &copy) : Matrix<Matrix<T> >(copy) {
    }

    template<typename T>
    Tensor<T>::Tensor(Tensor<T> &&copy) : Matrix<Matrix<T> >(copy) {
    }

    template<typename T>
    Tensor<T> Tensor<T>::zoom(int place) const {
        size_t height_ = this->arr[0][0].getN();
        size_t width_ = this->arr[0][0].getM();
        size_t depth_ = this->getM() - 1;
        if(place <= 0){
            throw Tensor<T>::TensorExeption("Неверный размер свободного пространства!");
        }
        Tensor<T> result(height_ + (height_-1) * place, width_ + (width_-1) * place, depth_);

        for(size_t i = 0; i < depth_; i++){
            result[i] = (*this)[i].zoom(place);
        }
        return result;
    }

    template<typename T>
    Matrix<T>& Tensor<T>::operator[](int index) {
        this->isInRange(index);
        return this->arr[0][index];
    }

    template<typename T>
    const Matrix<T>& Tensor<T>::operator[](int index) const {
        this->isInRange(index);
        return this->arr[0][index];
    }

    template<typename T>
    void Tensor<T>::Fill(const T &a) {
        size_t depth_ = this->m-1;
        for(size_t i = 0; i < depth_; i++){
            (*this)[i].Fill(a);
        }
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator=(const Tensor<T> &copy) {
        if (this == &copy) {
            return *this;
        }
        this->deinitMat();
        this->m = copy.getM();
        this->n = copy.getN();
        this->initMat();
        for(size_t i = 0; i < this->m; i++){
            this->arr[0][i] = copy.arr[0][i];
        }
        return *this;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Tensor<T> &ten) {
        out << ten.getDepth() << std::endl;
        for(size_t i =0; i < ten.getDepth(); i++){
            out << ten[i];
        }
        return out;
    }

    template<typename T>
    std::istream &operator>>(std::istream &in, Tensor<T> &ten) {
        size_t depth;
        in >> depth;
        ten = Tensor<T>(0, 0, depth);
        for(size_t i = 0; i < ten.getDepth(); i++){
            in >> ten[i];
        }
        return in;
    }

    template<typename T>
    Tensor <T> &Tensor<T>::operator+=(const Tensor<T> &mat) {
        if(this->getDepth() != mat.getDepth()){
            throw TensorExeption("Mismatch shapes of tensors!");
        }
        for(size_t i = 0; i < mat.getDepth(); i++){
            (*this)[i] += mat[i];
        }
        return *this;
    }

template<typename T>
Tensor <T> Tensor<T>::operator+(const Tensor<T> &mat) const{
    if(this->getDepth() != mat.getDepth()){
        throw TensorExeption("Mismatch shapes of tensors!");
    }
    Tensor<T> res(*this);
    res += mat;
    return res;
}

template<typename T>
    Tensor<T>::Tensor(const Matrix <T> &elem) : Matrix<Matrix<T> >(1, 2) {
        this->arr[0][0] = elem;
    }

    template<typename T>
    bool Tensor<T>::operator==(const Tensor<T> &ten) const {
        size_t height_ = this->arr[0][0].getN();
        size_t width_ = this->arr[0][0].getM();
        size_t depth_ = this->m -1;

        if ((height_ != ten.getHeight()) || (width_ != ten.getWidth())||(depth_ != ten.getDepth())) {
            return false;
        }

        for(size_t i = 0; i < depth_; i++){
            if(!((*this)[i] == ten[i])){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    Tensor<T>::~Tensor() {

    }

    template<typename T>
    void Tensor<T>::isInRange(int index) const {
        if ((index >= this->m-1) || (index < 0)) {
            throw TensorExeption("Индекс выходит за размер матрицы!");
        }
    }

}

#endif //ARTIFICIALNN_TENSOR_H
