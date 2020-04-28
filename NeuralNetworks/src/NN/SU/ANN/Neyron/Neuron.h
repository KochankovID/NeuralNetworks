#ifndef ARTIFICIALNN_NEYRON_H
#define ARTIFICIALNN_NEYRON_H
#include "Weights.h"
#include "Functors.h"

namespace NN {
    template<typename T>
    class Neuron;  // Объявление класса

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Neuron<T> &mat);  // Оператор вывода в поток

    template<typename T>
    std::istream &operator>>(std::istream &in, Neuron<T> &mat);  // Оператор ввода из потока

    template<typename T>
    class Neuron : public Weights<T>{
    public:
        // Конструкторы ----------------------------------------------------------
        Neuron(); // По умолчанию
        Neuron(const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор (нулевая матрица)
        Neuron(T **arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neuron(T *arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neuron(const Neuron<T> &copy); // Копирования
        Neuron(const Neuron<T> &&copy); // Move

        // Методы класса ---------------------------------------------------------
        static T FunkActiv(const T &e, const Func<T> &f);  // Функция активации нейрона
        T Summator(const Matrix<T> &a);  // Операция суммированию произведений входов на веса нейрона
        void setError(const Weights<T>& err);  // Установка ошибки весов
        Weights<T> getError(){ return error; };  // Получение ошибки весов

        // Перегрузки операторов ------------------------
        Neuron<T> &operator=(const Neuron<T> &copy); // Оператор присваивания
        friend std::ostream &operator<<<>(std::ostream &out, const Neuron<T> &mat); // Оператор вывод матрицы в поток
        friend std::istream &operator>><>(std::istream &in, Neuron<T> &mat); // Оператор чтение матрицы из потока

        // Деструктор ------------------------------------------------------------
        ~Neuron();

        // Класс исключения ------------------------------------------------------
        class NeyronExeption : public std::runtime_error {
        public:
            NeyronExeption(std::string str) : std::runtime_error(str) {};

            ~NeyronExeption() {};
        };
    private:
        // Поля класса ----------------------------------
        Weights<T> error;  // Ошибка весов
    };

#define D_Neuron Neuron<double>
#define F_Neuron Neyron<float>
#define I_Neuron Neuron<int>

    template<typename T>
    Neuron<T>::Neuron() : Weights<T>(), error() {
    }

    template<typename T>
    Neuron<T>::Neuron(const int &i_, const int &j_, const int &wbisas_) : Weights<T>(i_, j_, wbisas_), error(i_, j_) {
    }

    template<typename T>
    Neuron<T>::Neuron(T **arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_),
                                                                                    error(i_, j_){
    }

    template<typename T>
    Neuron<T>::Neuron(const Neuron<T> &copy) : Weights<T>(copy), error(copy.error) {
    }

    template<typename T>
    Neuron<T>::Neuron(T *arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_),
                                                                                   error(i_, j_){

    }

    template<typename T>
    Neuron<T>::Neuron(const Neuron<T> &&copy) : Weights<T>(copy), error(copy.error) {

    }

    template<typename T>
    inline T Neuron<T>::FunkActiv(const T &e, const Func<T> &f) {
        return f(e);
    }

    template<typename T>
    T Neuron<T>::Summator(const Matrix<T> &a) {
        if ((a.getN() != this->n) || (a.getM() != this->m)) {
            throw Neuron<T>::NeyronExeption(
                    "Несовпадение размера матрицы весов и размера матрицы входных сигналов!");
        }
        T sum = 0;

#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < a.getN(); i++) {
            for (int j = 0; j < a.getM(); j++) {
                sum += a[i][j] * (*this)[i][j];
            }
        }
        sum += this->wbias;
        return sum;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Neuron<T> &mat) {
        out << (Weights<T>) mat;
        out << mat.error;
        return out;
    }

    template<typename T>
    std::istream &operator>>(std::istream &in, Neuron<T> &mat) {
        in >> ((Weights<T> &) mat);
        in >> mat.error;
        return in;
    }

    template<typename T>
    Neuron<T>& Neuron<T>::operator=(const Neuron<T> &copy){
        if (this == &copy) {
            return *this;
        }

        if ((copy.n > this->n) || (copy.m > this->m)) {
            if (this->n == 0 && this->m == 0) {
                this->n = copy.n;
                this->m = copy.m;
                this->initMat();
                this->d = copy.d;
                this->wbias = copy.wbias;
            } else {
                this->deinitMat();
                this->n = copy.n;
                this->m = copy.m;
                this->initMat();
                this->d = copy.d;
                this->wbias = copy.wbias;
            }
        } else {
            this->n = copy.n;
            this->m = copy.m;
            this->d = copy.d;
            this->wbias = copy.wbias;
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
                this->arr[i][j] = copy.arr[i][j];
            }
        }
        error = copy.error;
        return *this;
    }

    template<typename T>
    Neuron<T>::~Neuron() {
    }

    template<typename T>
    void Neuron<T>::setError(const Weights <T> &err) {
        if((err.getM() != error.getM() )||(err.getN() != error.getN())){
            throw Neuron<T>::NeyronExeption(
                    "Несовпадение размера матрицы весов и размера матрицы ошибок!");
        }
        error = err;
    }
}
#endif //ARTIFICIALNN_NEYRON_H
