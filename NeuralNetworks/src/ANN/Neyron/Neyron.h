#ifndef ARTIFICIALNN_NEYRON_H
#define ARTIFICIALNN_NEYRON_H
#include "Weights.h"
#include "Functors.h"

namespace ANN {
    template<typename T>
    class Neyron;

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const Neyron<T> &mat);

    template<typename T>
    std::istream &operator>>(std::istream &in, Neyron<T> &mat);

    template<typename T>
    class Neyron : public Weights<T>{
    public:
        // Конструкторы ----------------------------------------------------------
        Neyron(); // По умолчанию
        Neyron(const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор (нулевая матрица)
        Neyron(T **arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neyron(T *arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neyron(const Neyron<T> &copy); // Копирования
        Neyron(const Neyron<T> &&copy); // Move

        // Методы класса ---------------------------------------------------------
        static T FunkActiv(const T &e, const Func<T> &f);  // Функция активации нейрона
        T Summator(const Matrix<T> &a);  // Операция суммированию произведений входов на веса нейрона
        void setError(const Weights<T>& err);
        Weights<T> getError(){ return error; };

        // Перегрузки операторов ------------------------
        Neyron<T> &operator=(const Neyron<T> &copy); // Оператор присваивания
        friend std::ostream &operator<<<>(std::ostream &out, const Neyron<T> &mat); // Оператор вывод матрицы в поток
        friend std::istream &operator>><>(std::istream &in, Neyron<T> &mat); // Оператор чтение матрицы из потока

        // Деструктор ------------------------------------------------------------
        ~Neyron();

        // Класс исключения ------------------------------------------------------
        class NeyronExeption : public std::runtime_error {
        public:
            NeyronExeption(std::string str) : std::runtime_error(str) {};

            ~NeyronExeption() {};
        };
    private:
        Weights<T> error;
    };

    template<typename T>
    Neyron<T>::Neyron() : Weights<T>(), error() {
    }

    template<typename T>
    Neyron<T>::Neyron(const int &i_, const int &j_, const int &wbisas_) : Weights<T>(i_, j_, wbisas_), error(i_, j_) {
    }

    template<typename T>
    Neyron<T>::Neyron(T **arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_),
            error(i_, j_){
    }

    template<typename T>
    Neyron<T>::Neyron(const Neyron<T> &copy) : Weights<T>(copy), error(copy.error) {
    }

    template<typename T>
    Neyron<T>::Neyron(T *arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_),
            error(i_, j_){

    }

    template<typename T>
    Neyron<T>::Neyron(const Neyron<T> &&copy) : Weights<T>(copy), error(copy.error) {

    }

    template<typename T>
    inline T Neyron<T>::FunkActiv(const T &e, const Func<T> &f) {
        return f(e);
    }

    template<typename T>
    T Neyron<T>::Summator(const Matrix<T> &a) {
        if ((a.getN() != this->n) || (a.getM() != this->m)) {
            throw Neyron<T>::NeyronExeption(
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
    std::ostream &operator<<(std::ostream &out, const Neyron<T> &mat) {
        out << (Weights<T>) mat;
        out << mat.error;
        return out;
    }

    template<typename T>
    std::istream &operator>>(std::istream &in, Neyron<T> &mat) {
        in >> ((Weights<T> &) mat);
        in >> mat.error;
        return in;
    }

    template<typename T>
    Neyron<T>& Neyron<T>::operator=(const Neyron<T> &copy){
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
    Neyron<T>::~Neyron() {
    }

    template<typename T>
    void Neyron<T>::setError(const Weights <T> &err) {
        if((err.getM() != error.getM() )||(err.getN() != error.getN())){
            throw Neyron<T>::NeyronExeption(
                    "Несовпадение размера матрицы весов и размера матрицы ошибок!");
        }
        error = err;
    }
}
#endif //ARTIFICIALNN_NEYRON_H
