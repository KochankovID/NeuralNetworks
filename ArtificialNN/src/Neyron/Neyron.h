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
        Neyron(const Neyron<T> &&copy); // Копирования

        // Методы класса ---------------------------------------------------------
        static T FunkActiv(const T &e, Func<T> &f);  // Функция активации нейрона
        virtual T Summator(const Matrix<T> &a);  // Операция суммированию произведений входов на веса нейрона

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
    };

    template<typename T>
    Neyron<T>::Neyron() : Weights<T>() {
    }

    template<typename T>
    Neyron<T>::Neyron(const int &i_, const int &j_, const int &wbisas_) : Weights<T>(i_, j_, wbisas_) {
    }

    template<typename T>
    Neyron<T>::Neyron(T **arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_) {
    }

    template<typename T>
    Neyron<T>::Neyron(const Neyron<T> &copy) : Weights<T>(copy) {
    }

    template<typename T>
    Neyron<T>::Neyron(T *arr_, const int &i_, const int &j_, const int &wbisas_) : Weights<T>(arr_, i_, j_, wbisas_) {

    }

    template<typename T>
    Neyron<T>::Neyron(const Neyron<T> &&copy) : Weights<T>(copy) {

    }

    template<typename T>
    inline T Neyron<T>::FunkActiv(const T &e, Func<T> &f) {
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
        return out;
    }

    template<typename T>
    std::istream &operator>>(std::istream &in, Neyron<T> &mat) {
        in >> ((Weights<T> &) mat);
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
        return *this;
    }

    template<typename T>
    Neyron<T>::~Neyron() {
    }
}
#endif //ARTIFICIALNN_NEYRON_H
