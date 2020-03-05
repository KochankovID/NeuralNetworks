#include "Weights.h"

#ifndef ARTIFICIALNN_NEYRON_H
#define ARTIFICIALNN_NEYRON_H
namespace ANN {

    template<typename T>
    class Neyron : public Weights{
    public:
        // Конструкторы ----------------------------------------------------------
        Neyron(); // По умолчанию
        Neyron(const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор (нулевая матрица)
        Neyron(T **arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neyron(T *arr_, const int &i_, const int &j_, const int &wbisas_ = 0); // Инициализатор
        Neyron(const Neyron<T> &copy); // Копирования
        Neyron(const Neyron<T> &&copy); // Копирования

        // Методы класса ---------------------------------------------------------
        static Y FunkActiv(const T &e, Func<T> &f);  // Функция активации нейрона
        virtual T Summator(const Matrix<T> &a, const Weights<T> &w);  // Операция суммированию произведений входов на веса нейрона

        // Деструктор ------------------------------------------------------------
        ~Neyron();
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
    inline T Neyron<T>::FunkActiv(const T &e, ::Func<T> &f) {
        return f(e);
    }

    template<typename T, typename Y>
    T Neyron<T, Y>::Summator(const Matrix<T> &a) {
        if ((a.getN() != this->n) || (a.getM() != this->m)) {
            throw Base_Perceptron<T, Y>::NeyronPerceptronExeption(
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

}
#endif //ARTIFICIALNN_NEYRON_H
