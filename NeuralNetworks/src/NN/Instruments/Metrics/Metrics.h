#ifndef ARTIFICIALNN_METRIX_H
#define ARTIFICIALNN_METRIX_H

#include "Metr.h"
#include <algorithm>
#include <math.h>

namespace NN {

    // Класс метрики среднеквадратичной ошибки
    template<typename T>
    class RMS_error : public Metr<T> {
    public:
        // Конструкторы ---------------------------------
        explicit RMS_error() : Metr<T>("RMS_error") {};

        // Перегрузки операторов ------------------------
        Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            Matrix<double> error_v(1, 1);

            for (int i = 0; i < out.getM(); i++) {
                error_v[0][0] += (correct[0][i] - out[0][i]) * (correct[0][i] - out[0][i]);
            }
            error_v[0][0] /=out.getM();
            return error_v;
        }

        // Деструктор -----------------------------------
        ~RMS_error() {};
    };

    // Класс производной среднеквадратичной ошибки
    template<typename T>
    class RMS_errorD : public Metr<T> {
    public:
        // Конструкторы ---------------------------------
        explicit RMS_errorD(): Metr<T>("RMS_errorD") {};

        // Перегрузки операторов ------------------------
        Matrix<double > operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN(), m = out.getM();
            Matrix<double> error_vector(1, m);

            for(size_t j = 0; j < m; j++){
                error_vector[0][j] = -(2.0 / out.getM()) * (correct[0][j] - out[0][j]);
            }

            return error_vector;
        }

        // Деструктор -----------------------------------
        ~RMS_errorD() {};
    };

    // Класс метрики бинарной точности
    template<typename T>
    class BinaryAccuracy : public Metr<T> {
    public:
        // Конструкторы ---------------------------------
        explicit BinaryAccuracy() : Metr<T>("BinaryAccuracy") {};

        // Перегрузки операторов ------------------------
        Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN() , m = out.getM();
            Matrix<double> metrix_vector(1, 1);
            int answer;
            int right;

            metrix_vector[0][0] += out[0][0] == correct[0][0] ? 1 : 0;

            return metrix_vector;
        }

        // Деструктор -----------------------------------
        ~BinaryAccuracy() = default;;
    };

    // Класс метрики точность
    template<typename T>
    class Accuracy : public Metr<T> {
    public:
        // Конструкторы ---------------------------------
        explicit Accuracy() : Metr<T>("Accuracy") {};

        // Перегрузки операторов ------------------------
        Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN() , m = out.getM();
            Matrix<double> metrix_vector(1, 1);
            int answer;
            int right;

            for(size_t i = 0; i < n; i++){
                answer = std::max_element(out[i], out[i]+10) - out[i];
                right = std::max_element(correct[i], correct[i]+10) - correct[i];
                metrix_vector[0][0] += answer == right ? 1 : 0;

            }

            return metrix_vector;
        }

        // Деструктор -----------------------------------
        ~Accuracy() = default;;
    };

#define D_RMS_error RMS_error<double>
#define F_RMS_error RMS_error<float>
#define I_RMS_error RMS_error<int>

#define D_RMS_errorD RMS_errorD<double>
#define F_RMS_errorD RMS_errorD<float>
#define I_RMS_errorD RMS_errorD<int>

#define D_BinaryAccuracy BinaryAccuracy<double>
#define F_BinaryAccuracy BinaryAccuracy<float>
#define I_BinaryAccuracy BinaryAccuracy<int>

#define D_Accuracy Accuracy<double>
#define F_Accuracy Accuracy<float>
#define I_Accuracy Accuracy<int>
}

#endif //ARTIFICIALNN_METRIX_H
