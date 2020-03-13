#ifndef ARTIFICIALNN_METRIX_H
#define ARTIFICIALNN_METRIX_H

#include "Metr.h"
#include <algorithm>
#include <math.h>

namespace ANN {

    template<typename T>
    class RMS_error : public Metr<T> {
    public:
        explicit RMS_error() {};
        Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN() , m = out.getM();
            Matrix<double> error_v(1, n);

            for (int i = 0; i < n; i++) {
                for(int j = 0; j < m; j++) {
                    error_v[0][i] += (correct[i][j] - out[i][j]) * (correct[i][j] - out[i][j]);
                }
                error_v[0][i] /= m;
            }
            return error_v;
        }

        ~RMS_error() {};
    };

    template<typename T>
    class RMS_errorD : public Metr<T> {
    public:
        explicit RMS_errorD(): Metr<T>() {};

        Matrix<double > operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN(), m = out.getM();
            Matrix<double> error_vector(1, m);

            for(size_t j = 0; j < m; j++){
                error_vector[0][j] = -(2.0 / out.getM()) * (correct[0][j] - out[0][j]);
            }

            return error_vector;
        }

        ~RMS_errorD() {};
    };

    template<typename T>
    class Accuracy : public Metr<T> {
    public:
        explicit Accuracy() : Metr<T>() {};
        Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const {
            size_t n = out.getN() , m = out.getM();
            Matrix<double> metrix_vector(1, n);
            T answer;
            T right;

            for(size_t i = 0; i < n; i++){
                answer = std::max_element(out[i], out[i]+10) - out[i];
                right = std::max_element(correct[i], correct[i]+10) - correct[i];
                metrix_vector[0][i] += answer == right ? 1 : 0;
            }

            return metrix_vector;
        }

        ~Accuracy() {};
    };
}

#endif //ARTIFICIALNN_METRIX_H
