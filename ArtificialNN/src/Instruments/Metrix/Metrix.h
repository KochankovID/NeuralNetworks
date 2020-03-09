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
        Matrix<T> operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            int n = out.getN();
            Matrix<T> error_v(1, out.getN());
            for (int i = 0; i < n; i++) {
                for(int j = 0; j < out.getM(); j++) {
                    error_v[0][i] += (correct[i][j] - out[i][j]) * (correct[i][j] - out[i][j]);
                }
                error_v[0][i] /= n;
            }
            return error_v;
        }

        ~RMS_error() {};
    };

    template<typename T>
    class RMS_errorD : public Metr<T> {
    public:
        explicit RMS_errorD(size_t o_):o(o_) {};
        Matrix<T> operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            Matrix<T> error_vector(1, out.getM());
            for(size_t i = 0 ; i < out.getM(); i++){
                error_vector[0][i] = -(2.0 / out.getM()) * (correct[o][i] - out[o][i]);
            }
            return error_vector;
        }

        ~RMS_errorD() {};
    private:
        size_t o;
    };

    template<typename T>
    class Accuracy : public Metr<T> {
    public:
        explicit Accuracy() : Metr<T>() {};
        Matrix<T> operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            Matrix<T> metrix_vector(1, out.getN());
            T answer;
            T right;
            size_t n = out.getN();
            for (int i = 0; i < n; i++) {
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
