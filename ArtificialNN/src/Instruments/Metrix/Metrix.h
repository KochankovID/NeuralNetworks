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
        T operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            int n = out.getN();
            T temp_err;
            T error = 0;
            for (int i = 0; i < n; i++) {
                temp_err = 0;
                for(int j = 0; j < out.getM(); j++) {
                    temp_err += (correct[i][j] - out[i][j]) * (correct[i][j] - out[i][j]);
                }
                temp_err /= n;
                temp_err = std::sqrt(temp_err);
                error += temp_err;
            }
            return error / n;
        }

        ~RMS_error() {};
    };

    template<typename T>
    class RMS_errorD : public Metr<T> {
    public:
        explicit RMS_errorD() {};
        T operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            int n = out.getN();
            T temp_err;
            T error = 0;
            for (int i = 0; i < n; i++) {
                temp_err = 0;
                for(int j = 0; j < out.getM(); j++) {
                    temp_err += (correct[i][j] - out[i][j]) * (correct[i][j] - out[i][j]);
                }
                temp_err /= n;
                error += temp_err;
            }
            return error / n;
        }

        ~RMS_errorD() {};
    };

    template<typename T>
    class Accuracy : public Metr<T> {
    public:
        explicit Accuracy() : Metr<T>() {};
        T operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            T err = 0;
            T answer;
            T right;
            size_t n = out.getN();
            for (int i = 0; i < n; i++) {
                answer = std::max_element(out[i], out[i]+10) - out[i];
                right = std::max_element(correct[i], correct[i]+10) - correct[i];
                err += answer == right ? 1 : 0;
            }
            err /= n;
            return err;
        }

        ~Accuracy() {};
    };
}

#endif //ARTIFICIALNN_METRIX_H
