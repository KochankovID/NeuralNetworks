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
            T err = 0;
            size_t n = out.getM();
            for (int i = 0; i < n; i++) {
                err += (correct[0][i] - out[0][i]) * (correct[0][i] - out[0][i]);
            }
            err /= n;
            return std::sqrt(err);
        }

        ~RMS_error() {};
    };

    template<typename T>
    class Accuracy : public Metr<T> {
    public:
        explicit Accuracy() {};
        T operator()(const Matrix<T>& out, const Matrix<T>& correct) {
            T err = 0;
            size_t n = out.getM();
            for (int i = 0; i < n; i++) {
                err += out[0][i] == correct[0][i] ? 1 : 0;
            }
            err /= n;
            return std::sqrt(err);
        }

        ~Accuracy() {};
    };
}

#endif //ARTIFICIALNN_METRIX_H
