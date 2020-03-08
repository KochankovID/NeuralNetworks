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
        T operator()(const std::vector<T>& out, const std::vector<T>& correct) {
            T err = 0;
            size_t n = out.size();
            for (int i = 0; i < n; i++) {
                err += (correct[i] - out[i]) * (correct[i] - out[i]);
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
        T operator()(const std::vector<T>& out, const std::vector<T>& correct) {
            T err = 0;
            size_t n = out.size();
            for (int i = 0; i < n; i++) {
                err += out[i] == correct[i] ? 1 : 0;
            }
            err /= n;
            return std::sqrt(err);
        }

        ~Accuracy() {};
    };
}

#endif //ARTIFICIALNN_METRIX_H
