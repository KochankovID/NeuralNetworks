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
        void operator()(const std::vector<T>& out, const std::vector<T>& correct) {
            T err = 0;
            for (int i = 0; i < lenth; i++) {
                err += (correct[i] - out[i]) * (correct[i] - out[i]);
            }
            err /= n;
            return std::sqrt(err);
        }

        ~RMS_error() {};
    };

}

#endif //ARTIFICIALNN_METRIX_H
