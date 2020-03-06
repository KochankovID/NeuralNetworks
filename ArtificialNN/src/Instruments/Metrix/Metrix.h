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
        void operator()(std::vector<T> out, std::vector<T> correct) {
            T err = 0;
            for (int i = 0; i < lenth; i++) {
                err += (a[i] - y[i]) * (a[i] - y[i]);
            }
            err /= 2;
            return err;
        }

        ~RMS_error() {};
    };


}

#endif //ARTIFICIALNN_METRIX_H
