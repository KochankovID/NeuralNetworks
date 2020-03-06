#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include <algorithm>
#include <math.h>

namespace ANN {

    template<typename T>
    class SimpleGrad : public Grad_speed<T> {
    public:
        explicit SimpleGrad(const double &a_) : Grad_speed<T>(a_) {};
        void operator()(Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x) {

            cv::parallel_for_(cv::Range(0, w.getN()), [&](const cv::Range &range) {
                for (int i = range.start; i < range.end; i++) {
                    for (int j = 0; j < w.getM(); j++) {
                        w[i][j] -= (w.GetD() * E * F(x) * in[i][j]);
                    }
                }
            });
            w.GetWBias() -= E * F(x) * w.GetD();
        }

        ~SimpleGrad() {};
    };
}

#endif //ARTIFICIALNN_GRADIENTS_H
