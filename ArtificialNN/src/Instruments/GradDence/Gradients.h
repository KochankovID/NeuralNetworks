#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include <algorithm>
#include "opencv2/opencv.hpp"

#include <math.h>

namespace ANN {

    template<typename T>
    class SimpleGrad : public Grad_speed<T> {
    public:
        explicit SimpleGrad(const double &a_) : Grad_speed<T>(a_) {};
        void operator()(Neyron <T> &w, const Matrix <T> &in, const Func <T> &F) {
            T x = w.Summator(in);

            cv::parallel_for_(cv::Range(0, w.getN()), [&](const cv::Range &range) {
                for (int i = range.start; i < range.end; i++) {
                    for (int j = 0; j < w.getM(); j++) {
                        w[i][j] -= (w.GetD() * this->a * F(x) * in[i][j]);
                    }
                }
            });
            w.GetWBias() -= this->a * F(x) * w.GetD();
        }

        ~SimpleGrad() {};
    };
}

#endif //ARTIFICIALNN_GRADIENTS_H
