#ifndef ARTIFICIALNN_DENCELAYER_H
#define ARTIFICIALNN_DENCELAYER_H

#include "Neyrons.h"
#include <string>

namespace ANN {

    template<typename T>
    class DenceLayer {
    public:
        DenceLayer(size_t number_neyrons, Func<T> F, Func<T> FD, );
        DenceLayer(const DenceLayer& copy);

        passThrough(const Matrix<T>& in);

        getWeightsFromFile(const std::string file_name);
        saveWeightsToFile(const std::string file_name);

        BackPropagation(const DenceLayer<T>& y);
        GradDes(Grad<T>& G, const Matrix <T>& in);


        ~DenceLayer();
    private:
        Matrix<Neyron<T>> m_;
        Func<T> F_;
        Func<T> FD_;
    };

}

#endif //ARTIFICIALNN_DENCELAYER_H
