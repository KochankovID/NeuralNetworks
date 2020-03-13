#ifndef ARTIFICIALNN_INITIALIZERS_H
#define ARTIFICIALNN_INITIALIZERS_H

#include "Init.h"
#include <random>

namespace ANN {

    template<typename T>
    class SimpleInitializator : public Init<T> {
    public:
        explicit SimpleInitializator() : Init<T>() {};

        T operator()(){
            srand(time(0));
            return double(rand()) / RAND_MAX - 0.5;
        }

        ~SimpleInitializator() {};
    };

}

#endif //ARTIFICIALNN_INITIALIZERS_H
