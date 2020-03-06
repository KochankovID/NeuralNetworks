#ifndef ARTIFICIALNN_METR_H
#define ARTIFICIALNN_METR_H

namespace ANN {

    template<typename T>
    class Metr {
    public:
        Metr() {};

        virtual T operator()(Func<T>& F, std::vector<T> out, std::vector<T> correct) = 0;

        virtual ~Metr() {};
    };
}

#endif //ARTIFICIALNN_METR_H
