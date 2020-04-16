#ifndef ARTIFICIALNN_INIT_H
#define ARTIFICIALNN_INIT_H

namespace ANN {

    template<typename T>
    class Init {
    public:
        Init() {};

        virtual T operator()() const = 0;

        virtual ~Init() {};
    };

}
#endif //ARTIFICIALNN_INIT_H
