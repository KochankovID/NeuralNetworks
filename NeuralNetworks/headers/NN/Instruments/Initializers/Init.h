#ifndef ARTIFICIALNN_INIT_H
#define ARTIFICIALNN_INIT_H

namespace NN {

    // Абстрактный класс инициализатора
    template<typename T>
    class Init {
    public:
        // Конструкторы ---------------------------------
        Init() {};

        // Перегрузки операторов ------------------------
        virtual T operator()() const = 0;

        // Деструктор -----------------------------------
        virtual ~Init() {};
    };

}
#endif //ARTIFICIALNN_INIT_H
