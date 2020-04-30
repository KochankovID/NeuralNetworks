#ifndef NEURALNETWORKS_SOM_H
#define NEURALNETWORKS_SOM_H

#include "Ndarray.h"
#include <string>

namespace NN{

    // Класс "Self organizing map"
    class SOM{
    public:
        // Конструкторы ---------------------------------
        SOM(int x, int y, int input_length, double learning_rate = 0.5, double radius = 1.0);  // Инициализатор
        SOM(const SOM& copy);  // Копирования

        // Методы класса --------------------------------
        void random_weights_init(const Ndarray<double>& data);

        // Перегрузки операторов ------------------------
        // Деструктор -----------------------------------
        ~SOM();

        // Класс исключений -----------------------------
        class SOMExeption : public std::logic_error {
        public:
            SOMExeption(std::string str) : std::logic_error(str) {};

            ~SOMExeption() {};
        };
    protected:
        // Поля класса ----------------------------------
        Ndarray<double > weights_;  // Веса сети
        double learning_rate_;
        double radius_;

        // Скрытые матоды класса ------------------------
    };

}

#endif //NEURALNETWORKS_SOM_H
