#ifndef NEURALNETWORKS_HOPFIELD_H
#define NEURALNETWORKS_HOPFIELD_H

#include "Ndarray.h"

namespace NN {

    class Hopfield{
    public:
        // Конструкторы ---------------------------------
        explicit Hopfield(int number_neyrons);  // инициализатор
        Hopfield(const Hopfield& copy);  // копирования

        // Методы класса --------------------------------
        void train(const Ndarray<int>& data);  // "Обучение сети" распознаванию нового образца
        Ndarray<int> fit(const Ndarray<int>& sample);  // Работа сети

        // Перегрузки операторов ------------------------
        // Деструктор -----------------------------------
        virtual ~Hopfield()= default;
        // Класс исключений -----------------------------
        class HopfieldExeption : public std::logic_error {
        public:
            HopfieldExeption(std::string str) : std::logic_error(str) {};

            ~HopfieldExeption() {};
        };
#ifdef Hopfield_TEST
    public:
        // Поля класса ----------------------------------
        Ndarray<int> weights_;  // Матрица весов

        // Скрытые матоды класса ------------------------
#else
    protected:
        // Поля класса ----------------------------------
        Ndarray<int> weights_;  // Матрица весов

        // Скрытые матоды класса ------------------------
#endif
    };

}
#endif //NEURALNETWORKS_HOPFIELD_H
