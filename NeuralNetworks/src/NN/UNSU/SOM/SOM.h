#ifndef NEURALNETWORKS_SOM_H
#define NEURALNETWORKS_SOM_H

#include "Ndarray.h"
#include <string>
#include "Initializers.h"

namespace NN{

    // Класс "Self organizing map"
    class SOM{
    public:
        // Конструкторы ---------------------------------
        SOM(int x, int y, int input_length, double learning_rate = 0.5, double radius = 1.0);  // Инициализатор
        SOM(const SOM& copy);  // Копирования

        // Методы класса --------------------------------
        void random_weights_init(const Init<double >& init);  // Инициализациня весов сети
        void train_random(const Ndarray<double >& data, int num_iteration);  // Обучение сети
        vector<int> winner(const Ndarray<double >& data);  // Поределение нерона - "победителя"
        double euclidean_distance(const Ndarray<double>& vect_1, const Ndarray<double>& vect_2);  // Расчет Евклидова расстояния между двумя векторами

        // Перегрузки операторов ------------------------
        // Деструктор -----------------------------------
        virtual ~SOM() = default;

        // Класс исключений -----------------------------
        class SOMExeption : public std::logic_error {
        public:
            SOMExeption(std::string str) : std::logic_error(str) {};

            ~SOMExeption() {};
        };

#ifdef TEST_SOM
    public:
        // Поля класса ----------------------------------
        Ndarray<double > weights_;  // Веса сети
        double learning_rate_;
        double radius_;

        // Скрытые матоды класса ------------------------
#else
    protected:
        // Поля класса ----------------------------------
        Ndarray<double > weights_;  // Веса сети
        double learning_rate_;
        double radius_;

        // Скрытые матоды класса ------------------------
#endif
    };

}

#endif //NEURALNETWORKS_SOM_H
