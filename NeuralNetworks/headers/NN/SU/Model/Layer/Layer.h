#ifndef ARTIFICIALNN_LAYER_H
#define ARTIFICIALNN_LAYER_H

#include "Gradients.h"
#include <fstream>
#include "Matrix.h"

namespace NN {

    // Абстрактный класс слой
    template<typename T>
    class Layer {
    public:
        // Конструкторы ----------------------------------------------------------
        explicit Layer(const std::string& type);  // Конструктор инициализатор
        Layer(const Layer<T>& copy);  // Конструктор копирования

        // Методы класса ---------------------------------------------------------
        std::string getType() const { return type_;};  // Получение имени слоя
        virtual Tensor<T> passThrough(const Tensor<T>& in) = 0;  // Проход через слой
        virtual Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor <T>& in) = 0;  // Обратное распространение о
        virtual void GradDes(ImpulsGrad<T>& G, const Tensor <T>& in) = 0;  // Градиентный спуск
        virtual void saveToFile(std::ofstream& file) = 0;  // Сохранение весов слоя в файл
        virtual void getFromFile(std::ifstream& file) = 0;  // Получение весов слоя из файла

        // Деструктор ------------------------------------------------------------
        ~Layer() = default;

    protected:
        // Поля класса ----------------------------------
        std::string type_;  // Имя слоя
    };

    template<typename T>
    Layer<T>::Layer(const Layer<T> &copy) {
        type_ = copy.type_;
    }

    template<typename T>
    Layer<T>::Layer(const std::string &type) {
        type_ = type;
    }
};
#endif //ARTIFICIALNN_LAYER_H
