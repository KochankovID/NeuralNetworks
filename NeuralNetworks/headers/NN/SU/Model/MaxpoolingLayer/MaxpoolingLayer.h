#ifndef ARTIFICIALNN_MAXPOOLINGLAYER_H
#define ARTIFICIALNN_MAXPOOLINGLAYER_H

#include "Filter.h"
#include "Layer.h"

namespace NN {

    // Класс слой "Макспулинга"
    template<typename T>
    class MaxpoolingLayer : public Layer<T> {
    public:
        // Конструкторы ----------------------------------------------------------
        MaxpoolingLayer(size_t n, size_t m);  // Конструктор инициализатор
        MaxpoolingLayer(const MaxpoolingLayer &copy);  // Конструктор копирования

        Tensor<T> passThrough(const Tensor<T> &in);  // Проход через слой
        Tensor<T> BackPropagation(const Tensor<T> &error, const Tensor<T> &in);  // Обратное распространение ошибки
        void GradDes(ImpulsGrad<T> &G, const Tensor<T> &in) {};  // Градиентный спуск
        void saveToFile(std::ofstream& file);  // Сохранение весов слоя в файл
        void getFromFile(std::ifstream& file);  // Получение весов слоя из файла

        // Деструктор ------------------------------------------------------------
        ~MaxpoolingLayer() = default;

#ifdef TEST_MaxLayer
        // TODO: Not neesesary
    public:
        size_t n_, m_;  // Размеры входного тензора
        Tensor<T> output;  // Выходной тензор (нужен для обучения)
        Matrix<T> passThrough(const Matrix<T> &in);  // Проход через слой для матрицы (служебная функция)
#else
        // TODO: Not neesesary
    private:
        size_t n_, m_;  // Размеры входного тензора
        Tensor<T> output;  // Выходной тензор (нужен для обучения)
        Matrix<T> passThrough(const Matrix<T> &in);  // Проход через слой для матрицы (служебная функция)
#endif
    };

#define D_MaxpoolingLayer MaxpoolingLayer<double>
#define F_MaxpoolingLayer MaxpoolingLayer<float>
#define I_MaxpoolingLayer MaxpoolingLayer<int>

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(size_t n, size_t m) : Layer<T>("MaxpoolingLayer"){
        n_ = n;
        m_ = m;
    }

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(const MaxpoolingLayer &copy) : Layer<T>(copy){
        n_ = copy.n_;
        m_ = copy.m_;
    }

    template<typename T>
    Tensor <T> MaxpoolingLayer<T>::passThrough(const Tensor <T> &in) {
        Tensor<T> result(in.getHeight()/this->n_, in.getWidth()/this->m_, in.getDepth());

        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] = MaxpoolingLayer<T>::passThrough(in[i]);
        }
        output = result;
        return result;
    }

    template<typename T>
    Matrix<T> MaxpoolingLayer<T>::passThrough(const Matrix<T>& in) {
        return Filter<T>::Pooling(in, n_, m_);
    }

    template<typename T>
    Tensor<T>
    MaxpoolingLayer<T>::BackPropagation(const Tensor<T>& error, const Tensor <T>& in) {
        return NN::BackPropagation(in, output, error, n_, m_);
    }

    template<typename T>
    void MaxpoolingLayer<T>::saveToFile(std::ofstream &file) {

    }

    template<typename T>
    void MaxpoolingLayer<T>::getFromFile(std::ifstream &file) {

    }

}
#endif //ARTIFICIALNN_MAXPOOLINGLAYER_H
