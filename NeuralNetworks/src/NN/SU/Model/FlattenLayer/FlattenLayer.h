#ifndef ARTIFICIALNN_FLATTENLAYER_H
#define ARTIFICIALNN_FLATTENLAYER_H

#include "Layer.h"
#include "DenceLayer.h"
#include "Layer.h"

namespace NN {

    // Класс "плоский" слой
    template<typename T>
    class FlattenLayer : public Layer<T>{
    public:
        // Конструкторы ---------------------------------
        FlattenLayer(size_t height_, size_t width_, size_t depth_) : height(height_),
            width(width_), depth(depth_), Layer<T>("FlattenLayer"){};  // Конструктор инициализатор
        FlattenLayer(const FlattenLayer<T>& copy);  // Конструктор копирования

        // Методы класса ---------------------------------------------------------
        Tensor<T> passThrough(const Tensor<T>& in);  // Проход через слой
        Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor <T>& in);  // Обратное распространение ошибки
        void GradDes(ImpulsGrad<T>& G, const Tensor <T>& in){};  // Градиентный спуск
        void saveToFile(std::ofstream& file);  // Сохранение весов слоя в файл
        void getFromFile(std::ifstream& file);  // Получение весов слоя из файла

        // Деструктор ------------------------------------------------------------
        ~FlattenLayer();

#ifdef TEST_FlatternLayer
        // TODO: Not neesesary
        // Поля класса ----------------------------------
        size_t height,  // Высота входного тензора
        width,  // Ширина входного тензора
        depth;  // Глубина входного тензора
#else
        // TODO: Not neesesary
    private:
        // Поля класса ----------------------------------
        size_t height,  // Высота входного тензора
        width,  // Ширина входного тензора
        depth;  // Глубина входного тензора
#endif
    };

#define D_FlattenLayer FlattenLayer<double>
#define F_FlattenLayer FlattenLayer<float>
#define I_FlattenLayer FlattenLayer<int>

    template<typename T>
    Tensor<T> FlattenLayer<T>::passThrough(const Tensor<T> &in) {

        Tensor<T> m_(1, in.getDepth()*in.getHeight()*in.getWidth(), 1);

        for(size_t z = 0; z < in.getDepth(); z++){
            for(size_t x = 0; x < in.getHeight(); x++){
                for(size_t y = 0; y < in.getWidth(); y++){
                    m_[0][0][z*in.getHeight()*in.getWidth() + x*in.getWidth() + y] = in[z][x][y];
                }
            }
        }
        return m_;
    }


    template<typename T>
    Tensor<T> FlattenLayer<T>::BackPropagation(const Tensor<T>& error, const Tensor <T>& in) {
        Tensor<T> m_(height, width, depth);

        if(in.getHeight()*in.getWidth()*in.getDepth() != height*width*depth) {
            throw std::runtime_error("Sizes of tensors are not equal!");
        }

        for(size_t z = 0; z < m_.getDepth(); z++){
            for(size_t x = 0; x < m_.getHeight(); x++){
                for(size_t y = 0; y < m_.getWidth(); y++){
                    m_[z][x][y] = error[0][0][z*m_.getHeight()*m_.getWidth() + x*m_.getWidth() + y];
                }
            }
        }

        return m_;
    }

    template<typename T>
    FlattenLayer<T>::FlattenLayer(const FlattenLayer<T> &copy) : Layer<T>("FlattenLayer") {
        height = copy.height;
        width = copy.width;
        depth = copy.depth;
    }

    template<typename T>
    void FlattenLayer<T>::saveToFile(std::ofstream &file) {

    }

    template<typename T>
    void FlattenLayer<T>::getFromFile(std::ifstream &file) {

    }

    template<typename T>
    FlattenLayer<T>::~FlattenLayer() = default;
}

#endif //ARTIFICIALNN_FLATTENLAYER_H
