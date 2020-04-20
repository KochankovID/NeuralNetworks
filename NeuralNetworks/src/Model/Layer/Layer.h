#ifndef ARTIFICIALNN_LAYER_H
#define ARTIFICIALNN_LAYER_H

#include "Gradients.h"
#include <fstream>
#include "Matrix.h"

namespace NN {

    template<typename T>
    class Layer {
    public:
        explicit Layer(const std::string& type);
        Layer(const Layer<T>& copy);

        std::string getType() const { return type_;};

        virtual Tensor<T> passThrough(const Tensor<T>& in) = 0;
        virtual Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor <T>& in) = 0;
        virtual void GradDes(const ImpulsGrad<T>& G, const Tensor <T>& in) = 0;
        virtual void saveToFile(std::ofstream& file) = 0;
        virtual void getFromFile(std::ifstream& file) = 0;

        ~Layer() = default;

    protected:
        std::string type_;
    };

    template<typename T>
    Layer<T>::Layer(const Layer<T> &copy) {
        type_ = copy.type_;
    }

    template<typename T>
    Layer<T>::Layer(const std::string &type) {
        type_ = type;
    }

//    template<typename T>
//    void Layer<T>::saveToFile(std::ofstream& file){
//        file << type_ << std::endl;
//    }
//
//    template<typename T>
//    void Layer<T>::getFromFile(std::ifstream& file){
//        file >> type_;
//    }
};
#endif //ARTIFICIALNN_LAYER_H
