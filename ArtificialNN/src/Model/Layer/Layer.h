#ifndef ARTIFICIALNN_LAYER_H
#define ARTIFICIALNN_LAYER_H

#include "Gradients.h"
#include "Matrix.h"

namespace ANN {

    template<typename T>
    class Layer {
    public:
        explicit Layer(const std::string& type);
        Layer(const Layer<T>& copy);

        std::string getType() const { return type_;};

        virtual Tensor<T> passThrough(const Tensor<T>& in) = 0;
        virtual Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor <T>& in) = 0;
        virtual void GradDes(ImpulsGrad<T>& G, const Tensor <T>& in) = 0;
        virtual void saveToFile(const std::string& file_name) = 0;
        virtual void getFromFile(const std::string& file_name) = 0;

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

    template<typename T>
    void Layer<T>::saveToFile(const std::string& file_name){
        std::ofstream file;
        file.open(file_name);
        if (!file.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        file << type_ << std::endl;
        file.close();
    }

    template<typename T>
    void Layer<T>::getFromFile(const std::string &file_name){
        std::ifstream file;
        file.open(file_name);
        if (!file.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        file >> type_;
    }
};
#endif //ARTIFICIALNN_LAYER_H
