#ifndef ARTIFICIALNN_LAYER_H
#define ARTIFICIALNN_LAYER_H

#include "Gradients.h"
#include "Matrix.h"

namespace ANN {

    template<typename T>
    class Layer {
    public:
        explicit Layer(const std::string& type, size_t in_height, size_t in_width, size_t in_depth,
                size_t out_height, size_t out_width, size_t out_depth);
        Layer(const Layer<T>& copy);

        std::string getType() const { return type_;};

        size_t getInHeight() const { return in_.getHeight(); };
        size_t getInWidth() const { return in_.getWidth(); };
        size_t getInDepth() const { return in_.getDepth(); };

        size_t getOutHeight() const { return out_.getHeight(); };
        size_t getOutWidth() const { return out_.getWidth(); };
        size_t getOutDepth() const { return out_.getDepth(); };

        virtual void passThrough(Tensor<T> in) = 0;

        virtual void BackPropagation() = 0;

        virtual void GradDes(Grad<T>& G, const Tensor <T>& in) = 0;
        virtual void GradDes(ImpulsGrad<T>& G, const Tensor <T>& in) = 0;

        virtual void saveToFile(const std::string& file_name) = 0;
        virtual void getFromFile(const std::string& file_name) = 0;

        ~Layer() = default;

    protected:
        std::string type_;

        Tensor<T> in_;
        Tensor<T> out_;
    };

    template<typename T>
    Layer<T>::Layer(const Layer<T> &copy) {
        type_ = copy.type_;
    }

    template<typename T>
    Layer<T>::Layer(const std::string &type, size_t in_height, size_t in_width, size_t in_depth, size_t out_height,
                    size_t out_width, size_t out_depth) {
        type_ = type;
        in_ = Tensor<T>(in_height, in_width, in_depth);
        out_ = Tensor<T>(out_height, out_width, out_depth);
    }

    template<typename T>
    void Layer<T>::saveToFile(const std::string& file_name){
        std::ofstream file;
        file.open(file_name);
        if (!file.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        file << type_ << std::endl;
        file << in_;
        file << out_;
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
        file >> in_;
        file >> out_;
    }
};
#endif //ARTIFICIALNN_LAYER_H
