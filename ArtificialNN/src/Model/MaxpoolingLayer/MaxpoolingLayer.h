#ifndef ARTIFICIALNN_MAXPOOLINGLAYER_H
#define ARTIFICIALNN_MAXPOOLINGLAYER_H

#include "Filters.h"
#include "Layer.h"

namespace ANN {
    
    template<typename T>
    class MaxpoolingLayer : public Layer<T> {
    public:
        MaxpoolingLayer(size_t n, size_t m);

        MaxpoolingLayer(const MaxpoolingLayer &copy);

        Tensor<T> passThrough(const Tensor<T> &in);

        Tensor<T> BackPropagation(const Tensor<T> &error, const Tensor<T> &in);

        void GradDes(const ImpulsGrad<T> &G, const Tensor<T> &in) {};

        void saveToFile(std::ofstream& file);
        void getFromFile(std::ifstream& file);

        ~MaxpoolingLayer() = default;

#ifdef TEST_MaxLayer
        // TODO: Not neesesary
    public:
        size_t n_, m_;
        Tensor<T> output;
        Matrix<T> passThrough(const Matrix<T> &in);
#else
        // TODO: Not neesesary
    private:
        size_t n_, m_;
        Tensor<T> output;
        Matrix<T> passThrough(const Matrix<T> &in);
#endif
    };

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
        return ANN::BackPropagation(in, output, error, n_, m_);
    }

    template<typename T>
    void MaxpoolingLayer<T>::saveToFile(std::ofstream &file) {

    }

    template<typename T>
    void MaxpoolingLayer<T>::getFromFile(std::ifstream &file) {

    }

}
#endif //ARTIFICIALNN_MAXPOOLINGLAYER_H
