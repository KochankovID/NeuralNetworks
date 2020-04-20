#ifndef ARTIFICIALNN_CONVOLUTIONLAYER_H
#define ARTIFICIALNN_CONVOLUTIONLAYER_H

#include "Filter.h"
#include "Layer.h"
#include "Initializers.h"
#include "Data.h"

namespace ANN{
    template<typename T>
    class ConvolutionLayer : public Matrix<Filter<T> >, public Layer<T>{
    public:
        ConvolutionLayer(size_t number_filters, size_t height, size_t width, size_t depth,
                const Init<T>& init, size_t step);
        ConvolutionLayer(const ConvolutionLayer& copy);

        Tensor<T> passThrough(const Tensor<T>& in);
        Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor<T>& input);
        void GradDes(const ImpulsGrad<T>& G, const Tensor<T>& input);

        void saveToFile(std::ofstream& file);
        void getFromFile(std::ifstream& file);

        ~ConvolutionLayer()= default;

#ifdef TEST_ConvLayer
    public:
        const Init<T>* I_;
        size_t step_;
        Matrix<Filter<T> > history;
        Tensor<T> error_;
#else
    private:
        const Init<T>* I_;
        size_t step_;
        Matrix<Filter<T> > history;
        Tensor<T> error_;
#endif
    };

#define D_ConvolutionLayer ConvolutionLayer<double>
#define F_ConvolutionLayer ConvolutionLayer<float>
#define I_ConvolutionLayer ConvolutionLayer<int>

    template<typename T>
    ConvolutionLayer<T>::ConvolutionLayer(size_t number_filters, size_t height, size_t width, size_t depth,
            const Init<T> &init, size_t step): Matrix<Filter<T> >(1, number_filters), Layer<T>("ConvolutionLayer") {
        I_ = &init;
        step_ = step;
        history = Matrix<Filter<T> >(1, number_filters);
        for(size_t i = 0; i < number_filters; i++){

            this->arr[0][i] = Filter<T>(height, width, depth);
            this->history[0][i] = Filter<T>(height, width, depth);
            this->history[0][i].Fill(0);
            for(size_t z = 0; z < depth; z++) {
                for (size_t x = 0; x < height; x++) {
                    for (size_t y = 0; y < width; y++) {
                        this->arr[0][i][z][x][y] = (*(this->I_))();
                    }
                }
            }
        }
    }

    template<typename T>
    ConvolutionLayer<T>::ConvolutionLayer(const ConvolutionLayer &copy) : Matrix<Filter<T> >(copy), Layer<T>(copy) {
        I_ = copy.I_;
        step_ = copy.step_;
        history = copy.history;
    }

    template<typename T>
    Tensor<T> ConvolutionLayer<T>::passThrough(const Tensor<T> &in) {
        auto size = Filter<T>::convolution_result_creation(in.getHeight(), in.getWidth(),
                this->arr[0][0].getHeight(), this->arr[0][0].getWidth(), step_);

        Tensor<T> result(size.first, size.second, this->getM());

        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] = this->arr[0][i].Svertka(in, 1);
        }
        return result;
    }

    template<typename T>
    Tensor<T> ConvolutionLayer<T>::BackPropagation(const Tensor<T>& error, const Tensor<T>& input) {
        error_ = error;
        return ANN::BackPropagation(error, *this, step_);
    }

    template<typename T>
    void
    ConvolutionLayer<T>::GradDes(const ImpulsGrad<T> &G, const Tensor<T>& input) {
        ANN::GradDes(G, input, *this, error_ ,step_, history);
    }

    template<typename T>
    void ConvolutionLayer<T>::saveToFile(std::ofstream &file) {
        file << *this << endl << history << endl << error_;
    }

    template<typename T>
    void ConvolutionLayer<T>::getFromFile(std::ifstream &file) {
        file >> *this;
        file >> history;
        file >> error_;
    }

}
#endif //ARTIFICIALNN_CONVOLUTIONLAYER_H
