#ifndef ARTIFICIALNN_CONVOLUTIONLAYER_H
#define ARTIFICIALNN_CONVOLUTIONLAYER_H

#include "Filters.h"
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
        void GradDes(ImpulsGrad<T>& G, const Tensor<T>& input);

        void getFromFile(const std::string& file_name);
        void saveToFile(const std::string& file_name);

        ~ConvolutionLayer()= default;

    private:
        const Init<T>* I_;
        size_t step_;
        Matrix<Tensor<T> > history;
    };

    template<typename T>
    ConvolutionLayer<T>::ConvolutionLayer(size_t number_filters, size_t height, size_t width, size_t depth,
            const Init<T> &init, size_t step): Matrix<Filter<T> >(1, number_filters), Layer<T>("ConvolutionLayer") {
        I_ = &init;
        step_ = step;
        history = Matrix<Tensor<T> >(1, number_filters);
        for(size_t i = 0; i < number_filters; i++){

            this->arr[0][i] = Filter<T>(height, width, depth);
            this->history[0][i] = Tensor<T>(height, width, depth);
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

    // TODO: rewrite
    template<typename T>
    void ConvolutionLayer<T>::getFromFile(const std::string &file_name) {
        ANN::getFiltresTextFile(*this, file_name);
    }

    // TODO: rewrite
    template<typename T>
    void ConvolutionLayer<T>::saveToFile(const std::string &file_name) {
        ANN::saveFiltersTextFile(*this, file_name);
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
        return ANN::BackPropagation(input, error, *this, step_);
    }

    template<typename T>
    void
    ConvolutionLayer<T>::GradDes(ImpulsGrad<T> &G, const Tensor<T>& input) {
        ANN::GradDes(G, *this, history);
    }

}
#endif //ARTIFICIALNN_CONVOLUTIONLAYER_H
