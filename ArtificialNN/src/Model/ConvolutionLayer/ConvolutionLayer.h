#ifndef ARTIFICIALNN_CONVOLUTIONLAYER_H
#define ARTIFICIALNN_CONVOLUTIONLAYER_H

#include "Filters.h"
#include "Initializers.h"
#include "Data.h"

namespace ANN{
    template<typename T>
    class ConvolutionLayer : public Matrix<Filter<T> >{
    public:
        ConvolutionLayer(size_t number_filters, std::pair<size_t, size_t> size_of_filters,
                const Init<T>& init, size_t step);

        ConvolutionLayer(const ConvolutionLayer& copy);

        void getFiltersFromFile(const std::string& file_name);
        void saveFiltersToFile(const std::string& file_name);

        Matrix<Matrix<T> > passThrough(const Matrix<Matrix<T> >& in);

        Matrix<Matrix<T> > BackPropagation(const Matrix<Matrix<T> >& error);

        void GradDes(Grad<T>& G, const Matrix<Matrix <T> >& input, const Matrix<Matrix <T> >& error);

        ~ConvolutionLayer()= default;

    private:
        const Init<T>* I_;
        size_t step_;
        Matrix<Matrix<T> > history;
    };

    template<typename T>
    ConvolutionLayer<T>::ConvolutionLayer(const ConvolutionLayer &copy) : Matrix<Filter<T> >(copy) {
        I_ = copy.I_;
        step_ = copy.step_;
    }

    template<typename T>
    ConvolutionLayer<T>::ConvolutionLayer(size_t number_filters, std::pair<size_t, size_t> size_of_filters,
            const Init<T> &init, size_t step): Matrix<Filter<T> >(1, number_filters) {
        I_ = &init;
        step_ = step;
        for(size_t i = 0; i < number_filters; i++){
            this->arr[0][i] = Filter<T>(size_of_filters.first, size_of_filters.second);
            for(size_t x = 0; x < size_of_filters.first; x++){
                for(size_t y = 0; y < size_of_filters.second; y++){
                    this->arr[0][i][x][y] = (*(this->I_))();
                }
            }
        }
    }

    template<typename T>
    void ConvolutionLayer<T>::getFiltersFromFile(const std::string &file_name) {
        ANN::getFiltresTextFile(*this, file_name);
    }

    template<typename T>
    void ConvolutionLayer<T>::saveFiltersToFile(const std::string &file_name) {
        ANN::saveFiltersTextFile(*this, file_name);
    }

    template<typename T>
    Matrix<Matrix<T>> ConvolutionLayer<T>::passThrough(const Matrix<Matrix<T> > &in) {
        Matrix<Matrix<T> > result(1, in.getM()*this->m);
        for(size_t i = 0; i < result.getM(); i++){
            result[0][i] = this->arr[0][i%this->m].Svertka(in[0][i/this->m], 1);
        }
        return result;
    }

    template<typename T>
    Matrix<Matrix<T> > ConvolutionLayer<T>::BackPropagation(const Matrix<Matrix<T>> &error) {
        return ANN::BackPropagation(error, *this, step_);
    }

    template<typename T>
    void ConvolutionLayer<T>::GradDes(Grad<T> &G, const Matrix<Matrix<T>> &input, const Matrix<Matrix<T>> &error) {
        ANN::GradDes(G, input, error, *this, step_);
    }

}
#endif //ARTIFICIALNN_CONVOLUTIONLAYER_H
