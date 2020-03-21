#ifndef ARTIFICIALNN_MAXPOOLINGLAYER_H
#define ARTIFICIALNN_MAXPOOLINGLAYER_H

#include "Filters.h"

namespace ANN {
    
    template<typename T>
    class MaxpoolingLayer{
    public:
    MaxpoolingLayer(size_t n, size_t m);

    MaxpoolingLayer(const MaxpoolingLayer& copy);

    Matrix<T> passThrough(const Matrix<T>& in);
    Matrix<Matrix<T> > passThrough(const Matrix<Matrix<T> >& in);

    Matrix<T> BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error);
    Matrix<Matrix<T> > BackPropagation(const Matrix<Matrix<T>>& input, const Matrix<Matrix<T>>& output,
                                       const Matrix<Matrix<T> > &error);

    ~MaxpoolingLayer()= default;

    private:
        size_t n_, m_;
    };

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(size_t n, size_t m) {
        n_ = n;
        m_ = m;
    }

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(const MaxpoolingLayer &copy) {
        n_ = copy.n_;
        m_ = copy.m_;
    }

    template<typename T>
    Matrix<T> MaxpoolingLayer<T>::passThrough(const Matrix<T>& in) {
        return Filter<T>::Pooling(in, n_, m_);
    }

    template<typename T>
    Matrix<T>
    MaxpoolingLayer<T>::BackPropagation(const Matrix<T> &input, const Matrix<T> &output, const Matrix<T> &error) {
        return ANN::BackPropagation(input, output, error, n_, m_);
    }

    template<typename T>
    Matrix<Matrix<T>>
    MaxpoolingLayer<T>::BackPropagation(const Matrix<Matrix<T>>& input, const Matrix<Matrix<T>>& output,
                                        const Matrix<Matrix<T> > &error) {
        return ANN::BackPropagation(input, output, error, n_, m_);
    }

    template<typename T>
    Matrix<Matrix<T>> MaxpoolingLayer<T>::passThrough(const Matrix<Matrix<T>>& in) {
        Matrix<Matrix<T> > result(in.getN(), in.getM());
        for(size_t i = 0; i < in.getN(); i++){
            for(size_t j = 0; j < in.getM(); j++){
                result[i][j] = MaxpoolingLayer<T>::passThrough(in[i][j]);
            }
        }
        return result;
    }

}
#endif //ARTIFICIALNN_MAXPOOLINGLAYER_H
