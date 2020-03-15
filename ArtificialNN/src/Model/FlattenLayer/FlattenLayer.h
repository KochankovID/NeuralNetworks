#ifndef ARTIFICIALNN_FLATTENLAYER_H
#define ARTIFICIALNN_FLATTENLAYER_H

#include "Layer.h"
#include "DenceLayers.h"

namespace ANN {

    template<typename T>
    class FlattenLayer{
    public:
        FlattenLayer(){};

        Matrix<T> passThrough(const Matrix<T> &in);
        Matrix<T> passThrough(const Matrix<Matrix<T>>& in);

        Matrix<T> passBack(const Matrix<T>& in);
        Matrix<Matrix<T>> passBack(const Matrix<T>& in, size_t n, size_t m, size_t ni, size_t mi);
        Matrix<Matrix<T>> passBack(const DenceLayer<T>& in,  size_t n, size_t m, size_t ni, size_t mi);

        ~FlattenLayer();
    private:

    };

    template<typename T>
    FlattenLayer<T>::~FlattenLayer() = default;

    template<typename T>
    Matrix<T> FlattenLayer<T>::passThrough(const Matrix<T> &in) {
        Matrix<T> out(1, in.getN()*in.getM());
        for(size_t i = 0; i < in.getN(); i++){
            for(size_t j = 0; j < in.getM(); j++){
                out[0][i*in.getM() + j] = out[i][j];
            }
        }
        return out;
    }

    template<typename T>
    Matrix<T> FlattenLayer<T>::passBack(const Matrix<T> &in) {
        Matrix<T> m_(1, in.getN()*in.getM());
        for(size_t i = 0; i < in.getN(); i++){
            for(size_t j = 0; j < in.getM(); j++){
                this->m_[0][i*in.getM() + j] = in[i][j];
            }
        }
        return m_;
    }

    template<typename T>
    Matrix<T> FlattenLayer<T>::passThrough(const Matrix<Matrix<T> > &in) {
        size_t n = in.getN(), m = in.getM(), ni = in[0][0].getN(), mi = in[0][0].getM();

        Matrix<T> m_(1, n*m*in[0][0].getN()*in[0][0].getM());
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                for(size_t x = 0; x < ni; x++){
                    for(size_t y = 0; y < mi; y++){
                        m_[0][i*m + j*ni + x*mi +y] = in[i][j][x][y];
                    }
                }
            }
        }
        return m_;
    }

    template<typename T>
    Matrix<Matrix<T>> FlattenLayer<T>::passBack(const Matrix<T> &in, size_t n, size_t m, size_t ni, size_t mi) {
        Matrix<T> m_(n, m);
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                m_ = Matrix<T>(ni,mi);
            }
        }
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                for(size_t x = 0; x < ni; x++){
                    for(size_t y = 0; y < mi; y++){
                        m_[i][j][x][y]= in[0][i*m + j*ni + x* mi + y];
                    }
                }
            }
        }
        return m_;
    }

    template<typename T>
    Matrix<Matrix<T> > FlattenLayer<T>::passBack(const DenceLayer<T> &in, size_t n, size_t m, size_t ni, size_t mi) {
        Matrix<Matrix<T>> m_(n, m);

        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                m_[i][j] = Matrix<T>(ni,mi);
                m_[i][j].Fill(0);
            }
        }
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                for(size_t x = 0; x < ni; x++){
                    for(size_t y = 0; y < mi; y++){
                        for(size_t ii = 0; ii < in[0][i*m + j*ni + x* mi + y].getN(), i++){
                            for(size_t jj = 0; jj < in[0][i*m + j*ni + x* mi + y].getM(); jj++){
                                m_[i][j][x][y] += in[0][i*m + j*ni + x* mi + y][ii][jj] * in[0][i*m + j*ni + x* mi + y].GetD();
                            }
                        }
                    }
                }
            }
        }
        return m_;
    }


}

#endif //ARTIFICIALNN_FLATTENLAYER_H
