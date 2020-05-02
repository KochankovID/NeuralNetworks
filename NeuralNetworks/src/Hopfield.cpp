#include "Hopfield.h"

using std::to_string;

NN::Hopfield::Hopfield(int number_neyrons) {
    if(number_neyrons <=0){
        throw HopfieldExeption("Wrong amount of neyrons: number_neyrons=" + to_string(number_neyrons));
    }
    vector<int> shape;
    shape.push_back(number_neyrons);
    shape.push_back(number_neyrons);
    weights_ = Ndarray<int>(shape);
    weights_.fill(0);
}

NN::Hopfield::Hopfield(const NN::Hopfield &copy) : weights_(copy.weights_){

}

void NN::Hopfield::train(const NN::Ndarray<int> &data) {
    if(data.shape().size() != 2){
        throw HopfieldExeption("Shape of pattern must be equal 1 : x.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[1] != weights_.shape()[1]){
        throw HopfieldExeption("Shape of pattern must be equal number_neyron : x.shape()[0]=" + to_string(data.shape().size()) +
                               " number_neyrons=" + to_string(weights_.shape()[0]));
    }
    for(size_t j = 0; j < data.shape()[0]; j++) {
        for (size_t i = 0; i < data.shape()[1]; i++) {
            if ((data(j, i) < -1) || (data(j, i) > 1)) {
                throw HopfieldExeption("Data must be 1 or -1 : x[" + to_string(i) + "]=" + to_string(data[i]));
            }
        }
    }
    for(size_t j = 0; j < data.shape()[0]; j++) {
        for (size_t x = 0; x < weights_.shape()[0]; x++) {
            for (size_t y = 0; y < weights_.shape()[1]; y++) {
                if (x == y) {
                    weights_(x, y) = 0;
                } else {
                    weights_(x, y) += data(j, x) * data(j, y);
                }
            }
        }
    }

    for (size_t x = 0; x < weights_.shape()[0]; x++) {
        for (size_t y = 0; y < weights_.shape()[1]; y++) {
            weights_(x,y) /= data.shape()[0];
        }
    }
}

NN::Ndarray<int> NN::Hopfield::fit(const NN::Ndarray<int>& sample) {
    if(sample.shape().size() != 1){
        throw HopfieldExeption("Shape of pattern must be equal 1 : x.shape().size()=" + to_string(sample.shape().size()));
    }
    if(sample.shape()[0] != weights_.shape()[0]){
        throw HopfieldExeption("Shape of pattern must be equal number_neyron : x.shape()[0]=" + to_string(sample.shape().size()) +
                               " number_neyrons=" + to_string(weights_.shape()[0]));
    }
    for(size_t i = 0; i < sample.shape()[0]; i++){
        if((sample[i] < -1)||(sample[i] > 1)){
            throw HopfieldExeption("Data must be 1 or -1 : x[" + to_string(i) + "]=" + to_string(sample[i]));
        }
    }
    auto begin_sample = sample;
    auto out_sample = sample;
    auto zeros = sample;
    zeros.fill(0);
    do {
        begin_sample = out_sample;
        for (size_t i = 0; i < weights_.shape()[0]; i++) {
            int summ = 0;
            for (size_t j = 0; j < weights_.shape()[0]; j++) {
                summ += weights_(i, j) * out_sample[j];
            }
            if (summ < 0) {
                out_sample[i] = -1;
            }
            if (summ > 0) {
                out_sample[i] = 1;
            }
        }
    }while (begin_sample - out_sample == zeros);
    return out_sample;
}
