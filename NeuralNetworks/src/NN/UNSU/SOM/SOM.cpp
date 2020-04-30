#include "SOM.h"

using std::to_string;

NN::SOM::SOM(int x, int y, int input_length, double learning_rate, double radius) : learning_rate_(learning_rate), radius_(radius){
    if((x <= 0)||(y <= 0)){
        throw SOMExeption("Size of SOM can't be negative or null:  x=" + to_string(x) + " y=" + to_string(y));
    }
    if(input_length <= 0){
        throw SOMExeption("Input length can't be negative or null:  input_length=" + to_string(input_length));
    }

    weights_ = Ndarray<double >(3, x, y, input_length);
}

NN::SOM::SOM(const NN::SOM &copy) : learning_rate_(copy.learning_rate_), radius_(copy.radius_), weights_(copy.weights_){

}

void NN::SOM::random_weights_init(const NN::SimpleInitializator<double> &init) {
    for(size_t i = 0; i < weights_.size(); i++){
        weights_[i] = init();
    }
}

void NN::SOM::train_random(const NN::Ndarray<double> &data, int num_iteration) {
    if(data.shape().size() != 3){
        throw SOMExeption("Shape of data isn't equivalent 3:  data.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[2] != weights_.shape()[2]){
        throw SOMExeption("Axis 2 of data isn't equivalent input_length:  data.shape()[2]=" +
        to_string(data.shape()[2]) + " input_length=" + to_string(weights_.shape()[2]));
    }
    for(size_t i = 0; i < )
}


