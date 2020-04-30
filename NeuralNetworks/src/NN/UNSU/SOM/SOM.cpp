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
    if(data.shape().size() != 2){
        throw SOMExeption("Shape of data isn't equivalent 2:  data.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[1] != weights_.shape()[2]){
        throw SOMExeption("Axis 1 of data isn't equivalent input_length:  data.shape()[2]=" +
        to_string(data.shape()[1]) + " input_length=" + to_string(weights_.shape()[2]));
    }
}

vector<int> NN::SOM::winner(const NN::Ndarray<double> &data) {
    if(data.shape().size() != 1){
        throw SOMExeption("Shape of data isn't equivalent 1:  data.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[1] != weights_.shape()[2]){
        throw SOMExeption("Axis 1 of data isn't equivalent input_length:  data.shape()[2]=" +
                          to_string(data.shape()[1]) + " input_length=" + to_string(weights_.shape()[2]));
    }
    Ndarray<double > results(2,weights_.shape()[0], weights_.shape()[1]);
    for(size_t x = 0; x < weights_.shape()[0]; x++){
        for(size_t y = 0; y < weights_.shape()[1]; y++){
            results(x,y) = euclidean_distance(data, weights_.subArray(2,x,y));
        }
    }
    return weights_.get_nd_index(results.argmin());
}

double NN::SOM::euclidean_distance(const NN::Ndarray<double> &data_exmp, const NN::Ndarray<double> &weights_neyron) {
    auto result = data_exmp - weights_neyron;
    result *= result;
    return std::sqrt(std::accumulate(result.begin(), result.end(), 0.0));
}