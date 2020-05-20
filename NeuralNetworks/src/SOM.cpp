#include "SOM.h"
#include <time.h>

using std::to_string;

NN::SOM::SOM(int x, int y, int input_length, double learning_rate, double radius) : learning_rate_(learning_rate), radius_(radius){
    if((x <= 0)||(y <= 0)){
        throw SOMExeption("Size of SOM can't be negative or null:  x=" + to_string(x) + " y=" + to_string(y));
    }
    if(input_length <= 0){
        throw SOMExeption("Input length can't be negative or null:  input_length=" + to_string(input_length));
    }
    vector<int > shape;
    shape.push_back(x);
    shape.push_back(y);
    shape.push_back(input_length);
    weights_ = Ndarray<double >(shape);
}

NN::SOM::SOM(const NN::SOM &copy) : learning_rate_(copy.learning_rate_), radius_(copy.radius_),
weights_(copy.weights_), history_(copy.history_){

}

void NN::SOM::random_weights_init(const NN::Init<double> &init) {
    for(size_t i = 0; i < weights_.size(); i++){
        weights_[i] = init();
    }
}

void NN::SOM::train(const NN::Ndarray<double> &data, int num_iteration) {
    if(data.shape().size() != 2){
        throw SOMExeption("Shape of data isn't equivalent 2:  data.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[1] != weights_.shape()[2]){
        throw SOMExeption("Axis 1 of data isn't equivalent input_length:  data.shape()[2]=" +
        to_string(data.shape()[1]) + " input_length=" + to_string(weights_.shape()[2]));
    }
    history_ = Ndarray<Ndarray<double >>(1, num_iteration);
    for(size_t epoch = 0; epoch < num_iteration; epoch++){
        for(size_t example = 0; example < data.shape()[0]; example++){
            auto data_exmpl = data.subArray(1,example);
            auto index_winner = winner(data_exmpl);
            update(data_exmpl, index_winner,example,num_iteration*data.shape()[0]);
            history_[epoch] = weights_;
        }
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
    srand(time(0));
    history_ = Ndarray<Ndarray<double >>(1, num_iteration);
    for(size_t epoch = 0; epoch < num_iteration; epoch++){
        size_t example = (double(rand()) / RAND_MAX) * data.shape()[0];
        auto data_exmpl = data.subArray(1, example);
        auto index_winner = winner(data_exmpl);
        update(data_exmpl, index_winner, epoch, num_iteration);
        history_[epoch] = weights_;
    }
}

vector<int> NN::SOM::winner(const NN::Ndarray<double> &data) const {
    if(data.shape().size() != 1){
        throw SOMExeption("Shape of data isn't equivalent 1:  data.shape().size()=" + to_string(data.shape().size()));
    }
    if(data.shape()[0] != weights_.shape()[2]){
        throw SOMExeption("Axis 1 of data isn't equivalent input_length:  data.shape()[2]=" +
                          to_string(data.shape()[1]) + " input_length=" + to_string(weights_.shape()[2]));
    }
    vector<int> shape;
    shape.push_back(weights_.shape()[0]);
    shape.push_back(weights_.shape()[1]);
    Ndarray<double > results(shape);
    for(size_t x = 0; x < weights_.shape()[0]; x++){
        for(size_t y = 0; y < weights_.shape()[1]; y++){
            results(x,y) = euclidean_distance(data, weights_.subArray(2,x,y));
        }
    }
    return Ndarray<double>::get_nd_index(results.argmin(), shape);
}

double NN::SOM::euclidean_distance(const NN::Ndarray<double> &vect_1, const NN::Ndarray<double> &vect_2) {
    auto result = vect_1 - vect_2;
    result *= result;
    return std::sqrt(std::accumulate(result.begin(), result.end(), 0.0));
}

void NN::SOM::update(const NN::Ndarray<double> &data, const vector<int> &winner, int iteration, int num_iteration) {
    double cur_learn_rate = decay_function(learning_rate_, iteration, num_iteration);
    double cur_radius = decay_function(radius_, iteration, num_iteration);
    double raius_sq = cur_radius * cur_radius;
    for(int x = 0; x < weights_.shape()[0]; x++){
        for(int y = 0; y < weights_.shape()[1]; y++){
            double dist_sq = (x - winner[0])*(x - winner[0])+(y - winner[1])*(y - winner[1]);
            if (dist_sq <= raius_sq){
                double influence = exp(-(dist_sq)/(2*raius_sq));  // Уменьшение в зависимости от расстояния
                for(int z = 0; z < weights_.shape()[2]; z++){
                    weights_(x,y,z) += cur_learn_rate * influence * (data[z] - weights_(x,y,z));
                }
            }
        }
    }
}

double NN::SOM::decay_function(double x, int t, int max_iter) {
    return x*exp(-double(t)/max_iter);
}
