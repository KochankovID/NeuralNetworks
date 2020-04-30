#include "SOM.h"

using std::to_string;

NN::SOM::SOM(int x, int y, int input_length, double learning_rate, double radius) : learning_rate_(learning_rate), radius_(radius){
    if((x <= 0)||(y <= 0)){
        throw SOMExeption("Size of SOM can't be negative or null:  x=" + to_string(x) + " y=" + to_string(y));
    }
    if(input_length <= 0){
        throw SOMExeption("Input length can't be negative or null:  input_length=" + to_string(input_length));
    }

    vector<size_t > shape;
    shape.push_back(x);
}
