#ifndef NEURALNETWORKS_NDARRAY_H
#define NEURALNETWORKS_NDARRAY_H

#include <algorithm>
#include <functional>
#include <cstdarg>

using std::vector;
namespace NN{
    template <typename T>
    class Ndarray{
    public:
        // Конструкторы ---------------------------------
        Ndarray();
        explicit Ndarray(vector<size_t> shape);
        Ndarray(const Ndarray& copy);
        Ndarray(const Ndarray&& copy);

        // Методы класса --------------------------------
        // Перегрузки операторов ------------------------
        T& operator()(int index, ...);
        const T& operator()(int index, ...) const;

        // Деструктор -----------------------------------
        ~Ndarray();

        // Класс исключений -----------------------------
        class NdarrayExeption : public std::logic_error {
        public:
            NdarrayExeption(std::string s) : std::logic_error(s) {}

            ~NdarrayExeption() {}
        };

#ifdef TEST_Ndarray
    public:
        // Поля класса ----------------------------------
        vector<size_t > shape_;
        T* buffer;
        size_t size_;

        // Скрытые матоды класса ------------------------
        void init_buffer();
#else
    protected:
        // Поля класса ----------------------------------
        vector<size_t > shape_;
        T* buffer;
        size_t size_;

        // Скрытые матоды класса ------------------------
#endif
    };

    template<typename T>
    Ndarray<T>::Ndarray() : shape_({0}), buffer(nullptr), size_(0){

    }

    template<typename T>
    Ndarray<T>::~Ndarray() {
        delete[] buffer;
    }

    template<typename T>
    Ndarray<T>::Ndarray(vector<size_t> shape) : shape_(shape) {
        init_buffer();
    }

    template<typename T>
    Ndarray<T>::Ndarray(const Ndarray &copy) {
        this->shape_ = copy.shape_;
        init_buffer();
        for(size_t i = 0; i < size_; i++){
            buffer[i] = copy.buffer[i];
        }
    }

    template<typename T>
    void Ndarray<T>::init_buffer() {
        if((shape_.size() == 1)&&(shape_[0] == 0)){
            buffer = nullptr;
            size_ = 0;
        }else{
            size_t t = 1;
            for(auto n : shape_){
                t *= n;
            }
            size_ = t;
            buffer = new T[t]();
        }
    }

    template<typename T>
    Ndarray<T>::Ndarray(const Ndarray &&copy) {

    }

    template<typename T>
    T &Ndarray<T>::operator()(int index, ...) {
        va_list arguments;
        va_start(arguments, index);

        size_t index_ = 0;

        if((index < 0) || (index >= shape_[0])){
            throw Ndarray<T>::NdarrayExeption("Wrong index!");
        }
        size_t base = size_ / shape_[0];
        index_ += index * base;
        for(int i = 1; i < shape_.size(); i++){
            index = va_arg(arguments, int);
            if((index < 0) || (index >= shape_[0])){
                throw Ndarray<T>::NdarrayExeption("Wrong index!");
            }
            base /= shape_[i];
            index_ += index * base;
        }

        return buffer[index_];
    }

    template<typename T>
    const T &Ndarray<T>::operator()(int index, ...) const {
        va_list arguments;
        va_start(arguments, index);

        size_t index_ = 0;

        if((index < 0) || (index >= shape_[0])){
            throw Ndarray<T>::NdarrayExeption("Wrong index!");
        }
        size_t base = size_ / shape_[0];
        index_ += index * base;
        for(int i = 1; i < shape_.size(); i++){
            index = va_arg(arguments, int);
            if((index < 0) || (index >= shape_[0])){
                throw Ndarray<T>::NdarrayExeption("Wrong index!");
            }
            base /= shape_[i];
            index_ += index * base;
        }

        return buffer[index_];
    }
}

#endif //NEURALNETWORKS_NDARRAY_H
