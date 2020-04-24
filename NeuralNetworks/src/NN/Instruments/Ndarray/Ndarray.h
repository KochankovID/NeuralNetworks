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
        Ndarray();  // По умолчанию
        explicit Ndarray(const vector<size_t>& shape);  // Инициализатор (создает н-мерный массив формы shape)
        Ndarray(const vector<size_t>& shape, const vector<T>& array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const Ndarray& copy);  // Копирования
        Ndarray(const Ndarray&& copy);  // Мув коструктор

        // Методы класса --------------------------------
        vector<size_t > shape() const { return shape_; }; // Возвращает форму массива
        size_t argmax() const;  // Возвращает индес наибольшего элемента в массиве
        Ndarray<vector<size_t>> argmax(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси

        // Перегрузки операторов ------------------------
        T& operator()(const std::vector<size_t>& indices);
        const T& operator()(const std::vector<size_t>& indices) const;

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
        vector<size_t > bases_;
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
        vector<size_t > bases_;

        // Скрытые матоды класса ------------------------
#endif
    };

    template<typename T>
    Ndarray<T>::Ndarray() : shape_({0}), buffer(nullptr), size_(0), bases_({0}){

    }

    template<typename T>
    Ndarray<T>::~Ndarray() {
        delete[] buffer;
    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t>& shape) : shape_(shape), bases_(shape_.size()) {
        init_buffer();
    }

    template<typename T>
    Ndarray<T>::Ndarray(const Ndarray &copy) {
        this->shape_ = copy.shape_;
        this->size_ = copy.size_;
        this->bases_ = copy.bases_;
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
            bases_[0] = 0;
        }else{
            size_t t = 1;
            for(int i = shape_.size()-1; i >= 0; i--){
                t *= shape_[i];
                bases_[i] = t/shape_[i];
            }
            size_ = t;
            buffer = new T[t]();
        }
    }

    template<typename T>
    Ndarray<T>::Ndarray(const Ndarray &&copy) {
        this->shape_ = copy.shape_;
        this->size_ = copy.size_;
        this->bases_ = copy.bases_;
        buffer = copy.buffer;
        copy.buffer = nullptr;
    }

    template<typename T>
    T &Ndarray<T>::operator()(const std::vector<size_t>& indices) {
        if(indices.size() != shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong index!");
        }
        size_t index_ = 0;
        for(int i = 0; i < shape_.size(); i++){
            if((indices[i] < 0) || (indices[i] >= shape_[i])){
                throw Ndarray<T>::NdarrayExeption("Wrong index!");
            }
            index_ += indices[i] * bases_[i];
        }
        return buffer[index_];
    }

    template<typename T>
    const T &Ndarray<T>::operator()(const std::vector<size_t> &indices) const {
        if(indices.size() != shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong index!");
        }
        size_t index_ = 0;
        for(int i = 0; i < shape_.size(); i++){
            if((indices[i] < 0) || (indices[i] >= shape_[i])){
                throw Ndarray<T>::NdarrayExeption("Wrong index!");
            }
            index_ += indices[i] * bases_[i];
        }
        return buffer[index_];
    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t> &shape, const vector<T>& array) : shape_(shape), bases_(shape_.size()){
        init_buffer();
        if(array.size() != size_){
            throw Ndarray<T>::NdarrayExeption("Wrong size of vector!");
        }
        for(int i = 0; i < size_; i++){
            buffer[i] = array[i];
        }
    }

    template<typename T>
    size_t Ndarray<T>::argmax() const {
        size_t max_index = 0;
        for(int i = 1; i < size_; i++){
            if(buffer[i] > buffer[max_index]){
                max_index = i;
            }
        }
        return max_index;
    }

    template<typename T>
    Ndarray<vector<size_t>> Ndarray<T>::argmax(size_t axis) const {
        if(axis > shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }

        Ndarray<T> new_arr(this->shape().erase(axis));
        vector<size_t > index(new_arr.shape_.size());

        for(int i = 0; i < index.size(); i++){
            for(int j = 0; j < new_arr.shape_[i]; j++){

            }
        }
    }
}

#endif //NEURALNETWORKS_NDARRAY_H
