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
        Ndarray(const vector<size_t>& shape, const T* array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const Ndarray& copy);  // Копирования
        Ndarray(Ndarray&& copy);  // Мув коструктор

        // Методы класса --------------------------------
        vector<size_t > shape() const { return shape_; }; // Возвращает форму массива
        size_t argmax() const;  // Возвращает индес наибольшего элемента в массиве
        Ndarray<size_t > argmax(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси
        size_t argmin() const;  // Возвращает индес наибольшего элемента в массиве
        T max();  // Возвращает наибольший элемент в массиве
        T min();  // Возвращает наименьший элемент в массиве
        Ndarray<size_t > argmin(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси
        vector<size_t > get_nd_index(size_t indes) const; // Преобразует 1D индекс в ND
        void fill(const T& value); // Заполняет массив указанным значением
        Ndarray<T> flatten();

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
    Ndarray<T>::Ndarray(Ndarray &&copy) {
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
    Ndarray<T>::Ndarray(const vector<size_t> &shape, const T *array) : shape_(shape), bases_(shape_.size()) {
        init_buffer();
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
    Ndarray<size_t > Ndarray<T>::argmax(size_t axis) const {
        if(axis > shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin()+axis);
        Ndarray<size_t > new_arr(shape_t);
        vector<size_t > index(new_arr.shape_.size());

        for(int i = 0; i < new_arr.size_; i++){
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto max_index = index;
            for(int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if((*this)(index) > (*this)(max_index)){
                    max_index = index;
                }
            }
            new_arr.buffer[i] = max_index[axis];
        }
        return new_arr;
    }

    template<typename T>
    vector<size_t> Ndarray<T>::get_nd_index(size_t index) const {
        if(index > size_){
            throw Ndarray<T>::NdarrayExeption("Wrong index!");
        }
        vector<size_t > index_;
        size_t tmp;
        for(size_t i = 0; i < bases_.size(); i++){
            tmp = index /bases_[i];
            index_.push_back(tmp);
            index -= tmp * bases_[i];
        }
        return index_;
    }

    template<typename T>
    size_t Ndarray<T>::argmin() const {
        size_t min_index = 0;
        for(int i = 1; i < size_; i++){
            if(buffer[i] < buffer[min_index]){
                min_index = i;
            }
        }
        return min_index;
    }

    template<typename T>
    Ndarray<size_t> Ndarray<T>::argmin(size_t axis) const {
        if(axis > shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin()+axis);
        Ndarray<size_t > new_arr(shape_t);
        vector<size_t > index(new_arr.shape_.size());

        for(int i = 0; i < new_arr.size_; i++){
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto min_index = index;
            for(int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if((*this)(index) < (*this)(min_index)){
                    min_index = index;
                }
            }
            new_arr.buffer[i] = min_index[axis];
        }
        return new_arr;
    }

    template<typename T>
    T Ndarray<T>::max() {
        size_t max_index = 0;
        for(int i = 1; i < size_; i++){
            if(buffer[i] > buffer[max_index]){
                max_index = i;
            }
        }
        return buffer[max_index];
    }

    template<typename T>
    T Ndarray<T>::min() {
        size_t min_index = 0;
        for(int i = 1; i < size_; i++){
            if(buffer[i] < buffer[min_index]){
                min_index = i;
            }
        }
        return buffer[min_index];
    }

    template<typename T>
    void Ndarray<T>::fill(const T& value) {
        std::fill(buffer, buffer+size_, value);
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::flatten() {
        return Ndarray<T>({size_}, this->buffer);
    }

}

#endif //NEURALNETWORKS_NDARRAY_H
