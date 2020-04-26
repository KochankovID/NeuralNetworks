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
        explicit Ndarray(const vector<size_t >& shape);  // Инициализатор (создает н-мерный массив формы shape)
        Ndarray(const vector<size_t>& shape, const vector<T>& array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const vector<size_t>& shape, const T* array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const Ndarray& copy);  // Копирования
        Ndarray(Ndarray&& copy);  // Мув коструктор

        // Методы класса --------------------------------
        vector<size_t > shape() const { return shape_; }; // Возвращает форму массива
        size_t argmax() const;  // Возвращает индес наибольшего элемента в массиве
        size_t argmin() const;  // Возвращает индес наибольшего элемента в массиве
        Ndarray<size_t > argmax(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси
        Ndarray<size_t > argmin(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси
        T max();  // Возвращает наибольший элемент в массиве
        T min();  // Возвращает наименьший элемент в массиве
        Ndarray<T > max(size_t axis) const;  // Возвращает массив наибольших значений взятых по указанной оси
        Ndarray<T > min(size_t axis) const;  // Возвращает массив наибольших значений взятых по указанной оси
        vector<size_t > get_nd_index(size_t index) const;  // Преобразует 1D индекс в ND
        size_t get_1d_index(vector<size_t > index) const;  // Преобразует 1D индекс в ND
        void fill(const T& value);  // Заполняет массив указанным значением
        Ndarray<T> flatten();  // Возвращает копию массива в 1d измерении
        void reshape(const vector<int > &shape);  // Меняет "форму массива" без изменения его элементов
        void sort(bool order = true);  // Сортирует массив в 1d
        void sort(size_t axis); // Сортирует массив вдоль выбранной оси


        // Перегрузки операторов ------------------------
        T& operator()(const std::vector<size_t>& index);
        const T& operator()(const std::vector<size_t>& index) const;
        T& operator[](int index);
        const T& operator[](int index) const;

        // Итератор -------------------------------------
        class NdarrayIterator;
        friend class NdarrayIterator;
        class NdarrayIterator : std::iterator<std::random_access_iterator_tag(), T>{
        public:
            // Конструкторы -----------------------------
            NdarrayIterator(Ndarray<T> &ndarray, size_t axis, const vector<size_t > &start_index);

            // Перегрузки операторов --------------------
            NdarrayIterator& operator++();
            NdarrayIterator& operator--();
            NdarrayIterator& operator+=(int n);
            NdarrayIterator& operator-=(int n);
            NdarrayIterator operator+(int n) const;
            NdarrayIterator operator-(int n) const;
            NdarrayIterator operator-(const NdarrayIterator &iter) const;
            T& operator*();
            const T& operator*() const;
            T& operator[](int index);
            const T& operator[](int index) const;
            bool operator<(const NdarrayIterator &iter) const;
            bool operator<=(const NdarrayIterator &iter) const;
            bool operator>(const NdarrayIterator &iter) const;
            bool operator>=(const NdarrayIterator &iter) const;

            friend typename Ndarray<T>::NdarrayIterator operator+(int n, const typename Ndarray<T>::NdarrayIterator &iter) {
                auto copy = iter;
                copy.index_[iter.axis_]+=n;
                return copy;
            }

            friend typename Ndarray<T>::NdarrayIterator operator-(int n, const typename Ndarray<T>::NdarrayIterator &iter) {
                auto copy = iter;
                copy.index_[iter.axis_]-=n;
                return copy;
            }


        private:
            Ndarray<T> &ndarray_;
            vector<size_t > index_;
            size_t axis_;

            void is_the_same(const NdarrayIterator& iter) const;
        };

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
        void is_in_range(int index) const;
        void is_in_range(vector<size_t > index) const;
#else
    protected:
        // Поля класса ----------------------------------
        vector<size_t > shape_;
        vector<size_t > bases_;
        T* buffer;
        size_t size_;
        void is_in_range(int index) const;
        void is_in_range(vector<size_t > index) const;
        // Скрытые матоды класса ------------------------
#endif
    };

    template<typename T>
    Ndarray<T>::NdarrayIterator::NdarrayIterator(Ndarray<T> &ndarray, size_t axis, const vector<size_t> &start_index) {
        ndarray_ = ndarray;
        axis_ = axis;
        index_ = start_index;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator++() {
        index_[axis_]++;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator--() {
        index_[axis_]++;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator+=(int n) {
        index_[axis_]+=n;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator-=(int n) {
        index_[axis_]-=n;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::NdarrayIterator::operator+(int n) const {
        auto copy = *this;
        copy.index_[axis_]+=n;
        return copy;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::NdarrayIterator::operator-(int n) const {
        auto copy = *this;
        copy.index_[axis_]-=n;
        return copy;
    }

    template<typename T>
    T &Ndarray<T>::NdarrayIterator::operator[](int index) {
        auto copy = index_;
        copy[axis_]+=index;
        return B(copy);
    }

    template<typename T>
    const T &Ndarray<T>::NdarrayIterator::operator[](int index) const {
        auto copy = index_;
        copy[axis_]+=index;
        return B(copy);
    }

    template<typename T>
    T &Ndarray<T>::NdarrayIterator::operator*() {
        return ndarray_(index_);
    }

    template<typename T>
    const T &Ndarray<T>::NdarrayIterator::operator*() const {
        return ndarray_(index_);;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::NdarrayIterator::operator-(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] - iter.index_[axis_];
    }

    template<typename T>
    void Ndarray<T>::NdarrayIterator::is_the_same(const NdarrayIterator& iter) const{
        if(axis_ != iter.axis_){
            throw Ndarray<T>::NdarrayExeption("Cann't compare iterators with axis " + std::to_string(axis_) +
                                              " with iterator with axis " + std::to_string(iter.axis_));
        }
        for(int i = 0; i < index_.size(); i++){
            if(i == axis_){
                continue;
            }
            if(index_[i] != iter.index_[i]){
                std::string str(index_.begin(), index_.end());
                std::string str1(index_.begin(), index_.end());
                throw Ndarray<T>::NdarrayExeption("Cann't compare iterators with index " + str +
                                                  " with iterator with index " + str1);
            }
        }
    }

    template<typename T>
    bool Ndarray<T>::NdarrayIterator::operator<(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] < iter.index_[axis_] ? true : false;
    }

    template<typename T>
    bool Ndarray<T>::NdarrayIterator::operator<=(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] < iter.index_[axis_] ? true : false;
    }

    template<typename T>
    bool Ndarray<T>::NdarrayIterator::operator>(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] > iter.index_[axis_] ? true : false;
    }

    template<typename T>
    bool Ndarray<T>::NdarrayIterator::operator>=(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] >= iter.index_[axis_] ? true : false;
    }

    template<typename T>
    Ndarray<T>::Ndarray() : shape_({0}), buffer(nullptr), size_(0), bases_({0}){

    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t> &shape) : shape_(shape), bases_(shape_.size()){
        init_buffer();
    }

    template<typename T>
    Ndarray<T>::~Ndarray() {
        delete[] buffer;
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
    T &Ndarray<T>::operator()(const std::vector<size_t>& index) {
        is_in_range(index);
        size_t index_ = 0;
        for(int i = 0; i < shape_.size(); i++){
            index_ += index[i] * bases_[i];
        }
        return buffer[index_];
    }

    template<typename T>
    const T &Ndarray<T>::operator()(const std::vector<size_t> &index) const {
        is_in_range(index);
        size_t index_ = 0;
        for(int i = 0; i < shape_.size(); i++){
            index_ += index[i] * bases_[i];
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
        this->is_in_range(index);
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

    template<typename T>
    Ndarray<T> Ndarray<T>::max(size_t axis) const {
        if(axis > shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin()+axis);
        Ndarray<T > new_arr(shape_t);
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
            new_arr.buffer[i] = (*this)(max_index);
        }
        return new_arr;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::min(size_t axis) const {
        if(axis > shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin()+axis);
        Ndarray<T > new_arr(shape_t);
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
            new_arr.buffer[i] = (*this)(min_index);
        }
        return new_arr;
    }

    template<typename T>
    void Ndarray<T>::reshape(const vector<int > &shape) {
        int size_temp = 1;
        int koll_unnkown_dim = 0;
        for(auto i = shape.begin(); i < shape.end(); i++){
            if(((*i) > int(size_))||((*i) < -1)){
                throw Ndarray<T>::NdarrayExeption("Cannot reshape array of size " + std::to_string(size_) +
                " into shape " + std::to_string(*i));
            }
            if(*i == -1) {
                if (koll_unnkown_dim == 0) {
                    koll_unnkown_dim++;
                } else {
                    throw Ndarray<T>::NdarrayExeption("Can only specify one unknown dimension");
                }
            }else{
                size_temp *= *i;
                if(size_temp > size_){
                    std::string str(shape.begin(), shape.end());
                    throw Ndarray<T>::NdarrayExeption("Cannot reshape array of size " + std::to_string(size_) + " into shape (" + str+ ")");
                }
            }
        }
        if(size_ % size_temp){
            std::string str(shape.begin(), shape.end());
            throw Ndarray<T>::NdarrayExeption("Cannot reshape array of size " + std::to_string(size_) +  " into shape (" + str+ ")");
        }

        size_t t = 1;
        shape_.resize(shape.size());
        for(int i = shape.size()-1; i >= 0; i--){
            if(shape[i] == -1){
                shape_[i] = size_ / size_temp;
            }else{
                shape_[i] = shape[i];
            }
            t *= shape_[i];
            bases_[i] = t/shape_[i];
        }
        size_ = t;
    }

    template<typename T>
    void Ndarray<T>::sort(bool order){
        if(order) {
            std::sort(buffer, buffer + size_, std::less<T>());
        }else{
            std::sort(buffer, buffer + size_, std::greater<T>());
        }
    }

    template<typename T>
    T &Ndarray<T>::operator[](int index) {
        if((index < 0)||(index > size_)){
            throw Ndarray<T>::NdarrayExeption("Wrong index! array size " + std::to_string(size_) + " and index is " + std::to_string(index));
        }
        return buffer[index];
    }

    template<typename T>
    const T &Ndarray<T>::operator[](int index) const {
        if((index < 0)||(index > size_)){
            throw Ndarray<T>::NdarrayExeption("Wrong index! array size " + std::to_string(size_) + " and index is " + std::to_string(index));
        }
        return buffer[index];
    }

    template<typename T>
    size_t Ndarray<T>::get_1d_index(vector<size_t> index) const {
        is_in_range(index);
        size_t index_res = 0;
        for(int i = 0; i <  index.size(); i++){
            index_res += index[i] * bases_[i];
        }
        return index_res;
    }

    template<typename T>
    void Ndarray<T>::is_in_range(int index) const {
        if((index < 0)||(index >= size_)){
            throw Ndarray<T>::NdarrayExeption("Index " + std::to_string(index) + " is out of bounds for array size " +
            std::to_string(size_));
        }
    }

    template<typename T>
    void Ndarray<T>::is_in_range(vector<size_t> index) const {
        if(index.size() != shape_.size()){
            throw Ndarray<T>::NdarrayExeption("Index size " + std::to_string(index.size()) + " is not match shape size "+
            std::to_string(shape_.size()));
        }
        for(int i = 0; i < shape_.size(); i++){
            if(index[i] >= shape_[i]){
                throw Ndarray<T>::NdarrayExeption("Index " + std::to_string(index.size()) + " is out of bount for axis "+
                std::to_string(i) + " with size " + std::to_string(shape_[i]));
            }
        }
    }
}

#endif //NEURALNETWORKS_NDARRAY_H
