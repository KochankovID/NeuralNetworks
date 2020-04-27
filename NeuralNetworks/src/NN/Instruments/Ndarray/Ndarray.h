#ifndef NEURALNETWORKS_NDARRAY_H
#define NEURALNETWORKS_NDARRAY_H

#include <algorithm>
#include <functional>
#include <cstdarg>
#include <numeric>

using std::vector;
namespace NN {
    template<typename T>
    class Ndarray;

    template<typename T>
    Ndarray<T> matrix_multiplication(Ndarray<T> &left, Ndarray<T> &right);

    template<typename T>
    class Ndarray {
    public:
        // Конструкторы ---------------------------------
        Ndarray();  // По умолчанию
        explicit Ndarray(const vector<size_t> &shape);  // Инициализатор (создает н-мерный массив формы shape)
        Ndarray(const vector<size_t> &shape,
                const vector<T> &array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const vector<size_t> &shape,
                const T *array);  // Инициализатор (создает н-мерный массив формы shape) и инициализирует значениями из array
        Ndarray(const Ndarray &copy);  // Копирования
        Ndarray(Ndarray &&copy);  // Мув коструктор

        // Методы класса --------------------------------
        vector<size_t> shape() const { return shape_; }; // Возвращает форму массива
        Ndarray<T> flatten();  // Возвращает копию массива в 1d измерении
        void reshape(const vector<int> &shape);  // Меняет "форму массива" без изменения его элементов

        size_t argmax() const;  // Возвращает индес наибольшего элемента в массиве
        size_t argmin() const;  // Возвращает индес наибольшего элемента в массиве
        Ndarray<size_t>
        argmax(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси
        Ndarray<size_t>
        argmin(size_t axis) const;  // Возвращает массив индесов наибольших значений взятых по указанной оси

        T max();  // Возвращает наибольший элемент в массиве
        T min();  // Возвращает наименьший элемент в массиве
        Ndarray<T> max(size_t axis) const;  // Возвращает массив наибольших значений взятых по указанной оси
        Ndarray<T> min(size_t axis) const;  // Возвращает массив наибольших значений взятых по указанной оси

        vector<size_t> get_nd_index(size_t index) const;  // Преобразует 1D индекс в ND
        size_t get_1d_index(vector<size_t> index) const;  // Преобразует 1D индекс в ND
        static vector<size_t> get_nd_index(size_t index, const vector<size_t> &shape);  // Преобразует 1D индекс в ND
        static size_t get_1d_index(vector<size_t> index, const vector<size_t> &shape);  // Преобразует 1D индекс в ND

        void fill(const T &value);  // Заполняет массив указанным значением

        void sort(bool order);  // Сортирует массив в 1d по возрастанию (true) по убыванию (false)
        void
        sort(size_t axis, bool order); // Сортирует массив вдоль выбранной оси по возрастанию (true) по убыванию (false)

        T *begin();  // Возвращает указатель на начало массива
        const T *begin() const;  // Возвращает указатель на начало массива
        T *end();  // Возвращает указатель на элемет следущий после массива
        const T *end() const;  // Возвращает указатель на элемет следущий после массива

        class NdarrayIterator;  // Объявление класса итератора
        NdarrayIterator iter(size_t axis,
                             const vector<size_t> &start_index);  // Возвращает итератор вдоль оси axis с указанного индекса
        NdarrayIterator iter_begin(size_t axis,
                                   const vector<size_t> &index);  // Возвращает итератор вдоль оcи axis с указанного индекса с начала оси
        NdarrayIterator iter_end(size_t axis,
                                 const vector<size_t> &index);  // Возвращает итератор вдоль оcи axis с указанного индекса с конца оси

        T mean(); // Возвращает среднее значение элементов массива
        Ndarray<T> mean(size_t axis); // Возвращает среднее значение элементов массива вдоль оси axis


        friend Ndarray<T> matrix_multiplication<>(Ndarray<T> &left, Ndarray<T> &right);

        Ndarray<T> matmul(Ndarray &array);

        // Перегрузки операторов ------------------------
        Ndarray &operator=(const Ndarray &copy);

        T &operator()(const std::vector<size_t> &index);

        const T &operator()(const std::vector<size_t> &index) const;

//        T &operator()(int index, ...);

        T &operator[](int index);

        const T &operator[](int index) const;

        Ndarray &operator+=(const T &value);

        Ndarray &operator-=(const T &value);

        Ndarray operator+(const T &value) const;

        Ndarray operator-(const T &value) const;

        Ndarray &operator*=(const T &value);

        Ndarray &operator/=(const T &value);

        Ndarray operator*(const T &value) const;

        Ndarray operator/(const T &value) const;

        Ndarray &operator+=(const Ndarray &value);

        Ndarray &operator-=(const Ndarray &value);

        Ndarray operator+(const Ndarray &value) const;

        Ndarray operator-(const Ndarray &value) const;

        Ndarray &operator*=(const Ndarray &value);

        Ndarray &operator/=(const Ndarray &value);

        Ndarray operator*(const Ndarray &value) const;

        Ndarray operator/(const Ndarray &value) const;

        Ndarray<bool> operator>(const Ndarray &value) const;

        Ndarray<bool> operator>=(const Ndarray &value) const;

        bool operator==(const Ndarray &value) const;

        bool operator!=(const Ndarray &value) const;

        Ndarray<bool> operator<=(const Ndarray &value) const;

        Ndarray<bool> operator<(const Ndarray &value) const;

        // Итератор -------------------------------------
        friend class NdarrayIterator;

        class NdarrayIterator : public std::iterator<std::random_access_iterator_tag, T, int> {
        public:
            // Конструкторы -----------------------------
            NdarrayIterator(Ndarray<T> &ndarray, size_t axis, const vector<size_t> &start_index);

            NdarrayIterator(const NdarrayIterator &copy);

            // Перегрузки операторов --------------------
            NdarrayIterator &operator=(const NdarrayIterator &iter);

            NdarrayIterator &operator++();

            NdarrayIterator &operator--();

            NdarrayIterator &operator+=(int n);

            NdarrayIterator &operator-=(int n);

            NdarrayIterator operator+(int n) const;

            NdarrayIterator operator-(int n) const;

            int operator-(const NdarrayIterator &iter) const;

            T &operator*();

            const T &operator*() const;

            T &operator[](int index);

            const T &operator[](int index) const;

            bool operator<(const NdarrayIterator &iter) const;

            bool operator<=(const NdarrayIterator &iter) const;

            bool operator==(const NdarrayIterator &iter) const;

            bool operator!=(const NdarrayIterator &iter) const;

            bool operator>(const NdarrayIterator &iter) const;

            bool operator>=(const NdarrayIterator &iter) const;

            friend typename Ndarray<T>::NdarrayIterator
            operator+(int n, const typename Ndarray<T>::NdarrayIterator &iter) {
                auto copy = iter;
                copy.index_[iter.axis_] += n;
                return copy;
            }

            friend typename Ndarray<T>::NdarrayIterator
            operator-(int n, const typename Ndarray<T>::NdarrayIterator &iter) {
                auto copy = iter;
                copy.index_[iter.axis_] -= n;
                return copy;
            }


        private:
            Ndarray<T> &ndarray_;
            vector<size_t> index_;
            size_t axis_;

            void is_the_same(const NdarrayIterator &iter) const;
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
        vector<size_t> shape_;
        vector<size_t> bases_;
        T *buffer;
        size_t size_;

        // Скрытые матоды класса ------------------------
        void init_buffer();

        void is_in_range(int index) const;

        void is_in_range(vector<size_t> index) const;

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
    Ndarray<T>::NdarrayIterator::NdarrayIterator(Ndarray<T> &ndarray, size_t axis, const vector<size_t> &start_index) :
            ndarray_(ndarray) {
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
        index_[axis_]--;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator+=(int n) {
        index_[axis_] += n;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &Ndarray<T>::NdarrayIterator::operator-=(int n) {
        index_[axis_] -= n;
        return *this;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::NdarrayIterator::operator+(int n) const {
        auto copy = *this;
        copy.index_[axis_] += n;
        return copy;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::NdarrayIterator::operator-(int n) const {
        auto copy = *this;
        copy.index_[axis_] -= n;
        return copy;
    }

    template<typename T>
    T &Ndarray<T>::NdarrayIterator::operator[](int index) {
        auto copy = index_;
        copy[axis_] += index;
        return ndarray_(copy);
    }

    template<typename T>
    const T &Ndarray<T>::NdarrayIterator::operator[](int index) const {
        auto copy = index_;
        copy[axis_] += index;
        return ndarray_(copy);
    }

    template<typename T>
    T &Ndarray<T>::NdarrayIterator::operator*() {
        return ndarray_(index_);
    }

    template<typename T>
    const T &Ndarray<T>::NdarrayIterator::operator*() const {
        return ndarray_(index_);
    }

    template<typename T>
    int Ndarray<T>::NdarrayIterator::operator-(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return int(index_[axis_]) - int(iter.index_[axis_]);
    }

    template<typename T>
    void Ndarray<T>::NdarrayIterator::is_the_same(const NdarrayIterator &iter) const {
        if (axis_ != iter.axis_) {
            throw Ndarray<T>::NdarrayExeption("Cann't compare iterators with axis " + std::to_string(axis_) +
                                              " with iterator with axis " + std::to_string(iter.axis_));
        }
        for (int i = 0; i < index_.size(); i++) {
            if (i == axis_) {
                continue;
            }
            if (index_[i] != iter.index_[i]) {
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
        return index_[axis_] <= iter.index_[axis_] ? true : false;
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
    bool Ndarray<T>::NdarrayIterator::operator==(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] == iter.index_[axis_] || false;
    }

    template<typename T>
    bool Ndarray<T>::NdarrayIterator::operator!=(const Ndarray::NdarrayIterator &iter) const {
        is_the_same(iter);
        return index_[axis_] != iter.index_[axis_] || false;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator &
    Ndarray<T>::NdarrayIterator::operator=(const Ndarray<T>::NdarrayIterator &copy) {
        ndarray_ = copy.ndarray_;
        axis_ = copy.axis_;
        index_ = copy.index_;
        return *this;
    }

    template<typename T>
    Ndarray<T>::NdarrayIterator::NdarrayIterator(const NdarrayIterator &copy) : ndarray_(copy.ndarray_) {
        axis_ = copy.axis_;
        index_ = copy.index_;
    }

    template<typename T>
    Ndarray<T>::Ndarray() : shape_({0}), buffer(nullptr), size_(0), bases_({0}) {

    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t> &shape) : shape_(shape), bases_(shape_.size()) {
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
        for (size_t i = 0; i < size_; i++) {
            buffer[i] = copy.buffer[i];
        }
    }

    template<typename T>
    void Ndarray<T>::init_buffer() {
        if ((shape_.size() == 1) && (shape_[0] == 0)) {
            buffer = nullptr;
            size_ = 0;
            bases_[0] = 0;
        } else {
            size_t t = 1;
            for (int i = shape_.size() - 1; i >= 0; i--) {
                t *= shape_[i];
                bases_[i] = t / shape_[i];
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
    T &Ndarray<T>::operator()(const std::vector<size_t> &index) {
        is_in_range(index);
        size_t index_ = 0;
        for (int i = 0; i < shape_.size(); i++) {
            index_ += index[i] * bases_[i];
        }
        return buffer[index_];
    }

    template<typename T>
    const T &Ndarray<T>::operator()(const std::vector<size_t> &index) const {
        is_in_range(index);
        size_t index_ = 0;
        for (int i = 0; i < shape_.size(); i++) {
            index_ += index[i] * bases_[i];
        }
        return buffer[index_];
    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t> &shape, const vector<T> &array) : shape_(shape), bases_(shape_.size()) {
        init_buffer();
        if (array.size() != size_) {
            throw Ndarray<T>::NdarrayExeption("Wrong size of vector!");
        }
        for (int i = 0; i < size_; i++) {
            buffer[i] = array[i];
        }
    }

    template<typename T>
    Ndarray<T>::Ndarray(const vector<size_t> &shape, const T *array) : shape_(shape), bases_(shape_.size()) {
        init_buffer();
        for (int i = 0; i < size_; i++) {
            buffer[i] = array[i];
        }
    }

    template<typename T>
    size_t Ndarray<T>::argmax() const {
        size_t max_index = 0;
        for (int i = 1; i < size_; i++) {
            if (buffer[i] > buffer[max_index]) {
                max_index = i;
            }
        }
        return max_index;
    }

    template<typename T>
    Ndarray<size_t> Ndarray<T>::argmax(size_t axis) const {
        if (axis > shape_.size()) {
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin() + axis);
        Ndarray<size_t> new_arr(shape_t);
        vector<size_t> index(new_arr.shape_.size());

        for (int i = 0; i < new_arr.size_; i++) {
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto max_index = index;
            for (int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if ((*this)(index) > (*this)(max_index)) {
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
        vector<size_t> index_;
        size_t tmp;
        for (size_t i = 0; i < bases_.size(); i++) {
            tmp = index / bases_[i];
            index_.push_back(tmp);
            index -= tmp * bases_[i];
        }
        return index_;
    }

    template<typename T>
    size_t Ndarray<T>::argmin() const {
        size_t min_index = 0;
        for (int i = 1; i < size_; i++) {
            if (buffer[i] < buffer[min_index]) {
                min_index = i;
            }
        }
        return min_index;
    }

    template<typename T>
    Ndarray<size_t> Ndarray<T>::argmin(size_t axis) const {
        if (axis > shape_.size()) {
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin() + axis);
        Ndarray<size_t> new_arr(shape_t);
        vector<size_t> index(new_arr.shape_.size());

        for (int i = 0; i < new_arr.size_; i++) {
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto min_index = index;
            for (int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if ((*this)(index) < (*this)(min_index)) {
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
        for (int i = 1; i < size_; i++) {
            if (buffer[i] > buffer[max_index]) {
                max_index = i;
            }
        }
        return buffer[max_index];
    }

    template<typename T>
    T Ndarray<T>::min() {
        size_t min_index = 0;
        for (int i = 1; i < size_; i++) {
            if (buffer[i] < buffer[min_index]) {
                min_index = i;
            }
        }
        return buffer[min_index];
    }

    template<typename T>
    void Ndarray<T>::fill(const T &value) {
        std::fill(buffer, buffer + size_, value);
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::flatten() {
        return Ndarray<T>({size_}, this->buffer);
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::max(size_t axis) const {
        if (axis > shape_.size()) {
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin() + axis);
        Ndarray<T> new_arr(shape_t);
        vector<size_t> index(new_arr.shape_.size());

        for (int i = 0; i < new_arr.size_; i++) {
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto max_index = index;
            for (int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if ((*this)(index) > (*this)(max_index)) {
                    max_index = index;
                }
            }
            new_arr.buffer[i] = (*this)(max_index);
        }
        return new_arr;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::min(size_t axis) const {
        if (axis > shape_.size()) {
            throw Ndarray<T>::NdarrayExeption("Wrong axis!");
        }
        auto shape_t = this->shape_;
        shape_t.erase(shape_t.begin() + axis);
        Ndarray<T> new_arr(shape_t);
        vector<size_t> index(new_arr.shape_.size());

        for (int i = 0; i < new_arr.size_; i++) {
            index = new_arr.get_nd_index(i);
            index.insert(index.begin() + axis, 0);
            auto min_index = index;
            for (int j = 1; j < shape_[axis]; j++) {
                index[axis] = j;
                if ((*this)(index) < (*this)(min_index)) {
                    min_index = index;
                }
            }
            new_arr.buffer[i] = (*this)(min_index);
        }
        return new_arr;
    }

    template<typename T>
    void Ndarray<T>::reshape(const vector<int> &shape) {
        int size_temp = 1;
        int koll_unnkown_dim = 0;
        for (auto i = shape.begin(); i < shape.end(); i++) {
            if (((*i) > int(size_)) || ((*i) < -1)) {
                throw Ndarray<T>::NdarrayExeption("Cannot reshape array of size " + std::to_string(size_) +
                                                  " into shape " + std::to_string(*i));
            }
            if (*i == -1) {
                if (koll_unnkown_dim == 0) {
                    koll_unnkown_dim++;
                } else {
                    throw Ndarray<T>::NdarrayExeption("Can only specify one unknown dimension");
                }
            } else {
                size_temp *= *i;
                if (size_temp > size_) {
                    std::string str(shape.begin(), shape.end());
                    throw Ndarray<T>::NdarrayExeption(
                            "Cannot reshape array of size " + std::to_string(size_) + " into shape (" + str + ")");
                }
            }
        }
        if (size_ % size_temp) {
            std::string str(shape.begin(), shape.end());
            throw Ndarray<T>::NdarrayExeption(
                    "Cannot reshape array of size " + std::to_string(size_) + " into shape (" + str + ")");
        }

        size_t t = 1;
        shape_.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; i--) {
            if (shape[i] == -1) {
                shape_[i] = size_ / size_temp;
            } else {
                shape_[i] = shape[i];
            }
            t *= shape_[i];
            bases_[i] = t / shape_[i];
        }
        size_ = t;
    }

    template<typename T>
    void Ndarray<T>::sort(bool order) {
        if (order) {
            std::sort(buffer, buffer + size_, std::less<T>());
        } else {
            std::sort(buffer, buffer + size_, std::greater<T>());
        }
    }

    template<typename T>
    void Ndarray<T>::sort(size_t axis, bool order) {
        vector<size_t> shape_temp = shape_;
        shape_temp.erase(shape_temp.begin() + axis);
        for (int i = 0; i < size_ / shape_[axis]; i++) {
            auto index = get_nd_index(i, shape_temp);
            index.insert(index.begin() + axis, 0);
            if (order) {
                std::sort(iter_begin(axis, index), iter_end(axis, index), std::less<T>());
            } else {
                std::sort(iter_begin(axis, index), iter_end(axis, index), std::greater<T>());
            }
        }
    }

    template<typename T>
    T &Ndarray<T>::operator[](int index) {
        if ((index < 0) || (index > size_)) {
            throw Ndarray<T>::NdarrayExeption(
                    "Wrong index! array size " + std::to_string(size_) + " and index is " + std::to_string(index));
        }
        return buffer[index];
    }

    template<typename T>
    const T &Ndarray<T>::operator[](int index) const {
        if ((index < 0) || (index > size_)) {
            throw Ndarray<T>::NdarrayExeption(
                    "Wrong index! array size " + std::to_string(size_) + " and index is " + std::to_string(index));
        }
        return buffer[index];
    }

    template<typename T>
    size_t Ndarray<T>::get_1d_index(vector<size_t> index) const {
        is_in_range(index);
        size_t index_res = 0;
        for (int i = 0; i < index.size(); i++) {
            index_res += index[i] * bases_[i];
        }
        return index_res;
    }

    template<typename T>
    void Ndarray<T>::is_in_range(int index) const {
        if ((index < 0) || (index >= size_)) {
            throw Ndarray<T>::NdarrayExeption("Index " + std::to_string(index) + " is out of bounds for array size " +
                                              std::to_string(size_));
        }
    }

    template<typename T>
    void Ndarray<T>::is_in_range(vector<size_t> index) const {
        if (index.size() != shape_.size()) {
            throw Ndarray<T>::NdarrayExeption(
                    "Index size " + std::to_string(index.size()) + " is not match shape size " +
                    std::to_string(shape_.size()));
        }
        for (int i = 0; i < shape_.size(); i++) {
            if (index[i] >= shape_[i]) {
                throw Ndarray<T>::NdarrayExeption(
                        "Index " + std::to_string(index.size()) + " is out of bount for axis " +
                        std::to_string(i) + " with size " + std::to_string(shape_[i]));
            }
        }
    }

    template<typename T>
    T *Ndarray<T>::begin() {
        return buffer;
    }

    template<typename T>
    const T *Ndarray<T>::begin() const {
        return buffer;
    }

    template<typename T>
    T *Ndarray<T>::end() {
        return buffer + size_;
    }

    template<typename T>
    const T *Ndarray<T>::end() const {
        return buffer + size_;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::iter(size_t axis, const vector<size_t> &start_index) {
        if (axis >= shape_.size()) {
            std::string str(shape_.begin(), shape_.end());
            throw Ndarray<T>::NdarrayExeption("Array with shape " + str + " doesn't have axis " + std::to_string(axis));
        }
        is_in_range(start_index);
        return Ndarray::NdarrayIterator(*this, axis, start_index);
    }

    template<typename T>
    vector<size_t> Ndarray<T>::get_nd_index(size_t index, const vector<size_t> &shape) {
        size_t size_ = 1;
        vector<size_t> bases_(shape.size());
        for (int i = shape.size() - 1; i >= 0; i--) {
            size_ *= shape[i];
            bases_[i] = size_ / shape[i];
        }
        if (index > size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Index " + std::to_string(index) + " bigger than size of shpe  " + std::to_string(size_));
        }
        vector<size_t> index_;
        size_t tmp;
        for (size_t i = 0; i < shape.size(); i++) {
            tmp = index / bases_[i];
            index_.push_back(tmp);
            index -= tmp * bases_[i];
        }
        return index_;
    }

    template<typename T>
    size_t Ndarray<T>::get_1d_index(vector<size_t> index, const vector<size_t> &shape) {
        if (index.size() != shape.size()) {
            throw Ndarray<T>::NdarrayExeption(
                    "Index size " + std::to_string(index.size()) + " is not match shape size " +
                    std::to_string(shape.size()));
        }
        for (int i = 0; i < shape.size(); i++) {
            if (index[i] >= shape[i]) {
                throw Ndarray<T>::NdarrayExeption(
                        "Index " + std::to_string(index.size()) + " is out of bount for axis " +
                        std::to_string(i) + " with size " + std::to_string(shape[i]));
            }
        }
        size_t size_ = 1;
        size_t index_res = 0;
        vector<size_t> bases_(shape.size());
        for (int i = shape.size() - 1; i >= 0; i--) {
            size_ *= shape[i];
            bases_[i] = size_ / shape[i];
        }
        for (int i = 0; i < index.size(); i++) {
            index_res += index[i] * bases_[i];
        }
        return index_res;
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::iter_begin(size_t axis, const vector<size_t> &index) {
        auto temp_index = index;
        temp_index[axis] = 0;
        return Ndarray::NdarrayIterator(*this, axis, temp_index);
    }

    template<typename T>
    typename Ndarray<T>::NdarrayIterator Ndarray<T>::iter_end(size_t axis, const vector<size_t> &index) {
        auto temp_index = index;
        temp_index[axis] += shape_[axis];
        return Ndarray::NdarrayIterator(*this, axis, temp_index);
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator=(const Ndarray<T> &copy) {
        delete[] buffer;
        this->shape_ = copy.shape_;
        this->size_ = copy.size_;
        this->bases_ = copy.bases_;
        init_buffer();
        for (size_t i = 0; i < size_; i++) {
            buffer[i] = copy.buffer[i];
        }
        return *this;
    }

    template<typename T>
    T Ndarray<T>::mean() {
        return double(std::accumulate(buffer, buffer + size_, 0)) / size_;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::mean(size_t axis) {
        vector<size_t> shape_temp = shape_;
        shape_temp.erase(shape_temp.begin() + axis);
        Ndarray<T> new_arr(shape_temp);
        for (int i = 0; i < new_arr.size_; i++) {
            auto index = get_nd_index(i, shape_temp);
            index.insert(index.begin() + axis, 0);
            new_arr[i] = double(std::accumulate(iter_begin(axis, index), iter_end(axis, index), 0)) / shape_[i];
        }
        return new_arr;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator+=(const T &value) {
        for (size_t i = 0; i < size_; i++) {
            buffer[i] += value;
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator-=(const T &value) {
        for (size_t i = 0; i < size_; i++) {
            buffer[i] -= value;
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator+(const T &value) const {
        auto temp = *this;
        temp += value;
        return temp;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator-(const T &value) const {
        auto temp = *this;
        temp -= value;
        return temp;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator*=(const T &value) {
        for (size_t i = 0; i < size_; i++) {
            buffer[i] *= value;
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator/=(const T &value) {
        for (size_t i = 0; i < size_; i++) {
            buffer[i] /= value;
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator*(const T &value) const {
        auto temp = *this;
        temp *= value;
        return temp;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator/(const T &value) const {
        auto temp = *this;
        temp /= value;
        return temp;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator+=(const Ndarray &value) {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        for (size_t i = 0; i < value.size_; i++) {
            buffer[i] += value.buffer[i];
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator-=(const Ndarray &value) {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        for (size_t i = 0; i < value.size_; i++) {
            buffer[i] -= value.buffer[i];
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator+(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        auto temp = *this;
        temp += value;
        return temp;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator-(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        auto temp = *this;
        temp -= value;
        return temp;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator*=(const Ndarray &value) {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        for (size_t i = 0; i < value.size_; i++) {
            buffer[i] *= value.buffer[i];
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> &Ndarray<T>::operator/=(const Ndarray &value) {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        for (size_t i = 0; i < value.size_; i++) {
            buffer[i] /= value.buffer[i];
        }
        return *this;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator*(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        auto temp = *this;
        temp *= value;
        return temp;
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::operator/(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        auto temp = *this;
        temp /= value;
        return temp;
    }

    template<typename T>
    Ndarray<bool> Ndarray<T>::operator>(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        Ndarray<bool> temp({size_});
        for (size_t i = 0; i < value.size_; i++) {
            temp[i] = buffer[i] > value.buffer[i];
        }
        return temp;
    }

    template<typename T>
    Ndarray<bool> Ndarray<T>::operator>=(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        Ndarray<bool> temp({size_});
        for (size_t i = 0; i < value.size_; i++) {
            temp[i] = buffer[i] >= value.buffer[i];
        }
        return temp;
    }

    template<typename T>
    Ndarray<bool> Ndarray<T>::operator<=(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        Ndarray<bool> temp({size_});
        for (size_t i = 0; i < value.size_; i++) {
            temp[i] = buffer[i] <= value.buffer[i];
        }
        return temp;
    }

    template<typename T>
    Ndarray<bool> Ndarray<T>::operator<(const Ndarray &value) const {
        if (size_ != value.size_) {
            throw Ndarray<T>::NdarrayExeption(
                    "Operands could not be broadcast together with sizes " + std::to_string(size_) +
                    " and " + std::to_string(value.size_));
        }
        Ndarray<bool> temp({size_});
        for (size_t i = 0; i < value.size_; i++) {
            temp[i] = buffer[i] < value.buffer[i];
        }
        return temp;
    }

    template<typename T>
    bool Ndarray<T>::operator==(const Ndarray &value) const {
        if (size_ != value.size_) {
            return false;
        }
        for (size_t i = 0; i < value.size_; i++) {
            if (buffer[i] != value.buffer[i]) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    bool Ndarray<T>::operator!=(const Ndarray &value) const {
        return !(*this == value);
    }

    template<typename T>
    Ndarray<T> Ndarray<T>::matmul(Ndarray &array) {
        return matrix_multiplication(*this, array);
    }

    template<typename T>
    Ndarray<T> __mm__(const Ndarray<T> &left, const Ndarray<T> &right) {
        Ndarray<T> result({left.shape_[0], right.shape_[1]});
        size_t M = result.shape_[0], N = result.shape_[1], K = left.shape_[1];
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                result[i * N + j] = 0;
                for (int k = 0; k < K; ++k)
                    result[i * N + j] += left.buffer[i * K + k] * right.buffer[k * N + j];
            }
        }
        return result;
    }

    template<typename T>
    Ndarray<T> matrix_multiplication(Ndarray<T> &left, Ndarray<T> &right) {
        vector<int> sh1(2);
        vector<int> sh2(2);
        switch (left.shape_.size()) {
            case 1:
                sh1[0] = 1;
                sh1[1] = left.shape_[0];
                left.reshape(sh1);
                switch (right.shape_.size()) {
                    case 1: {
                        sh2[0] = right.shape_[0];
                        sh2[1] = 1;
                        right.reshape(sh2);
                        if (left.shape_[1] != right.shape_[0]) {
                            sh1.resize(1);
                            sh1[0] = left.shape_[1];
                            left.reshape(sh1);
                            sh2.resize(1);
                            sh2[0] = right.shape_[0];
                            right.reshape(sh2);
                            throw typename Ndarray<T>::NdarrayExeption(
                                    "Input operand has a mismatch in its core dimension 0, (size " +
                                    std::to_string(left.shape_[0]) + " is different from " +
                                    std::to_string(left.shape_[0]) + ")");
                        }
                        auto res = __mm__(left, right);
                        sh1.resize(1);
                        sh1[0] = left.shape_[1];
                        left.reshape(sh1);
                        sh2.resize(1);
                        sh2[0] = right.shape_[0];
                        right.reshape(sh2);
                        return res;
                    }
                    case 2: {
                        if (left.shape_[1] != right.shape_[0]) {
                            sh1.resize(1);
                            sh1[0] = left.shape_[1];
                            left.reshape(sh1);
                            throw typename Ndarray<T>::NdarrayExeption(
                                    "Input operand has a mismatch in its core dimension 0, (size " +
                                    std::to_string(left.shape_[0]) + " is different from " +
                                    std::to_string(left.shape_[0]) + ")");
                        }
                        auto res = __mm__(left, right);
                        sh1.resize(1);
                        sh1[0] = left.shape_[1];
                        left.reshape(sh1);
                        return res;
                    }
                    default: {
                        sh1.resize(1);
                        sh1[0] = left.shape_[1];
                        left.reshape(sh1);
                        throw typename Ndarray<T>::NdarrayExeption(
                                "Shape of right element is bigger than 2: " + std::to_string(right.shape_.size()));
                    }
                }
                case 2: {
                switch (right.shape_.size()) {
                    case 1: {
                        sh2[0] = right.shape_[0];
                        sh2[1] = 1;
                        right.reshape(sh2);
                        if (left.shape_[1] != right.shape_[0]) {
                            sh2.resize(1);
                            sh2[0] = right.shape_[0];
                            right.reshape(sh2);
                            throw typename Ndarray<T>::NdarrayExeption(
                                    "Input operand has a mismatch in its core dimension 0, (size " +
                                    std::to_string(left.shape_[0]) + " is different from " +
                                    std::to_string(left.shape_[0]) + ")");
                        }
                        auto res = __mm__(left, right);
                        sh2.resize(1);
                        sh2[0] = right.shape_[0];
                        right.reshape(sh2);
                        return res;
                    }
                    case 2: {
                        if (left.shape_[1] != right.shape_[0]) {
                            sh1.resize(1);
                            sh1[0] = left.shape_[1];
                            left.reshape(sh1);
                            sh2.resize(1);
                            sh2[0] = right.shape_[0];
                            right.reshape(sh2);
                            throw typename Ndarray<T>::NdarrayExeption(
                                    "Input operand has a mismatch in its core dimension 0, (size " +
                                    std::to_string(left.shape_[0]) + " is different from " +
                                    std::to_string(left.shape_[0]) + ")");
                        }
                        return __mm__(left, right);
                    }
                    default: {
                        sh1.resize(1);
                        sh1[0] = left.shape_[1];
                        left.reshape(sh1);
                        throw typename Ndarray<T>::NdarrayExeption(
                                "Shape of right element is bigger than 2: " + std::to_string(right.shape_.size()));
                    }
                }
            }
            default: {
                sh1.resize(1);
                sh1[0] = left.shape_[1];
                left.reshape(sh1);
                throw typename Ndarray<T>::NdarrayExeption(
                        "Shape of right element is bigger than 2: " + std::to_string(right.shape_.size()));
            }
        }
    }


//    template<typename T>
//    T &Ndarray<T>::operator()(int index, ...) {
//        va_list arguments;
//        va_start(arguments, index);
//        if((index < 0)||(index >= shape_[0])) {
//            throw Ndarray<T>::NdarrayExeption("Wrong index!");
//        }
//        size_t t_ind = 0;
//        t_ind + index * bases_[0];
//        for(int i = 1; i < shape_.size(); i++){
//            int t_i = va_arg(arguments, int);
//            if((t_i < 0)||(t_i >= shape_[i])) {
//                throw Ndarray<T>::NdarrayExeption("Wrong index!");
//            }
//            t_ind += t_i * bases_[i];
//        }
//        return buffer[t_ind];
//    }
}

#endif //NEURALNETWORKS_NDARRAY_H
