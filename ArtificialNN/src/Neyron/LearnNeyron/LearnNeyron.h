#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Neyron.h"
#include "Functors.h"

namespace ANN {

    template<typename T>
    class LearnNeyron {

    public:
        // Конструкторы ----------------------------------------------------------
        LearnNeyron(); // По умолчанию
        LearnNeyron(const double &E_); // Инициализатор
        LearnNeyron(const NeyronPerceptron <T, Y> &copy) = delete; // Запрет копирования

        // Методы класса ---------------------------------------------------------
        // Метод обратного распространения ошибки
        static void BackPropagation(Matrix <Weights<T>> &w, const Weights <T> &y);
        static void BackPropagation(Matrix <Weights<T>> &w, const Matrix <Weights<T>> &y);

        // Метод градиентного спуска
        void GradDes(Func<T>& G, Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x);

        // Функция потерь
        static T loss_function();

        // Функция метрики
        static T metric_function();

        // Метод стягивания весов
        void retract(Matrix<Weights<T>> &weights, const int &decs);

        void retract(Weights<T> &weights, const int &decs);

        // Тасование последовательности
        void shuffle(int *arr, const int &lenth);

        // Метод получения доступа к кофиценту обучения
        double &getE() { return E; };

        // Перегрузка операторов -------------------------------------------------
        LearnNeyron &operator=(const LearnNeyron &copy) = delete; // Запрет копирования

        // Деструктор ------------------------------------------------------------
        ~LearnNeyron();

        // Класс исключения ------------------------------------------------------
        class LearningExeption : public std::runtime_error {
        public:
            LearningExeption(std::string str) : std::runtime_error(str) {};

            ~LearningExeption() {};
        };

    private:
        // Поля класса ----------------------------------
        double E; // Кофицент обучения
    };

    template<typename T>
    inline LearnNeyron<T>::LearnNeyron() : E(1) {
    }

    template<typename T >
    inline LearnNeyron<T>::LearnNeyron(const double &E_) : E(E_) {
    }

    template<typename T>
    inline void LearnNeyron<T, Y>::BackPropagation(Matrix <Weights<T>> &w, const Weights <T> &y) {
        for (int i = 0; i < y.getN(); i++) {
            for (int j = 0; j < y.getM(); j++) {
                w[i][j].GetD() += (y[i][j] * y.GetD());
            }
        }
    }

    template<typename T >
    inline void PerceptronLearning<T, Y>::BackPropagation(Matrix <Weights<T>> &w, const Matrix <Weights<T>> &y) {
        for (int o = 0; o < y.getN(); o++) {
            for (int u = 0; u < y.getM(); u++) {
                for (int i = 0; i < y[o][u].getN(); i++) {
                    for (int j = 0; j < y[o][u].getM(); j++) {
                        w[i][j].GetD() += (y[o][u][i][j] * y.GetD());
                    }
                }
            }
        }
    }

    template<typename T >
    inline void PerceptronLearning<T, Y>::GradDes(Func<T>& G, Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x) {
        if ((w.getN() != in.getN()) || (w.getM() != in.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        G(Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x);
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
