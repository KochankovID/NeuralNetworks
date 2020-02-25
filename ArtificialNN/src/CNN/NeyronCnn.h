#ifndef BASE_CNN
#define BASE_CNN

#include "Base_Cnn.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>

template <typename T>
class NeyronCnn : public Base_Cnn<T>
{
public:
	// Конструкторы ----------------------------------------------------------
	NeyronCnn();
	explicit NeyronCnn(const int &step_);
	NeyronCnn(const NeyronCnn<T> &copy) = delete; // Запрет копирования

	// Методы класса ---------------------------------------------------------
	// Операция свертки над матрицей значений
	Matrix<T> Svertka(const Matrix<T> &F, const Matrix<T> &a);

	// Получение доступа к шагу свертки
	int &GetStep() { return step; }

	// Перегрузка операторов -------------------------------------------------
	NeyronCnn &operator=(const NeyronCnn<T> &copy) = delete; // Запрет копирования

	// Деструктор ------------------------------------------------------------
	~NeyronCnn();

	// Класс исключения ------------------------------------------------------
	class NeyronCnnExeption : public Base_Cnn<T>::Base_CnnExeption
	{
	public:
		NeyronCnnExeption(std::string str) : Base_Cnn<T>::Base_CnnExeption(str){};
		~NeyronCnnExeption(){};
	};

private:
	// Поля класса -----------------------------------------------------------
	int step; // Шаг свертки
};

template <typename T>
NeyronCnn<T>::NeyronCnn() : Base_Cnn<T>(), step(1)
{
}

template <typename T>
NeyronCnn<T>::NeyronCnn(const int &step_) : Base_Cnn<T>(), step(step_)
{
}

template <typename T>
Matrix<T> NeyronCnn<T>::Svertka(const Matrix<T> &F, const Matrix<T> &a)
{
    // Проверка правильности задания шага свертки
	if ((step > a.getN()) || (step > a.getM()) || (step < 1))
	{
		throw NeyronCnnExeption("Задан невозможный шаг свертки!");
	}

	// Создание результирующей матрицы
	Matrix<T> rez((a.getN() - F.getN()) / step + 1, (a.getM() - F.getM()) / step + 1);

    cv::parallel_for_(cv::Range(0, rez.getN()),[&](const cv::Range& range){
        for (int i = range.start; i < range.end; i++)
        {
            for (int j = 0; j < rez.getM(); j++)
            {

                // Переменная в которой хранится текущая сумма свертки
                double sum;

                // Текущая матрица фокуса свертки
                Matrix<T> fokus;

                // Начало поэлементного умножения
                sum = 0;

                // Получение текущей подматрицы
                fokus = a.getPodmatrix(i * step, j * step, F.getN(), F.getM());

                // Вычисление суммы
                for (int ii = 0; ii < F.getN(); ii++)
                {
                    for (int jj = 0; jj < F.getM(); jj++)
                    {
                        sum += fokus[ii][jj] * F[ii][jj];
                    }
                }
                rez[i][j] = sum;
            }
        }
    });

	return rez;
}

//
//template<> inline Matrix<int> NeyronCnn<int>::Svertka(const Matrix<int> &F, const Matrix<int> &a)
//{
//    // Проверка правильности задания шага свертки
//    if ((step > a.getN()) || (step > a.getM()) || (step < 1))
//    {
//        throw NeyronCnnExeption("Задан невозможный шаг свертки!");
//    }
//
//    // Создание результирующей матрицы
//    Matrix<int> rez((a.getN() - F.getN()) / step + 1, (a.getM() - F.getM()) / step + 1);
//
//    cv::parallel_for_(cv::Range(0, rez.getN()),[&](const cv::Range& range){
//        for (int i = range.start; i < range.end; i++)
//        {
//            for (int j = 0; j < rez.getM(); j++)
//            {
//
//                // Переменная в которой хранится текущая сумма свертки
//                double sum;
//
//                // Текущая матрица фокуса свертки
//                Matrix<int> fokus;
//
//                // Начало поэлементного умножения
//                sum = 0;
//
//                // Получение текущей подматрицы
//                fokus = a.getPodmatrix(i * step, j * step, F.getN(), F.getM());
//
//                cv::v_int32x4 summ_register = cv::v_setall_s32(0);
//
//                // Вычисление суммы
//                for (int ii = 0; ii < F.getN(); ii++)
//                {
//
//                    for (int jj = 0; jj < F.getM()/4 ; jj++)
//                    {
//                        summ_register += cv::v_load(&F[ii][4 * jj]) * cv::v_load(&fokus[ii][4 * jj]);
//                    }
////                    if(F.getM() % 4){
////                        for(int jj = F.getM()-1; jj > F.getM()-1-F.getM() % 4; jj--){
////                            sum += fokus[ii][jj] * F[ii][jj];
////                        }
////                    }
//                }
//                sum += cv::v_reduce_sum(summ_register);
//
//                rez[i][j] = sum;
//            }
//        }
//    });
//
//    return rez;
//}
//
//template<> inline Matrix<float> NeyronCnn<float>::Svertka(const Matrix<float> &F, const Matrix<float> &a)
//{
//    // Проверка правильности задания шага свертки
//    if ((step > a.getN()) || (step > a.getM()) || (step < 1))
//    {
//        throw NeyronCnnExeption("Задан невозможный шаг свертки!");
//    }
//
//    // Создание результирующей матрицы
//    Matrix<float> rez((a.getN() - F.getN()) / step + 1, (a.getM() - F.getM()) / step + 1);
//
//    cv::parallel_for_(cv::Range(0, rez.getN()),[&](const cv::Range& range){
//        for (int i = range.start; i < range.end; i++)
//        {
//            for (int j = 0; j < rez.getM(); j++)
//            {
//
//                // Переменная в которой хранится текущая сумма свертки
//                double sum;
//
//                // Текущая матрица фокуса свертки
//                Matrix<float> fokus;
//
//                // Начало поэлементного умножения
//                sum = 0;
//
//                // Получение текущей подматрицы
//                fokus = a.getPodmatrix(i * step, j * step, F.getN(), F.getM());
//
//                cv::v_float32x4 vector_register_foucus;
//                cv::v_float32x4 vector_register_filter;
//                cv::v_float32x4 summ_register = cv::v_setall_f32(0);
//
//                // Вычисление суммы
//                for (int ii = 0; ii < F.getN(); ii++)
//                {
//
//                    for (int jj = 0; jj < F.getM()/4 ; jj++)
//                    {
//                        vector_register_foucus = cv::v_load(&fokus[ii][4 * jj]);
//                        vector_register_filter = cv::v_load(&F[ii][4 * jj]);
//                        vector_register_filter *= vector_register_foucus;
//                        summ_register += vector_register_filter;
//                    }
////                    if(F.getM() % 4){
////                        for(int jj = F.getM()-1; jj > F.getM()-1-F.getM() % 4; jj--){
////                            sum += fokus[ii][jj] * F[ii][jj];
////                        }
////                    }
//                }
//                sum += cv::v_reduce_sum(summ_register);
//
//                rez[i][j] = sum;
//            }
//        }
//    });
//
//    return rez;
//}
//
//template<> inline Matrix<short int> NeyronCnn<short int>::Svertka(const Matrix<short int> &F, const Matrix<short int> &a)
//{
//    // Проверка правильности задания шага свертки
//    if ((step > a.getN()) || (step > a.getM()) || (step < 1))
//    {
//        throw NeyronCnnExeption("Задан невозможный шаг свертки!");
//    }
//
//    // Создание результирующей матрицы
//    Matrix<short int> rez((a.getN() - F.getN()) / step + 1, (a.getM() - F.getM()) / step + 1);
//
//    cv::parallel_for_(cv::Range(0, rez.getN()),[&](const cv::Range& range){
//        for (int i = range.start; i < range.end; i++)
//        {
//            for (int j = 0; j < rez.getM(); j++)
//            {
//
//                // Переменная в которой хранится текущая сумма свертки
//                double sum;
//
//                // Текущая матрица фокуса свертки
//                Matrix<short int> fokus;
//
//                // Начало поэлементного умножения
//                sum = 0;
//
//                // Получение текущей подматрицы
//                fokus = a.getPodmatrix(i * step, j * step, F.getN(), F.getM());
//
//                cv::v_int16 summ_register;
//
//                // Вычисление суммы
//                for (int ii = 0; ii < F.getN(); ii++)
//                {
//
//                    for (int jj = 0; jj < F.getM()/8 ; jj++)
//                    {
//                        summ_register += cv::v_load(&F[ii][8 * jj]) * cv::v_load(&fokus[ii][8 * jj]);
//                    }
////                    if(F.getM() % 8){
////                        for(int jj = F.getM()-1; jj > F.getM()-1-F.getM() % 8; jj--){
////                            sum += fokus[ii][jj] * F[ii][jj];
////                        }
////                    }
//                }
//                sum += cv::v_reduce_sum(summ_register);
//
//                rez[i][j] = sum;
//            }
//        }
//    });
//
//    return rez;
//}

template <typename T>
NeyronCnn<T>::~NeyronCnn()
{
}
#endif
