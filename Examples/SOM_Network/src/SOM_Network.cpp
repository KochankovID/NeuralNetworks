//: Нейросеть кластеризующая цвета
#include "SOM.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace NN;
using namespace cv;

int randInt(int a, int b){
    return (double(rand()) / RAND_MAX) * (b-a) + a;
}

int main() {
    // задаём высоту и ширину окна
    const int height_w = 600;
    const int width_w = 600;
    // задаем размеры карты кохонена
    const int height = 120;
    const int width = 120;
    // коэффицент масштабирования
    const int k = 5;
    // количество итераций обучения
    const int num_iterations = 250;

    // создаем картинку
    Mat hw (height_w, width_w,CV_8UC3);

    // заполняем случайными значениями
    for(size_t x = 0; x < width_w; x+=k){
        for(size_t y = 0; y < height_w; y+=k){
            Vec3b pixel(randInt(0, 255), randInt(0, 255), randInt(0, 255));
            for(size_t i = 0; i < k; i++){
                for(size_t j = 0; j < k; j++){
                    hw.at<Vec3b>(x+i,y+j) = pixel;
                }
            }
        }
    }
    // создаём окошко
    namedWindow("SOM", 0);
    // показываем картинку в созданном окне
    imshow("SOM", hw);

    // ждём нажатия клавиши
    waitKey(1);

    // создаем карту кохонена
    SOM som(width, height,3, 0.1, 63);
    // инициализация весов
    for(size_t x = 0; x < width_w; x+=k){
        for(size_t y = 0; y < height_w; y+=k){
            auto pixel = hw.at<Vec3b>(x,y);
            som.weights()(x/k,y/k,0) = pixel[0]/255.;
            som.weights()(x/k,y/k,1) = pixel[1]/255.;
            som.weights()(x/k,y/k,2) = pixel[2]/255.;
        }
    }

    // создание набора цветов
    Ndarray<double > data({5,3});
    // синий
    data(0,0) = 0;
    data(0,1) = 0;
    data(0,2) = 1;

    // зеленый
    data(1,0) = 0;
    data(1,1) = 1;
    data(1,2) = 0;

    // красный
    data(2,0) = 1;
    data(2,1) = 0;
    data(2,2) = 0;

    // желтый
    data(3,0) = 0;
    data(3,1) = 1;
    data(3,2) = 1;

    // голубой
    data(4,0) = 1;
    data(4,1) = 1;
    data(4,2) = 0;

    // Обучение сети
    som.train_random(data, num_iterations);
    // производим обучение сети на одной итерации и выводим результат
    for(size_t i = 0; i < num_iterations; i++) {
        for (size_t x = 0; x < width_w; x+=k) {
            for (size_t y = 0; y < height_w; y+=k) {
                Vec3b pixel(som.history()[i]( x/k, y/k, 0)*255,
                        som.history()[i]( x/k, y/k, 1)*255,
                        som.history()[i](x/k, y/k, 2)*255);
                for(size_t i = 0; i < k; i++){
                    for(size_t j = 0; j < k; j++){
                        hw.at<Vec3b>(x+i,y+j) = pixel;
                    }
                }
            }
        }
        imshow("SOM", hw);
        waitKey(10);
    }

    // ждём нажатия клавиши
    waitKey(0);

    // освобождаем ресурсы
    destroyWindow("SOM");
    return 0;
}
