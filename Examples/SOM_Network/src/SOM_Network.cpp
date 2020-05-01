//: Нейросеть кластеризующая цвета
#include "SOM.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Макрос режима работы программы (с обучением или без)
#define Teach

using namespace std;
using namespace NN;
using namespace cv;

int randInt(int a, int b){
    return int((double(random()) / RAND_MAX) * (b-a) + a);
}

int main() {
    // задаём высоту и ширину окна
    int height = 400;
    int width = 400;

    // создаем картинку
    Mat hw (height, width,CV_8UC3);

    // создаём окошко
    namedWindow("Hello World", 0);
    // показываем картинку в созданном окне
    imshow("Hello World", hw);
    // ждём нажатия клавиши
    waitKey(0);

    // освобождаем ресурсы
//    releaseImage(&hw);
    destroyWindow("Hello World");
    return 0;
    return 0;
}
