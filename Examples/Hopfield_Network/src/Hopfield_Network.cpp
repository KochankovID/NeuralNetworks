//: Нейросеть восстанавливающая зашумленную цифру
#include "Hopfield.h"
#include <iostream>
#include "iomanip"
#include "csv.h"

using namespace std;
using namespace NN;

int main() {
    // создание обучающего примера
    Ndarray<int> sample({2,15});

    // считывание образа цифр 4 и 2
    io::CSVReader<16> in("./resources/training_nums.csv");
    int t;
    // считывание 4
    in.read_row(t, sample(0,0),
                sample(0,1), sample(0,2),
                sample(0,3), sample(0,4),
                sample(0,5), sample(0,6),
                sample(0,7), sample(0,8),
                sample(0,9), sample(0,10),
                sample(0,11), sample(0,12),
                sample(0,13), sample(0,14));

    // считывание 2
    in.read_row(t, sample(1,0),
                sample(1,1), sample(1,2),
                sample(1,3), sample(1,4),
                sample(1,5), sample(1,6),
                sample(1,7), sample(1,8),
                sample(1,9), sample(1,10),
                sample(1,11), sample(1,12),
                sample(1,13), sample(1,14));
    for(size_t i = 0; i < 15; i++){
        if(sample(0,i) == 0){
            sample(0,i) = -1;
        }
        if(sample(1,i) == 0){
            sample(1,i) = -1;
        }
    }
    
    // создание сети
    Hopfield hopfield(15);
    
    // обучение сети образу 4
    hopfield.train(sample);

    // создание тестирующих примеров
    Ndarray<int> test_samples({22,15});

    // считывание тестовой выборки
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    for (int i = 0; i < 22; i++) {
        in_test.read_row(t, test_samples(i, 0),
                         test_samples(i, 1), test_samples(i, 2),
                         test_samples(i, 3), test_samples(i, 4),
                         test_samples(i, 5), test_samples(i, 6),
                         test_samples(i, 7), test_samples(i, 8),
                         test_samples(i, 9), test_samples(i, 10),
                         test_samples(i, 11), test_samples(i, 12),
                         test_samples(i, 13), test_samples(i, 14),t);
        for(size_t j = 0; j < 15; j++){
            if(test_samples(i,j) == 0){
                test_samples(i,j) = -1;
            }
        }
    }


    Ndarray<Ndarray<int>> results({22});
    for(size_t i = 0; i < 22; i++){
        auto sample = test_samples.subArray(1,i);
        results[i] = hopfield.fit(sample);
        cout << endl;
        for(size_t x = 0; x < 5; x++){
            cout << setw(2) << sample[x*3+0] << ' ' << setw(2) << sample[x*3+1] << ' ' << setw(2) << sample[x*3+2] << "      "
            << setw(2) << results[i][x*3+0] << ' ' << setw(2) << results[i][x*3+1] << ' ' << setw(2) << results[i][x*3+2] << endl;
        }
    }

    return 0;
}
