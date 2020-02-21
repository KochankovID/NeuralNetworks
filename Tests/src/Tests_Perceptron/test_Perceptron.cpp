#include "opencv2/ts.hpp"
#include <fstream>
#include "Perceptrons.h"

class MyClass : public Func<int,double>
{
public:
    MyClass() : Func<int, double>() {};
    double operator()(const int& x) {
        if (x <= 0) {
            return -1.5;
        }
        else {
            return 1.5;
        }
    }
    ~MyClass() {};
};

TEST(Perceptron_, Summator_matrix){
    ID_Perceptron A;

    Matrix<int> Y(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Y[i][j] = 1;
        }
    }
    Weights<int> W(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            W[i][j] = 1;
        }
    }

    EXPECT_EQ(A.Summator(Y, W), 9);
}

TEST(Perceptron_, Summator_vector){
    ID_Perceptron A;

    std::vector<int> T;
    std::vector<int> TT;
    for (int i = 0; i < 3; i++) {
        T.push_back(1);
        TT.push_back(2);
    }
    EXPECT_EQ(A.Summator(T, TT), 6);
}

TEST(Perceptron_, FunkActiv){
    ID_Perceptron A;

    Matrix<int> Y(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Y[i][j] = 1;
        }
    }
    Weights<int> W(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            W[i][j] = 1;
        }
    }
    std::vector<int> T;
    std::vector<int> TT;
    for (int i = 0; i < 3; i++) {
        T.push_back(1);
        TT.push_back(2);
    }

    MyClass O;

    EXPECT_EQ(A.FunkActiv(A.Summator(Y, W), O), 1.5);
    W[0][0] = -1000;
    EXPECT_EQ(A.FunkActiv(A.Summator(Y, W), O), -1.5);
}

TEST(Perceptron_, Summator_matrix_wrong_argumetns){
    DD_Perceptron B;

    Matrix<double> Y(3, 2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            Y[i][j] = 1;
        }
    }
    Weights<double> W(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            W[i][j] = 1;
        }
    }

    EXPECT_ANY_THROW(B.Summator(Y, W));
}

TEST(Perceptron_, Summator_vector_wrong_argumetns){
    DD_Perceptron B;

    std::vector<double> T;
    std::vector<double> TT;
    for (int i = 0; i < 3; i++) {
        T.push_back(1);
        TT.push_back(2);
    }
    TT.pop_back();

    EXPECT_ANY_THROW(B.Summator(T,TT));
}