#include <opencv2/ts.hpp>
#include "Matrix.h"
#include "NeyronCnn.h"
#include <fstream>

TEST(NeyronCnn_, Padding){
    NeyronCnn<double> B;
    double **f = new double*[1];
    f[0] = new double[1];
    f[0][0] = 5;

    Matrix<double> FF(f,1, 1);
    B.Padding(FF);

    EXPECT_EQ(FF.getN(), 3);
    EXPECT_EQ(FF.getM(), 3);
    EXPECT_TRUE(((FF[0][0] == 0) || (FF[0][1] == 0) || (FF[0][2] == 0) || (FF[1][0] == 0) || (FF[1][2] == 0) || (FF[2][0] == 0) || (FF[2][1] == 0) || (FF[2][2] == 0) || (FF[1][1] != 0)));

    delete[](f[0]);
    delete[](f);
}

TEST(NeyronCnn_, Svertka){
    NeyronCnn<double> B;
    Filter<double> y(2, 2);
    double **f = new double*[1];
    f[0] = new double[1];
    f[0][0] = 5;
    Matrix<double> FF(f,1, 1);
    B.Padding(FF);
    FF = B.Svertka(y,FF);

    EXPECT_EQ(FF.getN(), 2);
    EXPECT_EQ(FF.getM(), 2);
    EXPECT_EQ(FF[0][0], 0);
    EXPECT_EQ(FF[0][1], 0);
    EXPECT_EQ(FF[1][0], 0);
    EXPECT_EQ(FF[1][1], 0);

    delete[](f[0]);
    delete[](f);
}

TEST(NeyronCnn_, Svertka_with_step){
    NeyronCnn<int> A;
    Matrix<int> T(1, 1);

    A.Padding(T);
    A.GetStep() = 2;
    EXPECT_EQ(A.GetStep(), 2);
    Filter<int> yy(2, 2);
    T = A.Svertka(yy,T);

    EXPECT_EQ(T.getN(), 1);
    EXPECT_EQ(T.getM(), 1);
    EXPECT_EQ(T[0][0], 0);
}

TEST(NeyronCnn_, Svertka_wrong_params){
    NeyronCnn<int> A;
    A.GetStep() = 5;
    Matrix<int> o(2, 2);
    Filter<int> y(2, 2);
    EXPECT_ANY_THROW(A.Svertka(y, o));
}

TEST(NeyronCnn_, MaxPooling){
    NeyronCnn<double> B;

    Matrix<double> FFF(4, 4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            FFF[i][j] = std::max(i, j);
        }
    }
    FFF = B.Pooling(FFF, 2,2);
    EXPECT_EQ(FFF.getN(), 2);
    EXPECT_EQ(FFF.getM(), 2);

    EXPECT_TRUE((FFF[0][0] == 1) || (FFF[0][1] == 3) || (FFF[1][0] == 3) || (FFF[1][1] == 3));
}

TEST(NeyronCnn_, Visualisation_of_work){
    NeyronCnn<int> A;
    std::ofstream file;

    file.open("NeyronTest.txt");

    Matrix<int> M(6, 6);
    int o = 0;
    A.GetStep() = 1;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            M[i][j] = o++;
        }
    }
    A.Padding(M);
    file << M;
    file << std::endl;

    Filter<int> a(2, 2);
    a[0][0] = 1;
    a[0][1] = 1;
    a[1][0] = 1;
    a[1][1] = 1;

    A.Svertka(a, M);
    file << M;
    file << std::endl;
    A.Pooling(M, 2, 2);
    file << M;
    file << std::endl;
    file.close();

    file.close();
}