#include "opencv2/ts.hpp"
#include <fstream>
#include "CNNLearns.h"

using namespace ANN;

TEST(CNNLearning_, Constructor){
    D_CNNLeaning A;
    EXPECT_EQ(A.getE(), 1);
    EXPECT_EQ(A.getStep(), 1);
}

TEST(CNNLearning_, GetE){
    D_CNNLeaning A;
    A.getE() = 2;
    EXPECT_EQ(A.getE(), 2);
    A.getE() = 1;
}

TEST(CNNLearning_, GetStep){
    D_CNNLeaning A;
    A.getStep() = 5;
    EXPECT_EQ(A.getStep(), 5);
    A.getStep() = 1;
}

TEST(CNNLearning_, GetStep_wrong_arguments){
    D_CNNLeaning A;
    Matrix<double> u(10, 10);
    Matrix<double> uu(2, 2);
    Filter<double> uuu(2, 2);
    EXPECT_ANY_THROW(A.GradDes(u, uu,uuu));
}

TEST(CNNLearning_, ReversConvolution){
    D_CNNLeaning A;
    Matrix<double> X(3,3);
    for (int i = 0; i < 3; i++) {
        X[i][0] = 1;
        X[i][1] = 2;
        X[i][2] = 3;
    }
    Filter <double> F (2, 2);
    F.Fill(1);
    NeyronCnn<double> g;
    auto Out = g.Svertka(F, X);
    auto Dnext = A.ReversConvolution(Out, F);

    std::ofstream out("CNNLearningTest.txt");
    out << Dnext;
    EXPECT_EQ(Dnext[0][0], 6);
    EXPECT_EQ(Dnext[0][1], 16);
    EXPECT_EQ(Dnext[0][2], 10);
    EXPECT_EQ(Dnext[1][0], 12);
    EXPECT_EQ(Dnext[1][1], 32);
    EXPECT_EQ(Dnext[1][2], 20);
    EXPECT_EQ(Dnext[2][0], 6);
    EXPECT_EQ(Dnext[2][1], 16);
    EXPECT_EQ(Dnext[2][2], 10);

    Matrix<double> U(1, 1);
    U.Fill(6);
    A.getStep() = 2;
    auto Dnexts = A.ReversConvolution(U, F);
    out << Dnext;
    EXPECT_EQ(Dnexts[0][0], 6);
    EXPECT_EQ(Dnexts[0][1], 6);
    EXPECT_EQ(Dnexts[0][2], 0);
    EXPECT_EQ(Dnexts[1][0], 6);
    EXPECT_EQ(Dnexts[1][1], 6);
    EXPECT_EQ(Dnexts[1][2], 0);
    EXPECT_EQ(Dnexts[2][0], 0);
    EXPECT_EQ(Dnexts[2][1], 0);
    EXPECT_EQ(Dnexts[2][2], 0);
}

TEST(CNNLearning_, ReversConvolution_wrong_arguments){
    D_CNNLeaning A;
    A.getStep() = -4;
    Matrix<double> u(2, 2);
    Filter<double> uu(2, 2);
    EXPECT_ANY_THROW(A.ReversConvolution(u, uu));
}

TEST(CNNLearning_, GradDesc){
    D_CNNLeaning A;
    Matrix<double> X(3,3);
    for (int i = 0; i < 3; i++) {
        X[i][0] = 1;
        X[i][1] = 2;
        X[i][2] = 3;
    }
    Filter <double> F (2, 2);
    F.Fill(1);
    NeyronCnn<double> g;
    auto Out = g.Svertka(F, X);
    A.GradDes(X, Out, F);
    EXPECT_EQ(F[0][0], -51);
    EXPECT_EQ(F[0][1], -83);
    EXPECT_EQ(F[1][0], -51);
    EXPECT_EQ(F[1][1], -83);
    std::ofstream out("CNNLearningTest.txt");
    out << F;
}

TEST(CNNLearning_, ReversPooling){
    D_CNNLeaning A;
    Matrix<double> ppp(2, 2);
    ppp[0][0] = 1;
    ppp[0][1] = 2;
    ppp[1][0] = 3;
    ppp[1][1] = 4;

    auto UUU = A.ReversPooling(ppp, 2, 2);

    EXPECT_EQ(UUU[0][0], 1);
    EXPECT_EQ(UUU[0][1], 1);
    EXPECT_EQ(UUU[0][2], 2);
    EXPECT_EQ(UUU[0][3], 2);
    EXPECT_EQ(UUU[1][0], 1);
    EXPECT_EQ(UUU[1][1], 1);
    EXPECT_EQ(UUU[1][2], 2);
    EXPECT_EQ(UUU[1][3], 2);
    EXPECT_EQ(UUU[2][0], 3);
    EXPECT_EQ(UUU[2][1], 3);
    EXPECT_EQ(UUU[2][2], 4);
    EXPECT_EQ(UUU[2][3], 4);
    EXPECT_EQ(UUU[3][0], 3);
    EXPECT_EQ(UUU[3][1], 3);
    EXPECT_EQ(UUU[3][2], 4);
    EXPECT_EQ(UUU[3][3], 4);

    std::ofstream out("CNNLearningTest.txt");
    out << UUU;
    out.close();
}