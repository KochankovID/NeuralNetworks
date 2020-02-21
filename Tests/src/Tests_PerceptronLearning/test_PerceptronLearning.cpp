#include "opencv2/ts.hpp"
#include <fstream>
#include "PLearns.h"
#include "Functors.h"

class Sigm : public DD_Func
{
public:
    Sigm(const double& a_) : DD_Func(), a(a_) {};
    double a;
    double operator()(const double& x) {
        double f = 1;
        const double e = 2.7182818284;
        for (int i = 0; i < a*x; i++)
        {
            f *= 1 / e;
        }
        f++;
        return 1 / f;
    }
    ~Sigm() {};
};

class SigmD : public Sigm
{
public:
    SigmD(const double& a_) : Sigm(a_) {};
    double operator()(const double& x) {
        double f = 1;
        f = Sigm::operator()(x)*(1 - Sigm::operator()(x));
        return f;
    }
    ~SigmD() {};
};


TEST(PerceptronLearning, Constructor){
    PerceptronLearning<double, double> B;
    EXPECT_EQ(B.getE(), 1);

    PerceptronLearning<int, int> O(0.4);
    EXPECT_EQ(O.getE(), 0.4);
}

TEST(PerceptronLearning, GetE){
    DD_Leaning A;
    EXPECT_EQ(A.getE(), 1);
}

TEST(PerceptronLearning, WTSimplePerceptron){
    DD_Leaning A;
    Matrix<double> O(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            O[i][j] = 1;
        }
    }
    Weights<double> OO(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            OO[i][j] = 2;
        }
    }
    A.WTSimplePerceptron(1, -1, OO, O);
    EXPECT_EQ(O[0][0], 1);
    EXPECT_EQ(O[0][1], 1);
    EXPECT_EQ(O[0][2], 1);

    EXPECT_EQ(O[1][0], 1);
    EXPECT_EQ(O[1][1], 1);
    EXPECT_EQ(O[1][2], 1);

    EXPECT_EQ(O[2][0], 1);
    EXPECT_EQ(O[2][1], 1);
    EXPECT_EQ(O[2][2], 1);

    EXPECT_EQ(OO[0][0], 4);
    EXPECT_EQ(OO[0][1], 4);
    EXPECT_EQ(OO[0][2], 4);

    EXPECT_EQ(OO[1][0], 4);
    EXPECT_EQ(OO[1][1], 4);
    EXPECT_EQ(OO[1][2], 4);

    EXPECT_EQ(OO[2][0], 4);
    EXPECT_EQ(OO[2][1], 4);
    EXPECT_EQ(OO[2][2], 4);
}

TEST(PerceptronLearning, RMS_error){
    DD_Leaning A;
    double yr[] = { 1,2,3 };
    double yyr[] = { 1,2,4 };
    EXPECT_EQ(A.RMS_error(yr, yyr, 3), 0.5);
}

TEST(PercetronLearning, PartDOutLay){
    DD_Leaning A;
    double yr[] = { 1,2,3 };
    double yyr[] = { 1,2,4 };
    EXPECT_EQ(A.PartDOutLay(yr[2], yyr[2]), 2);
}

TEST(PercetronLearning, BackPropagation){
    DD_Leaning A;
    Matrix<double> UUU(2, 2);
    UUU[0][0] = 1;
    UUU[0][1] = 1;
    UUU[1][0] = 1;
    UUU[1][1] = 1;

    Matrix<Weights<double>> II(2,2);
    Weights<double> IIu(2, 2);
    Matrix<double> III(2, 2);
    III[0][0] = 1;
    III[0][1] = 1;
    III[1][0] = 1;
    III[1][1] = 1;

    IIu[0][0] = 1;
    IIu[0][1] = 1;
    IIu[1][0] = 0;
    IIu[1][1] = 0;

    IIu.GetD() = 0.5;
    A.BackPropagation(II,IIu);
    EXPECT_EQ(II[0][0].GetD(), 0.5);
    EXPECT_EQ(II[0][1].GetD(), 0.5);
    EXPECT_EQ(II[1][0].GetD(), 0);
    EXPECT_EQ(II[1][1].GetD(), 0);

}

TEST(PercetronLearning, GradDes){
    DD_Leaning A;
    SigmD F(0.5);
    Matrix<Weights<double>> II(2,2);
    Weights<double> IIu(2, 2);
    Matrix<double> III(2, 2);

    III[0][0] = 1;
    III[0][1] = 1;
    III[1][0] = 1;
    III[1][1] = 1;

    IIu[0][0] = 1;
    IIu[0][1] = 1;
    IIu[1][0] = 0;
    IIu[1][1] = 0;
    A.BackPropagation(II,IIu);
    A.GradDes(IIu, III, F, 4);
    EXPECT_TRUE(IIu[0][0] <= 1.5);
    EXPECT_TRUE(IIu[0][1] <= 1.5);
    EXPECT_TRUE(IIu[1][0] <= 4);
    EXPECT_TRUE(IIu[1][1] <= 4);
}