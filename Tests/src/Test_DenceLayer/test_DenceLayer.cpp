#include <opencv2/ts.hpp>
#define TEST_DenceLayer
#include "DenceLayer.h"

using namespace ANN;
#define MAT_TEST(X,Y) for(size_t iii = 0; iii < X.getN(); iii++){ for(size_t jjj = 0; jjj < X.getM(); jjj++){ EXPECT_EQ(X[iii][jjj], Y); }}

class DenceLayer_methods : public ::testing::Test {
public:
    DenceLayer_methods() : dence1(2,10,F_2, f_2,I3, 0.0) {
        F_2 = Relu<double >(1);
        f_2 = ReluD<double >(1);
        I3 = SimpleInitializator<double>(1);
    }

    ~DenceLayer_methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        dence1[0][0].Fill(1);
        dence1[0][1].Fill(1);
    }
    void TearDown() { /* called after every test */ }
public:
    Relu<double> F_2;
    ReluD<double> f_2;
    SimpleInitializator<double> I3;
    DenceLayer<double> dence1;
};

TEST(DenceLayer_constructor, initializator_Test_works){
    //Arrange
    Relu<double> F_2(1);
    ReluD<double> f_2(1);
    SimpleInitializator<double> I3(1);
    // Act

    // Assert
    EXPECT_NO_THROW(D_DenceLayer dence1(2,10,F_2, f_2,I3, 0.0));
}

TEST(DenceLayer_constructor, initializator_Test){
    //Arrange
    Relu<double> F_2(1);
    ReluD<double> f_2(1);
    SimpleInitializator<double> I3(1);

    // Act
    D_DenceLayer dence1(2,10,F_2, f_2,I3, 0.0);

    // Assert
    EXPECT_EQ(dence1.derivative.getN(), 1);
    EXPECT_EQ(dence1.derivative.getM(), 2);
    EXPECT_EQ(dence1.history.getN(), 1);
    EXPECT_EQ(dence1.history.getM(), 2);
    EXPECT_EQ(dence1.dropout, 0);
    EXPECT_EQ(dence1.getN(), 1);
    EXPECT_EQ(dence1.getM(), 2);
    EXPECT_EQ(dence1[0][0].getN(), 1);
    EXPECT_EQ(dence1[0][0].getM(), 10);
    EXPECT_EQ(dence1[0][1].getN(), 1);
    EXPECT_EQ(dence1[0][1].getM(), 10);
    EXPECT_EQ(dence1.history[0][0].getN(), dence1[0][0].getN());
    EXPECT_EQ(dence1.history[0][0].getM(), dence1[0][0].getM());
    EXPECT_EQ(dence1.history[0][1].getN(), dence1[0][1].getN());
    EXPECT_EQ(dence1.history[0][1].getM(), dence1[0][1].getM());
    for(int i = 0; i < dence1.getM();i++ ){
        for(int j = 0; j < dence1[0][0].getN(); j++){
            for(int k = 0; k < dence1[0][0].getM(); k++){
                EXPECT_TRUE((dence1[0][i][j][k] > -1)&&(dence1[0][i][j][k] < 1)&&(dence1[0][i][j][k] != 0));
            }
        }
    }
}

TEST(DenceLayer_constructor, copy_Test_works){
    //Arrange
    Relu<double> F_2(1);
    ReluD<double> f_2(1);
    SimpleInitializator<double> I3(1);
    D_DenceLayer dence2(2,10,F_2, f_2,I3, 0.0);

    // Act

    // Assert
    EXPECT_NO_THROW(D_DenceLayer dence1(dence2));
}

TEST(DenceLayer_constructor, copy_Test){
    // Arrange
    Relu<double> F_2(1);
    ReluD<double> f_2(1);
    SimpleInitializator<double> I3(1);

    // Act
    D_DenceLayer dence2(2,10,F_2, f_2,I3, 0.0);
    D_DenceLayer dence1(dence2);
    // Assert
    EXPECT_EQ(dence1.derivative.getN(), 1);
    EXPECT_EQ(dence1.derivative.getM(), 2);
    EXPECT_EQ(dence1.history.getN(), 1);
    EXPECT_EQ(dence1.history.getM(), 2);
    EXPECT_EQ(dence1.dropout, 0);
    EXPECT_EQ(dence1.getN(), 1);
    EXPECT_EQ(dence1.getM(), 2);
    EXPECT_EQ(dence1[0][0].getN(), 1);
    EXPECT_EQ(dence1[0][0].getM(), 10);
    EXPECT_EQ(dence1[0][1].getN(), 1);
    EXPECT_EQ(dence1[0][1].getM(), 10);
    EXPECT_EQ(dence1.history[0][0].getN(), dence1[0][0].getN());
    EXPECT_EQ(dence1.history[0][0].getM(), dence1[0][0].getM());
    EXPECT_EQ(dence1.history[0][1].getN(), dence1[0][1].getN());
    EXPECT_EQ(dence1.history[0][1].getM(), dence1[0][1].getM());
    for(int i = 0; i < dence1.getM();i++ ){
        for(int j = 0; j < dence1[0][0].getN(); j++){
            for(int k = 0; k < dence1[0][0].getM(); k++){
                EXPECT_TRUE((dence1[0][i][j][k] > -1)&&(dence1[0][i][j][k] < 1)&&(dence1[0][i][j][k] != 0));
            }
        }
    }
}

TEST_F(DenceLayer_methods, passThrough_Test_works){
    // Arrange
    Tensor<double> in(1, 10, 1);

    // Act

    // Assert
    EXPECT_NO_THROW(dence1.passThrough(in));
}

TEST_F(DenceLayer_methods, passThrough_Test_all_one){
    // Arrange
    Tensor<double> in(1, 10, 1);
    Tensor<double> out;

    // Act
    in.Fill(1);
    out = dence1.passThrough(in);

    // Assert
    EXPECT_EQ(out[0][0][0], 10);
    EXPECT_EQ(dence1.derivative[0][0], 1);
}

TEST_F(DenceLayer_methods, passThrough_Test_some_one){
    // Arrange
    Tensor<double> in(1, 10, 1);
    Tensor<double> out;

    // Act
    in.Fill(1);
    in[0][0][0] = 0;
    in[0][0][9] = 0;
    out = dence1.passThrough(in);

    // Assert
    EXPECT_EQ(out[0][0][0], 8);
    EXPECT_EQ(dence1.derivative[0][0], 1);
}

TEST_F(DenceLayer_methods, BackPropagation_one_neyron_Test_works){
    // Arrange
    Tensor<double> error(1, 1, 1);
    Tensor<double> in(1,10,1);

    // Act
    error.Fill(-1);
    in.Fill(1);

    // Assert
    EXPECT_NO_THROW(dence1.BackPropagation(error, in));
}

TEST_F(DenceLayer_methods, BackPropagation_one_neyron_Test){
    // Arrange
    Tensor<double> error(1, 1, 1);
    Tensor<double> in(1,10,1);

    // Act
    error.Fill(20);
    in.Fill(1);
    dence1.derivative.Fill(1);

    dence1.BackPropagation(error, in);
    // Assert
    EXPECT_EQ(dence1[0][0].GetD(), 20);

}

TEST_F(DenceLayer_methods, BackPropagation_two_neyrons_Test_works){
    // Arrange
    Tensor<double> error(1, 1, 1);
    Tensor<double> in(1,10,1);

    // Act
    error.Fill(-6);

    // Assert
    EXPECT_NO_THROW(dence1.BackPropagation(error, in));
}

TEST_F(DenceLayer_methods, BackPropagation_two_neyron_Test){
    // Arrange
    Tensor<double> error(1, 2, 1);
    Tensor<double> in(1,10,1);

    // Act
    error.Fill(-6);
    dence1.derivative.Fill(1);

    dence1.BackPropagation(error, in);
    // Assert
    EXPECT_EQ(dence1[0][0].GetD(), -6);
    EXPECT_EQ(dence1[0][1].GetD(), -6);

}

TEST_F(DenceLayer_methods, BackPropagation_two_neyron_with_derivative_Test){
    // Arrange
    Tensor<double> error(1, 2, 1);
    Tensor<double> in(1,10,1);

    // Act
    error.Fill(-6);
    dence1.derivative.Fill(0.5);

    dence1.BackPropagation(error, in);
    // Assert
    EXPECT_EQ(dence1[0][0].GetD(), -3);
    EXPECT_EQ(dence1[0][1].GetD(), -3);

}

TEST_F(DenceLayer_methods, GradDence_Test_works){
    // Arrange
    Tensor<double> in(1,10,1);
    SGD<double> G;

    // Act
    dence1[0][0].GetD() = 1;
    dence1[0][1].GetD() = 1;

    // Assert
    EXPECT_NO_THROW(dence1.GradDes(G, in));
}

TEST_F(DenceLayer_methods, GradDence_Test){
    // Arrange
    Tensor<double> in(1,10,1);
    SGD<double> G;

    // Act
    in.Fill(1);
    dence1[0][0].GetD() = 1;
    dence1[0][1].GetD() = 1;
    dence1.GradDes(G, in);

    // Assert
    MAT_TEST(dence1[0][0], 0);
    MAT_TEST(dence1[0][1], 0);
    EXPECT_EQ(dence1[0][0].GetD(), 0);
    EXPECT_EQ(dence1[0][1].GetD(), 0);
}

TEST_F(DenceLayer_methods, GradDence_not_all_one_Test){
    // Arrange
    Tensor<double> in(1,10,1);
    SGD<double> G;

    // Act
    in.Fill(1);
    in[0][0][1] = 0;
    in[0][0][5] = 0;
    dence1[0][0].GetD() = 1;
    dence1[0][1].GetD() = 1;
    dence1.GradDes(G, in);

    // Assert
    EXPECT_EQ(dence1[0][0][0][0], 0);
    EXPECT_EQ(dence1[0][0][0][1], 1);
    EXPECT_EQ(dence1[0][0][0][2], 0);
    EXPECT_EQ(dence1[0][0][0][3], 0);
    EXPECT_EQ(dence1[0][0][0][4], 0);
    EXPECT_EQ(dence1[0][0][0][5], 1);
    EXPECT_EQ(dence1[0][0][0][6], 0);
    EXPECT_EQ(dence1[0][0][0][7], 0);
    EXPECT_EQ(dence1[0][0][0][8], 0);
    EXPECT_EQ(dence1[0][0][0][9], 0);

    EXPECT_EQ(dence1[0][1][0][0], 0);
    EXPECT_EQ(dence1[0][1][0][1], 1);
    EXPECT_EQ(dence1[0][1][0][2], 0);
    EXPECT_EQ(dence1[0][1][0][3], 0);
    EXPECT_EQ(dence1[0][1][0][4], 0);
    EXPECT_EQ(dence1[0][1][0][5], 1);
    EXPECT_EQ(dence1[0][1][0][6], 0);
    EXPECT_EQ(dence1[0][1][0][7], 0);
    EXPECT_EQ(dence1[0][1][0][8], 0);
    EXPECT_EQ(dence1[0][1][0][9], 0);
}

TEST_F(DenceLayer_methods, GradDence_history_Test){
    // Arrange
    Tensor<double> in(1,10,1);
    SGD<double> G(1, 0.5);


    // Act
    in.Fill(1);
    dence1[0][0].GetD() = 1;
    dence1[0][1].GetD() = 1;
    dence1.history[0][0].Fill(2);
    dence1.history[0][1].Fill(2);
    dence1.GradDes(G, in);


    // Assert
    MAT_TEST(dence1[0][0], -0.5);
    MAT_TEST(dence1[0][1], -0.5);
    EXPECT_EQ(dence1[0][0].GetD(), 0);
    EXPECT_EQ(dence1[0][1].GetD(), 0);
}

