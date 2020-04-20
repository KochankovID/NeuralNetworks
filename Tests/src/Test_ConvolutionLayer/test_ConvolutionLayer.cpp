#include <opencv2/ts.hpp>
#define TEST_ConvLayer
#include "ConvolutionLayer.h"

using namespace ANN;
#define MAT_TEST(X,Y) for(size_t iii = 0; iii < X.getN(); iii++){ for(size_t jjj = 0; jjj < X.getM(); jjj++){ EXPECT_EQ(X[iii][jjj], Y); }}

class ConvolutionLayer_methods : public ::testing::Test {
public:
    ConvolutionLayer_methods() : conv1(2, 5,5,2, I1, 1) {
        I1 = SimpleInitializator<double>(1);
    }

    ~ConvolutionLayer_methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        conv1[0][0].Fill(1);
        conv1[0][1].Fill(1);
    }
    void TearDown() { /* called after every test */ }
public:
    SimpleInitializator<double> I1;
    ConvolutionLayer<double> conv1;
};

TEST(ConvolutionLayer_constructor, initializator_Test_works){
    //Arrange
    SimpleInitializator<double> I1(1);
    // Act

    // Assert
    EXPECT_NO_THROW(ConvolutionLayer<double> conv1(6, 5,5,1, I1, 1));
}

TEST(ConvolutionLayer_constructor, initializator_Test){
    //Arrange
    SimpleInitializator<double> I1(1);
    
    // Act
    ConvolutionLayer<double> conv1(6, 5,5,1, I1, 1);
    
    // Assert
    EXPECT_EQ(conv1.step_, 1);
    EXPECT_EQ(conv1.history.getN(), 1);
    EXPECT_EQ(conv1.history.getM(), 6);
    EXPECT_EQ(conv1.history[0][0].getN(), conv1[0][0].getN());
    EXPECT_EQ(conv1.history[0][0].getM(), conv1[0][0].getM());
    EXPECT_EQ(conv1.history[0][1].getN(), conv1[0][1].getN());
    EXPECT_EQ(conv1.history[0][1].getM(), conv1[0][1].getM());
    EXPECT_EQ(conv1.history[0][2].getN(), conv1[0][2].getN());
    EXPECT_EQ(conv1.history[0][2].getM(), conv1[0][2].getM());
    EXPECT_EQ(conv1.history[0][3].getN(), conv1[0][3].getN());
    EXPECT_EQ(conv1.history[0][3].getM(), conv1[0][3].getM());
    EXPECT_EQ(conv1.history[0][4].getN(), conv1[0][4].getN());
    EXPECT_EQ(conv1.history[0][4].getM(), conv1[0][4].getM());
    EXPECT_EQ(conv1.history[0][5].getN(), conv1[0][5].getN());
    EXPECT_EQ(conv1.history[0][5].getM(), conv1[0][5].getM());
    EXPECT_EQ(conv1.getN(), 1);
    EXPECT_EQ(conv1.getM(), 6);

    for(int i = 0 ; i < 6; i++) {
        EXPECT_EQ(conv1[0][i].getDepth(), 1);
        EXPECT_EQ(conv1[0][i].getHeight(), 5);
        EXPECT_EQ(conv1[0][i].getWidth(), 5);
    }
}

TEST(ConvolutionLayer_constructor, copy_Test_works){
    //Arrange
    SimpleInitializator<double> I1(1);

    // Act
    ConvolutionLayer<double> conv2(6, 5,5,1, I1, 1);

    // Assert
    EXPECT_NO_THROW(ConvolutionLayer<double> conv1(conv2));
}

TEST(ConvolutionLayer_constructor, copy_Test){
    //Arrange
    SimpleInitializator<double> I1(1);

    // Act
    ConvolutionLayer<double> conv2(6, 5,5,1, I1, 1);
    ConvolutionLayer<double> conv1(conv2);

    // Assert
    EXPECT_EQ(conv1.step_, 1);
    EXPECT_EQ(conv1.history.getN(), 1);
    EXPECT_EQ(conv1.history.getM(), 6);
    EXPECT_EQ(conv1.history[0][0].getN(), conv1[0][0].getN());
    EXPECT_EQ(conv1.history[0][0].getM(), conv1[0][0].getM());
    EXPECT_EQ(conv1.history[0][1].getN(), conv1[0][1].getN());
    EXPECT_EQ(conv1.history[0][1].getM(), conv1[0][1].getM());
    EXPECT_EQ(conv1.history[0][2].getN(), conv1[0][2].getN());
    EXPECT_EQ(conv1.history[0][2].getM(), conv1[0][2].getM());
    EXPECT_EQ(conv1.history[0][3].getN(), conv1[0][3].getN());
    EXPECT_EQ(conv1.history[0][3].getM(), conv1[0][3].getM());
    EXPECT_EQ(conv1.history[0][4].getN(), conv1[0][4].getN());
    EXPECT_EQ(conv1.history[0][4].getM(), conv1[0][4].getM());
    EXPECT_EQ(conv1.history[0][5].getN(), conv1[0][5].getN());
    EXPECT_EQ(conv1.history[0][5].getM(), conv1[0][5].getM());
    EXPECT_EQ(conv1.getN(), 1);
    EXPECT_EQ(conv1.getM(), 6);

    for(int i = 0 ; i < 6; i++) {
        EXPECT_EQ(conv1[0][i].getDepth(), 1);
        EXPECT_EQ(conv1[0][i].getHeight(), 5);
        EXPECT_EQ(conv1[0][i].getWidth(), 5);
    }
}

TEST_F(ConvolutionLayer_methods, passThrough_Test_works){
    // Arrange
    Tensor<double> in(6, 6, 2);

    // Act

    // Assert
    EXPECT_NO_THROW(conv1.passThrough(in));
}

TEST_F(ConvolutionLayer_methods, passThrough_Test){
    // Arrange
    Tensor<double> in(6, 6, 2);
    Tensor<double> out;

    // Act
    in.Fill(1);
    out = conv1.passThrough(in);

    // Assert
    EXPECT_EQ(out.getDepth(), 2);
    EXPECT_EQ(out.getHeight(), 2);
    EXPECT_EQ(out.getWidth(), 2);

    MAT_TEST(out[0], 50);
    MAT_TEST(out[1], 50);
}

TEST_F(ConvolutionLayer_methods, passThrough_not_all_one_Test){
    // Arrange
    Tensor<double> in(6, 6, 2);
    Tensor<double> out;

    // Act
    in.Fill(1);
    in[0][0][0] = 0;
    in[1][0][2] = 0;
    out = conv1.passThrough(in);

    // Assert
    EXPECT_EQ(out.getDepth(), 2);
    EXPECT_EQ(out.getHeight(), 2);
    EXPECT_EQ(out.getWidth(), 2);

    EXPECT_EQ(out[0][0][0], 48);
    EXPECT_EQ(out[0][0][1], 49);
    EXPECT_EQ(out[0][1][0], 50);
    EXPECT_EQ(out[0][1][1], 50);

    EXPECT_EQ(out[1][0][0], 48);
    EXPECT_EQ(out[1][0][1], 49);
    EXPECT_EQ(out[1][1][0], 50);
    EXPECT_EQ(out[1][1][1], 50);
}

TEST_F(ConvolutionLayer_methods, BackPropagation_Test_works){
    // Arrange
    Tensor<double> in(6, 6, 2);
    Tensor<double> error(2, 2, 2);

    // Act

    // Assert
    EXPECT_NO_THROW(conv1.BackPropagation(error, in));
}

TEST_F(ConvolutionLayer_methods, BackPropagation_Test){
    // Arrange
    conv1 = ConvolutionLayer<double>(1, 2,2,2, I1, 1);
    Tensor<double> in(3, 3, 2);
    Tensor<double> error(2, 2, 1);
    Tensor<double> out;

    // Act
    conv1[0][0].Fill(1);
    in.Fill(1);
    error.Fill(1);
    out = conv1.BackPropagation(error, in);

    // Assert
    EXPECT_EQ(out.getDepth(), 2);
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 2);
    EXPECT_EQ(out[0][0][2], 1);

    EXPECT_EQ(out[0][1][0], 2);
    EXPECT_EQ(out[0][1][1], 4);
    EXPECT_EQ(out[0][1][2], 2);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 2);
    EXPECT_EQ(out[0][2][2], 1);
}
TEST_F(ConvolutionLayer_methods, BackPropagation_two_Filters_Test){
    // Arrange
    conv1 = ConvolutionLayer<double>(2, 2,2,2, I1, 1);
    Tensor<double> in(3, 3, 2);
    Tensor<double> error(2, 2, 2);
    Tensor<double> out;

    // Act
    conv1[0][0].Fill(1);
    conv1[0][1].Fill(1);
    in.Fill(1);
    error.Fill(1);
    out = conv1.BackPropagation(error, in);

    // Assert
    EXPECT_EQ(out.getDepth(), 2);
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);

    EXPECT_EQ(out[0][0][0], 2);
    EXPECT_EQ(out[0][0][1], 4);
    EXPECT_EQ(out[0][0][2], 2);

    EXPECT_EQ(out[0][1][0], 4);
    EXPECT_EQ(out[0][1][1], 8);
    EXPECT_EQ(out[0][1][2], 4);

    EXPECT_EQ(out[0][2][0], 2);
    EXPECT_EQ(out[0][2][1], 4);
    EXPECT_EQ(out[0][2][2], 2);
}

TEST_F(ConvolutionLayer_methods, GradDence_Test_works){
    // Arrange
    SGD<double> G;
    Tensor<double> in(6, 6, 2);

    // Act
    conv1.error_ = Tensor<double>(2,2,2);
    conv1.error_.Fill(1);

    // Assert
    EXPECT_NO_THROW(conv1.GradDes(G, in));
}

TEST_F(ConvolutionLayer_methods, GradDence_Test){
    // Arrange
    SGD<double> G;
    Tensor<double> in(6, 6, 2);

    // Act
    in.Fill(1);
    conv1.error_ = Tensor<double>(2,2,2);
    conv1.error_.Fill(1);
    conv1.GradDes(G, in);

    // Assert
    EXPECT_EQ(conv1.getN(), 1);
    EXPECT_EQ(conv1.getM(), 2);

    for(int i =0 ; i < 2; i++){
        MAT_TEST(conv1[0][i][0], -3);
        MAT_TEST(conv1[0][i][1], -3);
    }
}

TEST_F(ConvolutionLayer_methods, GradDence_not_all_one_Test){
    // Arrange
    SGD<double> G;
    Tensor<double> in(6, 6, 2);

    // Act
    in.Fill(1);
    in[0][0][0] = 0;
    in[1][0][5] = 0;
    conv1.error_ = Tensor<double>(2,2,2);
    conv1.error_.Fill(1);
    conv1.GradDes(G, in);

    // Assert
    EXPECT_EQ(conv1.getN(), 1);
    EXPECT_EQ(conv1.getM(), 2);

    EXPECT_EQ(conv1[0][0][0][0][0], -2);
    EXPECT_EQ(conv1[0][0][1][0][4], -2);

    EXPECT_EQ(conv1[0][1][0][0][0], -2);
    EXPECT_EQ(conv1[0][1][1][0][4], -2);
}

TEST_F(ConvolutionLayer_methods, GradDence_history_Test){
    // Arrange
    SGD<double> G(1, 0.5);
    Tensor<double> in(6, 6, 2);

    // Act
    in.Fill(1);
    conv1.error_ = Tensor<double>(2,2,2);
    conv1.error_.Fill(1);
    conv1.history[0][0].Fill(2);
    conv1.history[0][1].Fill(2);
    conv1.GradDes(G, in);

    // Assert
    EXPECT_EQ(conv1.getN(), 1);
    EXPECT_EQ(conv1.getM(), 2);

    for(int i =0 ; i < 2; i++){
        MAT_TEST(conv1[0][i][0], -2);
        MAT_TEST(conv1[0][i][1], -2);
    }
}