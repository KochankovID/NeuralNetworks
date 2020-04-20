#include <gtest/gtest.h>
#define TEST_FlatternLayer
#include "NN/Model/FlattenLayer/FlattenLayer.h"

using namespace NN;
#define MAT_TEST(X,Y) for(size_t iii = 0; iii < X.getN(); iii++){ for(size_t jjj = 0; jjj < X.getM(); jjj++){ EXPECT_EQ(X[iii][jjj], Y); }}


TEST(FlatternLayer_constructor, initializator_Test_works){
    //Arrange

    // Act

    // Assert
    EXPECT_NO_THROW(D_FlattenLayer flat1(5,5,12););
}

TEST(FlatternLayer_constructor, initializator_Test){
    //Arrange

    // Act
    D_FlattenLayer flat1(5,5,12);

    // Assert
    EXPECT_EQ(flat1.height, 5);
    EXPECT_EQ(flat1.width, 5);
    EXPECT_EQ(flat1.depth, 12);
}

TEST(FlatternLayer_constructor, copy_Test_works){
    //Arrange
    D_FlattenLayer flat2(5,5,12);
    // Act

    // Assert
    EXPECT_NO_THROW(D_FlattenLayer flat1(flat2););
}

TEST(FlatternLayer_constructor, copy_Test){
    //Arrange
    D_FlattenLayer flat2(5,5,12);

    // Act
    D_FlattenLayer flat1(flat2);

    // Assert
    EXPECT_EQ(flat1.height, 5);
    EXPECT_EQ(flat1.width, 5);
    EXPECT_EQ(flat1.depth, 12);
}

TEST(FlatternLayer_methods, passThrough_Test_works){
    //Arrange
    D_FlattenLayer flat1(5,5,12);
    Tensor<double> out;
    Tensor<double> in(5, 5, 12);

    // Act
    in.Fill(1);

    // Assert
    EXPECT_NO_THROW(out = flat1.passThrough(in));
}

TEST(FlatternLayer_methods, passThrough_Test){
    //Arrange
    D_FlattenLayer flat1(5,5,12);
    Tensor<double> out;
    Tensor<double> in(5, 5, 12);

    // Act
    in.Fill(1);
    out = flat1.passThrough(in);

    // Assert
    EXPECT_EQ(out.getDepth(), 1);
    EXPECT_EQ(out.getHeight(), 1);
    EXPECT_EQ(out.getWidth(), 300);

    for(int i =0; i < 300; i++){
        EXPECT_EQ(out[0][0][i], 1);
    }
}

TEST(FlatternLayer_methods, BackPropagation_Test){
    //Arrange
    D_FlattenLayer flat1(5,5,12);
    Tensor<double> error(1, 300, 1);
    Tensor<double> in(5, 5, 12);

    // Act
    error.Fill(1);

    // Assert
    EXPECT_NO_THROW(flat1.BackPropagation(error, in));
}

TEST(FlatternLayer_methods, BackPropagation_Test_works){
    //Arrange
    D_FlattenLayer flat1(5,5,12);
    Tensor<double> error(1, 300, 1);
    Tensor<double> in(5, 5, 12);
    Tensor<double> out;

    // Act
    error.Fill(1);
    out = flat1.BackPropagation(error, in);

    // Assert
    EXPECT_EQ(out.getDepth(), 12);
    EXPECT_EQ(out.getHeight(), 5);
    EXPECT_EQ(out.getWidth(), 5);

    for(int i = 0; i  < 12; i++) {
        MAT_TEST(out[i], 1);
    }
}