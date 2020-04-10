#include <opencv2/ts.hpp>
#define TEST_MaxLayer
#include "MaxpoolingLayers.h"

using namespace ANN;
#define MAT_TEST(X,Y) for(size_t iii = 0; iii < X.getN(); iii++){ for(size_t jjj = 0; jjj < X.getM(); jjj++){ EXPECT_EQ(X[iii][jjj], Y); }}

TEST(MaxpoolingLayer_constructor, initializator_Test_works){
    //Arrange

    // Act

    // Assert
    EXPECT_NO_THROW(D_MaxpoolingLayer maxp2(2,2););
}

TEST(MaxpoolingLayer_constructor, initializator_Test){
    //Arrange
    D_MaxpoolingLayer maxp2(2,2);

    // Act

    // Assert
    EXPECT_EQ(maxp2.m_, 2);
    EXPECT_EQ(maxp2.n_, 2);
}

TEST(MaxpoolingLayer_constructor, copy_Test_works){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);

    // Act

    // Assert
    EXPECT_NO_THROW(D_MaxpoolingLayer maxpz(maxp1););
}

TEST(MaxpoolingLayer_constructor, copy_Test){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);

    // Act
    D_MaxpoolingLayer maxp2(maxp1);

    // Assert
    EXPECT_EQ(maxp2.m_, 2);
    EXPECT_EQ(maxp2.n_, 2);
}

TEST(MaxpoolingLayer_methods, passThrough_works){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);
    Tensor<double> in(4, 4, 4);
    // Act

    // Assert
    EXPECT_NO_THROW(maxp1.passThrough(in));
}

TEST(MaxpoolingLayer_methods, passThrough_Test){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);
    Tensor<double> in(4, 4, 4);
    Tensor<double> out;

    // Act
    in.Fill(1);
    out = maxp1.passThrough(in);

    // Assert
    EXPECT_EQ(out.getHeight(), 2);
    EXPECT_EQ(out.getWidth(), 2);
    EXPECT_EQ(out.getDepth(), 4);

    MAT_TEST(out[0], 1);
    MAT_TEST(out[1], 1);
    MAT_TEST(out[2], 1);
    MAT_TEST(out[3], 1);

    EXPECT_EQ(out, maxp1.output);
}

TEST(MaxpoolingLayer_methods, BackPropagation_Test_works){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);
    Tensor<double> error(2, 2, 4);
    Tensor<double> input(4, 4, 4);

    // Act
    maxp1.output = error;

    // Assert
    EXPECT_NO_THROW(maxp1.BackPropagation(error, input));
}

TEST(MaxpoolingLayer_methods, BackPropagation_Test){
    //Arrange
    D_MaxpoolingLayer maxp1(2,2);
    Tensor<double> error(2, 2, 4);
    Tensor<double> input(4, 4, 4);
    Tensor<double> out;

    // Act
    for(int i = 0; i < 4; i++){
        input[i][0][0] = 1;
        input[i][0][1] = 0;
        input[i][1][0] = 0;
        input[i][1][1] = 0;
    }
    error.Fill(1);
    maxp1.output = error;
    out = maxp1.BackPropagation(error, input);

    // Assert
    EXPECT_EQ(out.getDepth(), 4);
    EXPECT_EQ(out.getHeight(), 4);
    EXPECT_EQ(out.getWidth(), 4);

    for(int i =0; i < 4; i++){
        EXPECT_EQ(out[i][0][0], 1);
        EXPECT_EQ(out[i][0][1], 0);
        EXPECT_EQ(out[i][1][0], 0);
        EXPECT_EQ(out[i][1][1], 0);
    }
}