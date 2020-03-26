#include "opencv2/ts.hpp"
#include <fstream>
#include <iostream>
#include "Tensor.h"
#include <functional>

using namespace ANN;

class Tensor_Methods : public ::testing::Test {
public:
    Tensor_Methods() { /* init protected members here */ }

    ~Tensor_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Tensor<int>(3,3,3);
        B = Tensor<double>(3,3, 3);
        for(int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    A[k][i][j] = i;
                    B[k][i][j] = j;
                }
            }
        }
    }
    void TearDown() { /* called after every test */ }
public:
    Tensor<int> A;
    Tensor<double> B;
};

TEST(Tensor_Constructor, By_default_Test){
    // Arrange

    // Act
    Tensor<int> m;

    // Assert
    EXPECT_EQ(m.getHeight(), 0);
    EXPECT_EQ(m.getWight(), 0);
    EXPECT_EQ(m.getDepth(), 0);
}

TEST(Tensor_Constructor, Initial_wrong_negative_height_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Tensor<int> m(-1, 2, 2));
}

TEST(Tensor_Constructor, Initial_wrong_negative_width_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Tensor<int> m(2, -2, 2));
}

TEST(Tensor_Constructor, Initial_wrong_negative_depth_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Tensor<int> m(2, 2, -2));
}

TEST(Tensor_Constructor, Initial_square_Test){
    // Arrange

    // Act
    Tensor<int> m(100, 100,1);

    // Assert
    EXPECT_EQ(m.getHeight(), 100);
    EXPECT_EQ(m.getWight(), 100);
    EXPECT_EQ(m.getDepth(), 1);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], 0);
        }
    }
}

TEST(Tensor_Constructor, Initial_not_square_one_Test){
    // Arrange

    // Act
    Tensor<int> m(50, 100,1);

    // Assert
    EXPECT_EQ(m.getHeight(), 50);
    EXPECT_EQ(m.getWight(), 100);
    EXPECT_EQ(m.getDepth(), 1);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], 0);
        }
    }
}

TEST(Tensor_Constructor, Initial_square_two_Test){
    // Arrange

    // Act
    Tensor<int> m(100, 50,1);

    // Assert
    EXPECT_EQ(m.getHeight(), 100);
    EXPECT_EQ(m.getWight(), 50);
    EXPECT_EQ(m.getDepth(), 1);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[0][i][j], 0);
        }
    }
}

TEST(Tensor_Constructor, Initial_3X3_Test){
    // Arrange

    // Act
    Tensor<int> m(3, 3, 3);

    // Assert
    EXPECT_EQ(m.getHeight(), 3);
    EXPECT_EQ(m.getWight(), 3);
    EXPECT_EQ(m.getDepth(), 3);

    for(size_t k = 0; k < 3; k++) {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                EXPECT_EQ(m[k][i][j], 0);
            }
        }
    }
}

TEST(Tensor_Constructor, Copy_Test){
    // Arrange
    Tensor<int> t(100,100,1);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            t[0][i][j] = i+j;
        }
    }

    // Act
    Tensor<int> m(t);

    // Assert
    EXPECT_EQ(m.getHeight(), 100);
    EXPECT_EQ(m.getWight(), 100);
    EXPECT_EQ(m.getDepth(), 1);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], i+j);
        }
    }
}

TEST_F(Tensor_Methods, Fill_Test){
    // Arrange

    // Act
    A.Fill(10);

    // Assert
    for(size_t k = 0; k < 3; k++) {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                EXPECT_EQ(A[k][i][j], 10);
            }
        }
    }
}

TEST_F(Tensor_Methods, Fill_zero_size_Test){
    // Arrange
    Tensor<int> a;

    // Act

    // Assert
    EXPECT_NO_THROW(a.Fill(10));
}

TEST_F(Tensor_Methods, zoom_one_place_Test) {
    // Arrange
    Tensor<int> R(2,2, 1);
    Tensor<int> new_R;

    // Act
    R.Fill(2);
    EXPECT_NO_THROW(new_R = R.zoom(1));

    // Assert
    EXPECT_EQ(new_R.getHeight(), 3);
    EXPECT_EQ(new_R.getWight(), 3);
    EXPECT_EQ(new_R.getDepth(), 1);

    EXPECT_EQ(new_R[0][0][0], 2);
    EXPECT_EQ(new_R[0][0][2], 2);
    EXPECT_EQ(new_R[0][2][0], 2);
    EXPECT_EQ(new_R[0][2][2], 2);
}

TEST_F(Tensor_Methods, zoom_two_place_Test) {
    // Arrange
    Tensor<int> R(2,2, 1);
    Tensor<int> new_R;

    // Act
    R.Fill(2);
    EXPECT_NO_THROW(new_R = R.zoom(2));

    // Assert
    EXPECT_EQ(new_R.getHeight(), 4);
    EXPECT_EQ(new_R.getWight(), 4);
    EXPECT_EQ(new_R.getDepth(), 1);

    EXPECT_EQ(new_R[0][0][0], 2);
    EXPECT_EQ(new_R[0][0][3], 2);
    EXPECT_EQ(new_R[0][3][0], 2);
    EXPECT_EQ(new_R[0][3][3], 2);
}

TEST_F(Tensor_Methods, zoom_wrong_place_zero_Test) {
    // Arrange
    Tensor<int> R(2, 2,2);

    // Act

    // Assert
    EXPECT_ANY_THROW(R.zoom(0));
}

TEST_F(Tensor_Methods, zoom_wrong_place_negative_Test) {
    // Arrange
    Tensor<int> R(2, 2, 1);

    // Act

    // Assert
    EXPECT_ANY_THROW(R.zoom(-1));
}