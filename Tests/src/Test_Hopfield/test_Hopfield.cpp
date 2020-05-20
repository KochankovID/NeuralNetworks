#include <gtest/gtest.h>
#define Hopfield_TEST
#include "Hopfield.h"

using namespace NN;

TEST(Hopfield_constructor, initializer_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(Hopfield(10));
}

TEST(Hopfield_constructor, initializer_correct){
    // Arrange
    // Act
    Hopfield hopfield(10);

    // Assert
    EXPECT_EQ(hopfield.weights_.shape().size(), 2);
    EXPECT_EQ(hopfield.weights_.shape()[0], 10);
    EXPECT_EQ(hopfield.weights_.shape()[0], 10);
}

TEST(Hopfield_method, train_works){
    // Arrange
    Hopfield hopfield(3);
    Ndarray<int> ndarray(1,3);

    // Act
    ndarray[0] = -1;
    ndarray[1] = 1;
    ndarray[2] = -1;

    // Assert
    EXPECT_ANY_THROW(hopfield.train(ndarray));
}

TEST(Hopfield_method, train_correct){
    // Arrange
    Hopfield hopfield(3);
    Ndarray<int> ndarray(2,1,3);

    // Act
    ndarray[0] = -1;
    ndarray[1] = 1;
    ndarray[2] = -1;
    hopfield.train(ndarray);

    // Assert
    EXPECT_EQ(hopfield.weights_[0], 0);
    EXPECT_EQ(hopfield.weights_[1], -1);
    EXPECT_EQ(hopfield.weights_[2], 1);

    EXPECT_EQ(hopfield.weights_[3], -1);
    EXPECT_EQ(hopfield.weights_[4], 0);
    EXPECT_EQ(hopfield.weights_[5], -1);

    EXPECT_EQ(hopfield.weights_[6], 1);
    EXPECT_EQ(hopfield.weights_[7], -1);
    EXPECT_EQ(hopfield.weights_[8], 0);
}