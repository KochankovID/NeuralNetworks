#include <gtest/gtest.h>
#define TEST_SOM
#include "SOM.h"

using namespace NN;

class SOM_Methods : public ::testing::Test {
public:
    SOM_Methods() : som(2,2,2) { /* init protected members here */ }

    ~SOM_Methods() { /* free protected members here */ }

    void SetUp() {/* called before every test */}
    void TearDown() { /* called after every test */ }
public:
    SimpleInitializator<double> init;
    SOM som;
};

TEST(SOM_constructor, initializator_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(SOM(2,3,4););
}

TEST(SOM_constructor, initializator_correct){
    // Arrange
    SOM som = SOM(2,3,4);;

    // Act

    // Assert
    EXPECT_EQ(som.weights_.shape()[0], 2);
    EXPECT_EQ(som.weights_.shape()[1], 3);
    EXPECT_EQ(som.weights_.shape()[2], 4);
    EXPECT_EQ(som.learning_rate_, 0.5);
    EXPECT_EQ(som.radius_, 1.0);
}

TEST(SOM_constructor, initializator_wrong_size_of_map){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(SOM(-1,4,4));
    EXPECT_ANY_THROW(SOM(3,0,4));
}

TEST(SOM_constructor, initializator_wrong_input_size){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(SOM(4,4,-2));
    EXPECT_ANY_THROW(SOM(3,2,0));
}

TEST(SOM_constructor, copy_works){
    // Arrange
    SOM som = SOM(2,3,4);

    // Act
    // Assert
    EXPECT_NO_THROW(SOM som1(som));
}

TEST(SOM_constructor, copy_correct){
    // Arrange
    SOM som = SOM(2,3,4);

    // Act
    SOM som1(som);

    // Assert
    EXPECT_EQ(som1.weights_.shape()[0], 2);
    EXPECT_EQ(som1.weights_.shape()[1], 3);
    EXPECT_EQ(som1.weights_.shape()[2], 4);
    EXPECT_EQ(som1.learning_rate_, 0.5);
    EXPECT_EQ(som1.radius_, 1.0);
}

TEST_F(SOM_Methods, random_weights_init_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(som.random_weights_init(init));
}

TEST_F(SOM_Methods, random_weights_init_correct){
    // Arrange
    Zeros<double > init1;

    // Act
    som.random_weights_init(init1);

    // Assert
    for(int i = 0; i < 6; i++) {
        EXPECT_EQ(som.weights_[i], 0);
    }
}

TEST_F(SOM_Methods, euclidean_distance_works){
    // Arrange
    Ndarray<double > one({4});
    Ndarray<double > two({4});

    // Act
    one.fill(1);
    two.fill(1);

    // Assert
    EXPECT_NO_THROW(som.euclidean_distance(one, two));
}

TEST_F(SOM_Methods, euclidean_distance_1_2_correct){
    // Arrange
    Ndarray<double > one({4});
    Ndarray<double > two({4});

    // Act
    one.fill(1);
    two.fill(2);
    double result = som.euclidean_distance(one, two);

    // Assert
    EXPECT_EQ(result, 2);
}

TEST_F(SOM_Methods, euclidean_distance_2_2_correct){
    // Arrange
    Ndarray<double > one({4});
    Ndarray<double > two({4});

    // Act
    one.fill(2);
    two.fill(2);
    double result = som.euclidean_distance(one, two);

    // Assert
    EXPECT_EQ(result, 0);
}

TEST_F(SOM_Methods, euclidean_distance_4_2_correct){
    // Arrange
    Ndarray<double > one({4});
    Ndarray<double > two({4});

    // Act
    one.fill(4);
    two.fill(2);
    double result = som.euclidean_distance(one, two);

    // Assert
    EXPECT_EQ(result, 4);
}

TEST_F(SOM_Methods, winner_works){
    // Arrange
    Ndarray<double > ndarray({2});

    // Act
    ndarray.fill(2);
    som.random_weights_init(Zeros<double>());
    som.weights_[0] = 2;
    som.weights_[1] = 2;
    som.weights_[2] = 2;
    som.weights_[3] = 2;

    // Assert
    EXPECT_NO_THROW(som.winner(ndarray));
}

TEST_F(SOM_Methods, winner_correct){
    // Arrange
    Ndarray<double > ndarray({2});

    // Act
    ndarray.fill(2);
    som.random_weights_init(Zeros<double>());
    som.weights_[0] = 2;
    som.weights_[1] = 2;
    som.weights_[2] = 2;
    som.weights_[3] = 2;
    auto index = som.winner(ndarray);

    // Assert
    EXPECT_EQ(index.size(), 2);
    EXPECT_EQ(index[0], 0);
    EXPECT_EQ(index[1], 0);
}