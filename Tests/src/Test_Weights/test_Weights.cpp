#include "opencv2/ts.hpp"
#include <fstream>
#include "Weights.h"

class Weights_Methods : public ::testing::Test {
public:
    Weights_Methods() { /* init protected members here */ }

    ~Weights_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Weights<int>(3,3);
        B = Weights<double>(3,3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i][j] = i;
                B[i][j] = j;
            }
        }
    }

    void TearDown() { /* called after every test */ }
public:
    Weights<int> A;
    Weights<double> B;
};

TEST(Weights_Constructor, By_default_Test){
    // Arrange

    // Act
    Weights<int> m;

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
}

TEST(Weights_Constructor, Initial_first_square_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Weights<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_not_square_one_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[50];
        for(size_t j = 0; j < 50; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Weights<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_not_square_two_Test){
    // Arrange
    int** arr = new int*[50];
    for(size_t i = 0; i < 50; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Weights<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_wrong_negative_size_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(arr, -2, -2));
}

TEST(Weights_Constructor, Initial_second_square_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_not_square_one_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_not_square_two_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_wrong_negative_size_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(arr, -2, -2));
}

TEST(Weights_Constructor, Initial_third_square_Test){
    // Arrange

    // Act
    Weights<int> m(100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_not_square_one_Test){
    // Arrange

    // Act
    Weights<int> m(50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_not_square_two_Test){
    // Arrange

    // Act
    Weights<int> m(100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_wrong_negative_size_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(-2, -2));
}

TEST(Weights_Constructor, Copy_Test){
    // Arrange
    Weights<int> t(100,100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            t[i][j] = i+j;
        }
    }

    // Act
    Weights<int> m(t);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}


