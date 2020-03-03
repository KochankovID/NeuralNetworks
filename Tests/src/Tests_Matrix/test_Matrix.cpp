#include "opencv2/ts.hpp"
#include <fstream>
#include "Matrix.h"

class TestSerialization : public ::testing::Test {
public:
    TestSerialization() { /* init protected members here */ }

    ~TestSerialization() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        Matrix<int> A(3,3);
        Matrix<double> B(3,3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i][j] = i;
                B[i][j] = j;
            }
        }
    }

    void TearDown() { /* called after every test */ }
};

TEST(Matrix, Matrix_Constructor_by_default_Test){
    // Arrange
    Matrix<int> m;

    // Act

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
}

TEST(Matrix, Matrix_Constructor_initial_first_square_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Matrix<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_first_not_square_one_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[50];
        for(size_t j = 0; j < 50; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Matrix<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_first_not_square_two_Test){
    // Arrange
    int** arr = new int*[50];
    for(size_t i = 0; i < 50; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Matrix<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_first_wrong_negative_size_Test){
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
    EXPECT_ANY_THROW(Matrix<int> m(arr, -2, -2));
}

TEST(Matrix, Matrix_Constructor_initial_second_square_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = 3;
    }

    // Act
    Matrix<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_second_not_square_one_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Matrix<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_second_not_square_two_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Matrix<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Matrix, Matrix_Constructor_initial_second_wrong_negative_size_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Matrix<int> m(arr, -2, -2));
}

TEST(Matrix, Matrix_Matrix_Constructor_initial_third_Test){
    // Arrange
    int* arr = new int[9000];
    for(size_t i = 0; i < 9000; i++){
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Matrix<int> m(arr, 100, 100));
}
