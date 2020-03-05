#include "opencv2/ts.hpp"
#include <fstream>
#include "Weights.h"

class Weights_Methods : public ::testing::Test {
public:
    Weights_Methods() { /* init protected members here */ }

    ~Weights_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Weights<int>(3, 3, 5);
        B = Weights<double>(3, 3, 6);
        A.GetD() = 10;
        B.GetD() = 15;
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

TEST(Weights_Constructor, By_default_Test) {
    // Arrange

    // Act
    Weights<int> m;

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
    EXPECT_EQ(m.GetD(), 0);
    EXPECT_EQ(m.GetWBias(), 0);
}

TEST(Weights_Constructor, Initial_first_square_Test) {
    // Arrange
    int **arr = new int *[100];
    for (size_t i = 0; i < 100; i++) {
        arr[i] = new int[100];
        for (size_t j = 0; j < 100; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Weights<int> m(arr, 100, 100, 5);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 5);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], i + j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_not_square_one_Test) {
    // Arrange
    int **arr = new int *[100];
    for (size_t i = 0; i < 100; i++) {
        arr[i] = new int[50];
        for (size_t j = 0; j < 50; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Weights<int> m(arr, 100, 50, 3);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    EXPECT_EQ(m.GetWBias(), 3);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 50; j++) {
            EXPECT_EQ(m[i][j], i + j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_not_square_two_Test) {
    // Arrange
    int **arr = new int *[50];
    for (size_t i = 0; i < 50; i++) {
        arr[i] = new int[100];
        for (size_t j = 0; j < 100; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Weights<int> m(arr, 50, 100, 8);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 8);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 50; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], i + j);
        }
    }
}

TEST(Weights_Constructor, Initial_first_wrong_negative_size_Test) {
    // Arrange
    int **arr = new int *[100];
    for (size_t i = 0; i < 100; i++) {
        arr[i] = new int[100];
        for (size_t j = 0; j < 100; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(arr, -2, -2, 8));
}

TEST(Weights_Constructor, Initial_second_square_Test) {
    // Arrange
    int *arr = new int[10000];
    for (size_t i = 0; i < 10000; i++) {
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 0);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_not_square_one_Test) {
    // Arrange
    int *arr = new int[5000];
    for (size_t i = 0; i < 5000; i++) {
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 100, 50, 5);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    EXPECT_EQ(m.GetWBias(), 5);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 50; j++) {
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_not_square_two_Test) {
    // Arrange
    int *arr = new int[5000];
    for (size_t i = 0; i < 5000; i++) {
        arr[i] = 3;
    }

    // Act
    Weights<int> m(arr, 50, 100, 19);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 19);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 50; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Weights_Constructor, Initial_second_wrong_negative_size_Test) {
    // Arrange
    int *arr = new int[10000];
    for (size_t i = 0; i < 10000; i++) {
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(arr, -2, -2));
}

TEST(Weights_Constructor, Initial_third_square_Test) {
    // Arrange

    // Act
    Weights<int> m(100, 100, 55);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 55);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_not_square_one_Test) {
    // Arrange

    // Act
    Weights<int> m(50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 0);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 50; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_not_square_two_Test) {
    // Arrange

    // Act
    Weights<int> m(100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    EXPECT_EQ(m.GetWBias(), 0);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 50; j++) {
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Weights_Constructor, Initial_third_wrong_negative_size_Test) {
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Weights<int> m(-2, -2));
}

TEST(Weights_Constructor, Copy_Test) {
    // Arrange
    Weights<int> t(100, 100, 9);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            t[i][j] = i + j;
        }
    }

    // Act
    Weights<int> m(t);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    EXPECT_EQ(m.GetWBias(), 9);
    EXPECT_EQ(m.GetD(), 0);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            EXPECT_EQ(m[i][j], i + j);
        }
    }
}

TEST_F(Weights_Methods, GetD_get_Test) {
    // Arrange

    // Act

    // Assert
    EXPECT_EQ(A.GetD(), 10);
    EXPECT_EQ(B.GetD(), 15);
}

TEST_F(Weights_Methods, GetD_set_Test) {
    // Arrange

    // Act
    A.GetD() = 49;
    B.GetD() = 4;

    // Assert
    EXPECT_EQ(A.GetD(), 49);
    EXPECT_EQ(B.GetD(), 4);
}

TEST_F(Weights_Methods, GetD_get_const_Test) {
    // Arrange
    Weights<int> R(1,1);

    // Act
    R.GetD() = 100;
    const Weights<int> Y(R);

    // Assert
    EXPECT_EQ(Y.GetD(), 100);
}

TEST_F(Weights_Methods, GetWBias_get_Test) {
    // Arrange

    // Act

    // Assert
    EXPECT_EQ(A.GetWBias(), 5);
    EXPECT_EQ(B.GetWBias(), 6);
}

TEST_F(Weights_Methods, GetWBias_set_Test) {
    // Arrange

    // Act
    A.GetWBias() = 49;
    B.GetWBias() = 4;

    // Assert
    EXPECT_EQ(A.GetWBias(), 49);
    EXPECT_EQ(B.GetWBias(), 4);
}

TEST_F(Weights_Methods, GetWBias_get_const_Test) {
    // Arrange
    Weights<int> R(1,1);

    // Act
    R.GetWBias() = 100;
    const Weights<int> Y(R);


    // Assert
    EXPECT_EQ(Y.GetWBias(), 100);
}

TEST_F(Weights_Methods, outsrteam_operator){
    // Arrange
    std::ofstream file;
    std::ifstream fileIn;

    int n, m, d, w;
    int arr[9];

    // Act
    file.open("WeightsTest.txt");
    EXPECT_NO_THROW(file << A);
    file.close();
    fileIn.open("WeightsTest.txt");
    fileIn >> n;
    fileIn >> m;
    for(size_t i = 0; i < 9; i++){
        fileIn >> arr[i];
    }
    fileIn >> d;
    fileIn >> w;

    // Assert
    EXPECT_EQ(n, 3);
    EXPECT_EQ(m, 3);
    EXPECT_EQ(w, 5);
    EXPECT_EQ(d, 10);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(arr[i*3+j], A[i][j]);
        }
    }
}

TEST_F(Weights_Methods, intsrteam_operator){
    // Arrange
    Weights<int> M;
    std::ofstream file;
    std::ifstream fileIn;

    // Act
    file.open("WeightsTest.txt");
    file << A;
    file.close();
    fileIn.open("WeightsTest.txt");
    EXPECT_NO_THROW(fileIn >> M);

    // Assert
    EXPECT_EQ(M.getN(), 3);
    EXPECT_EQ(M.getM(), 3);
    EXPECT_EQ(M.GetWBias(), 5);
    EXPECT_EQ(M.GetD(), 10);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(M[i][j], A[i][j]);
        }
    }
}