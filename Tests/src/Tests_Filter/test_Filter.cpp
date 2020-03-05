#include "opencv2/ts.hpp"
#include <fstream>
#include "Filter.h"

using namespace ANN;

class Filter_Methods : public ::testing::Test {
public:
    Filter_Methods() { /* init protected members here */ }

    ~Filter_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Filter<int>(3,3);
        B = Filter<double>(3,3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i][j] = i;
                B[i][j] = j;
            }
        }
    }

    void TearDown() { /* called after every test */ }
public:
    Filter<int> A;
    Filter<double> B;
};

TEST(Filter_Constructor, By_default_Test){
    // Arrange

    // Act
    Filter<int> m;

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
}

TEST(Filter_Constructor, Initial_first_square_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Filter<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Filter_Constructor, Initial_first_not_square_one_Test){
    // Arrange
    int** arr = new int*[100];
    for(size_t i = 0; i < 100; i++){
        arr[i] = new int[50];
        for(size_t j = 0; j < 50; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Filter<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Filter_Constructor, Initial_first_not_square_two_Test){
    // Arrange
    int** arr = new int*[50];
    for(size_t i = 0; i < 50; i++){
        arr[i] = new int[100];
        for(size_t j = 0; j < 100; j++){
            arr[i][j] = i+j;
        }
    }

    // Act
    Filter<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST(Filter_Constructor, Initial_first_wrong_negative_size_Test){
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
    EXPECT_ANY_THROW(Filter<int> m(arr, -2, -2));
}

TEST(Filter_Constructor, Initial_second_square_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = 3;
    }

    // Act
    Filter<int> m(arr, 100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Filter_Constructor, Initial_second_not_square_one_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Filter<int> m(arr, 100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Filter_Constructor, Initial_second_not_square_two_Test){
    // Arrange
    int* arr = new int[5000];
    for(size_t i = 0; i < 5000; i++){
        arr[i] = 3;
    }

    // Act
    Filter<int> m(arr, 50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 3);
        }
    }
}

TEST(Filter_Constructor, Initial_second_wrong_negative_size_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Filter<int> m(arr, -2, -2));
}

TEST(Filter_Constructor, Initial_third_square_Test){
    // Arrange

    // Act
    Filter<int> m(100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Filter_Constructor, Initial_third_not_square_one_Test){
    // Arrange

    // Act
    Filter<int> m(50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Filter_Constructor, Initial_third_not_square_two_Test){
    // Arrange

    // Act
    Filter<int> m(100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Filter_Constructor, Initial_third_wrong_negative_size_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Filter<int> m(-2, -2));
}

TEST(Filter_Constructor, Copy_Test){
    // Arrange
    Filter<int> t(100,100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            t[i][j] = i+j;
        }
    }

    // Act
    Filter<int> m(t);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST_F(Filter_Methods, roate_180_Test){
    // Arrange

    // Act
    EXPECT_NO_THROW(A = A.roate_180());
    EXPECT_NO_THROW(B = B.roate_180());
    // Assert
    EXPECT_EQ(A.getN(), 3);
    EXPECT_EQ(A.getM(), 3);
    EXPECT_EQ(B.getN(), 3);
    EXPECT_EQ(B.getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(A[i][j], 2-i);
            EXPECT_EQ(B[i][j], 2-j);
        }
    }
}

TEST_F(Filter_Methods, roate_180_null_size_Test){
    // Arrange
    Filter<int> T(0,0);

    // Act
    EXPECT_NO_THROW(T = T.roate_180());
    // Assert
    EXPECT_EQ(T.getN(), 0);
    EXPECT_EQ(T.getM(), 0);
}

TEST_F(Filter_Methods, assignment_operator_bigger_size_Test){
    // Arrange
    Filter<int> D(4,4);
    D.Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A.getN(), 4);
    EXPECT_EQ(A.getM(), 4);
    for(size_t i = 0; i < 4; i++){
        for(size_t j = 0; j < 4; j++){
            EXPECT_EQ(A[i][j], 5);
        }
    }
}

TEST_F(Filter_Methods, assignment_operator_smaller_size_Test){
    // Arrange
    Filter<int> D(2, 2);
    D.Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A.getN(), 2);
    EXPECT_EQ(A.getM(), 2);
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            EXPECT_EQ(A[i][j], 5);
        }
    }
}

TEST_F(Filter_Methods, assignment_operator_zero_size_Test){
    // Arrange
    Filter<int> D(0, 0);

    // Act
    EXPECT_NO_THROW(D = A);


    // Assert
    EXPECT_EQ(D.getN(), 3);
    EXPECT_EQ(D.getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[i][j], i);
        }
    }
}

TEST_F(Filter_Methods, outsrteam_operator){
    // Arrange
    std::ofstream file;
    std::ifstream fileIn;

    int n, m;
    int arr[9];

    // Act
    file.open("FilterTest.txt");
    EXPECT_NO_THROW(file << A);
    file.close();
    fileIn.open("FilterTest.txt");
    fileIn >> n;
    fileIn >> m;
    for(size_t i = 0; i < 9; i++){
        fileIn >> arr[i];
    }

    // Assert
    EXPECT_EQ(n, 3);
    EXPECT_EQ(m, 3);

    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(arr[i*3+j], A[i][j]);
        }
    }
}

TEST_F(Filter_Methods, intsrteam_operator){
    // Arrange
    Filter<int> M;
    std::ofstream file;
    std::ifstream fileIn;

    // Act
    file.open("FilterTest.txt");
    file << A;
    file.close();
    fileIn.open("FilterTest.txt");
    EXPECT_NO_THROW(fileIn >> M);

    // Assert
    EXPECT_EQ(M.getN(), 3);
    EXPECT_EQ(M.getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(M[i][j], A[i][j]);
        }
    }
}