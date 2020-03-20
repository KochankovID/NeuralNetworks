#include "opencv2/ts.hpp"
#include <fstream>
#include <iostream>
#include "Matrix.h"
#include <functional>

using namespace ANN;

class Matrix_Methods : public ::testing::Test {
public:
    Matrix_Methods() { /* init protected members here */ }

    ~Matrix_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Matrix<int>(3,3);
        B = Matrix<double>(3,3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i][j] = i;
                B[i][j] = j;
            }
        }
    }

    void TearDown() { /* called after every test */ }
public:
    Matrix<int> A;
    Matrix<double> B;
};

TEST(Matrix_Constructor, By_default_Test){
    // Arrange

    // Act
    Matrix<int> m;

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
}

TEST(Matrix_Constructor, Initial_first_square_Test){
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

TEST(Matrix_Constructor, Initial_first_not_square_one_Test){
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

TEST(Matrix_Constructor, Initial_first_not_square_two_Test){
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

TEST(Matrix_Constructor, Initial_first_wrong_negative_size_Test){
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

TEST(Matrix_Constructor, Initial_second_square_Test){
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

TEST(Matrix_Constructor, Initial_second_not_square_one_Test){
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

TEST(Matrix_Constructor, Initial_second_not_square_two_Test){
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

TEST(Matrix_Constructor, Initial_second_wrong_negative_size_Test){
    // Arrange
    int* arr = new int[10000];
    for(size_t i = 0; i < 10000; i++){
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Matrix<int> m(arr, -2, -2));
}

TEST(Matrix_Constructor, Initial_third_square_Test){
    // Arrange

    // Act
    Matrix<int> m(100, 100);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Matrix_Constructor, Initial_third_not_square_one_Test){
    // Arrange

    // Act
    Matrix<int> m(50, 100);

    // Assert
    EXPECT_EQ(m.getN(), 50);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Matrix_Constructor, Initial_third_not_square_two_Test){
    // Arrange

    // Act
    Matrix<int> m(100, 50);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[i][j], 0);
        }
    }
}

TEST(Matrix_Constructor, Initial_third_wrong_negative_size_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Matrix<int> m(-2, -2));
}

TEST(Matrix_Constructor, Copy_Test){
    // Arrange
    Matrix<int> t(100,100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            t[i][j] = i+j;
        }
    }

    // Act
    Matrix<int> m(t);

    // Assert
    EXPECT_EQ(m.getN(), 100);
    EXPECT_EQ(m.getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[i][j], i+j);
        }
    }
}

TEST_F(Matrix_Methods, GetN_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_EQ(A.getN(), 3);
    EXPECT_EQ(B.getN(), 3);
}

TEST_F(Matrix_Methods, GetM_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_EQ(A.getM(), 3);
    EXPECT_EQ(B.getM(), 3);
}

TEST_F(Matrix_Methods, Max_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_EQ(A.Max(), 2);
    EXPECT_EQ(B.Max(), 2);
}

TEST_F(Matrix_Methods, getCopy_Test){
    // Arrange
    int** arr;

    // Act
    arr = A.getCopy();

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(arr[i][j], A[i][j]);
        }
    }
}

TEST_F(Matrix_Methods, Fill_Test){
    // Arrange

    // Act
    A.Fill(10);

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(A[i][j], 10);
        }
    }
}

TEST_F(Matrix_Methods, Fill_zero_size_Test){
    // Arrange
    Matrix<int> a;

    // Act

    // Assert
    EXPECT_NO_THROW(a.Fill(10));
}

TEST_F(Matrix_Methods, getPodmatrix_Test){
    // Arrange
    Matrix<int> D;
    // Act
    D = A.getPodmatrix(0,0,2,2);

    // Assert
    EXPECT_EQ(D.getN(), 2);
    EXPECT_EQ(D.getM(), 2);
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            EXPECT_EQ(D[i][j], i);
        }
    }
}

TEST_F(Matrix_Methods, getPodmatrix_wrong_first_position_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(A.getPodmatrix(-1, 1, 1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(1, -1, 1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(-1, -1, 1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(3, 1, 1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(1, 3, 1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(3, 3, 1, 1));
}

TEST_F(Matrix_Methods, getPodmatrix_wrong_size_podmatrix_Test){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, -1, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, 1, -1));
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, -1, -1));
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, 4, 1));
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, 1, 4));
    EXPECT_ANY_THROW(A.getPodmatrix(0, 0, 4, 4));
}

TEST_F(Matrix_Methods, getPodmatrix_null_size_podmatrix_Test){
    // Arrange
    Matrix<int> D0, D1, D2;

    // Act
    EXPECT_NO_THROW(D0 = A.getPodmatrix(0, 0, 0, 1));
    EXPECT_NO_THROW(D1 = A.getPodmatrix(0, 0, 1, 0));
    EXPECT_NO_THROW(D2 = A.getPodmatrix(0, 0, 0, 0));

    // Assert
    EXPECT_EQ(D0.getN(), 0);
    EXPECT_EQ(D1.getN(), 0);
    EXPECT_EQ(D2.getN(), 0);

    EXPECT_EQ(D0.getM(), 0);
    EXPECT_EQ(D1.getM(), 0);
    EXPECT_EQ(D2.getM(), 0);
}

TEST_F(Matrix_Methods, assignment_operator_bigger_size_Test){
    // Arrange
    Matrix<int> D(4,4);
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

TEST_F(Matrix_Methods, assignment_operator_smaller_size_Test){
    // Arrange
    Matrix<int> D(2, 2);
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

TEST_F(Matrix_Methods, assignment_operator_zero_size_Test){
    // Arrange
    Matrix<int> D(0, 0);

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

TEST_F(Matrix_Methods, summ_operator_equal_size_Test){
    // Arrange
    Matrix<int> D(A);

    // Act
    EXPECT_NO_THROW(D = D + A);


    // Assert
    EXPECT_EQ(D.getN(), 3);
    EXPECT_EQ(D.getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[i][j], i+i);
        }
    }
}

TEST_F(Matrix_Methods, summ_operator_wrong_not_equal_size_Test){
    // Arrange
    Matrix<int> D(2,2);

    // Act


    // Assert
    EXPECT_ANY_THROW(D + A);
}

TEST_F(Matrix_Methods, mul_operator_equal_size){
    // Arrange
    Matrix<int> D(3,3);
    D.Fill(1);

    // Act
    EXPECT_NO_THROW(D = D * A);

    // Assert
    EXPECT_EQ(D.getN(), 3);
    EXPECT_EQ(D.getM(), 3);

    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[i][j], 3);
        }
    }
}

TEST_F(Matrix_Methods, mul_operator_not_equal_size){
    // Arrange
    Matrix<int> D(1,3);
    D.Fill(1);

    // Act
    EXPECT_NO_THROW(D = D * A);

    // Assert
    EXPECT_EQ(D.getN(), 1);
    EXPECT_EQ(D.getM(), 3);

    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[i][j], 3);
        }
    }
}

TEST_F(Matrix_Methods, mul_operator_wrong_size){
    // Arrange
    Matrix<int> D(3,4);
    Matrix<int> G;

    // Act

    // Assert
    EXPECT_ANY_THROW(D * A);
    EXPECT_ANY_THROW(G * A);
}

TEST_F(Matrix_Methods, mul_operator_zero_size){
    // Arrange
    Matrix<int> G;

    // Act
    EXPECT_NO_THROW(G = G * G);

    // Assert
    EXPECT_EQ(G.getN(), 0);
    EXPECT_EQ(G.getM(), 0);
}

TEST_F(Matrix_Methods, mul_operator_matrix_on_constant){
    // Arrange
    double k = 0;

    // Act
    EXPECT_NO_THROW(B = B * k);

    // Assert
    EXPECT_EQ(B.getN(), 3);
    EXPECT_EQ(B.getM(), 3);

    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(B[i][j], 0);
        }
    }
}

TEST_F(Matrix_Methods, mul_operator_constant_on_matrix){
    // Arrange
    double k = 0;

    // Act
    EXPECT_NO_THROW(B = k * B);

    // Assert
    EXPECT_EQ(B.getN(), 3);
    EXPECT_EQ(B.getM(), 3);

    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(B[i][j], 0);
        }
    }
}

TEST_F(Matrix_Methods, outsrteam_operator){
    // Arrange
    std::ofstream file;
    std::ifstream fileIn;

    int n, m;
    int arr[9];

    // Act
    file.open("MatrixTest.txt");
    EXPECT_NO_THROW(file << A);
    file.close();
    fileIn.open("MatrixTest.txt");
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

TEST_F(Matrix_Methods, intsrteam_operator){
    // Arrange
    Matrix<int> M;
    std::ofstream file;
    std::ifstream fileIn;

    // Act
    file.open("MatrixTest.txt");
    file << A;
    file.close();
    fileIn.open("MatrixTest.txt");
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

TEST_F(Matrix_Methods, index_operator) {
    // Arrange

    // Act

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(A[i][j], i);
        }
    }
}

TEST_F(Matrix_Methods, index_operator_bigger_index) {
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(B[10]);
}

TEST_F(Matrix_Methods, index_operator_negative_index) {
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(B[-1]);
}

TEST_F(Matrix_Methods, const_index_operator) {
    // Arrange
    const Matrix<int> F(A);
    // Act

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(F[i][j], i);
        }
    }
}

TEST_F(Matrix_Methods, const_index_operator_bigger_index) {
    // Arrange
    const Matrix<double> F(B);

    // Act

    // Assert
    EXPECT_ANY_THROW(F[10]);
}

TEST_F(Matrix_Methods, const_index_operator_negative_index) {
    // Arrange
    const Matrix<double> F(B);

    // Act

    // Assert
    EXPECT_ANY_THROW(F[-1]);
}

TEST_F(Matrix_Methods, compare_operator) {
    // Arrange
    Matrix<double> F(B);
    Matrix<double> F1(B);

    // Act
    F1[0][2] = 6;

    // Assert
    EXPECT_TRUE(F == B);
    EXPECT_FALSE(F1 == B);

}

TEST_F(Matrix_Methods, compare_operator_different_size) {
    // Arrange
    Matrix<double> F(1, 1);

    // Act

    // Assert
    EXPECT_FALSE(F == B);

}

TEST_F(Matrix_Methods, mean_Test) {
    // Arrange
    double mean_b;
    int mean_a;

    // Act
    EXPECT_NO_THROW(mean_b = B.mean());
    EXPECT_NO_THROW(mean_a = A.mean());

    // Assert
    EXPECT_EQ(mean_b, 1);
    EXPECT_EQ(mean_a, 1);

}

TEST_F(Matrix_Methods, zoom_one_place_Test) {
    // Arrange
    Matrix<int> R(2,2);
    Matrix<int> new_R;

    // Act
    R.Fill(2);
    EXPECT_NO_THROW(new_R = R.zoom(1));

    // Assert
    EXPECT_EQ(new_R.getN(), 3);
    EXPECT_EQ(new_R.getM(), 3);

    EXPECT_EQ(new_R[0][0], 2);
    EXPECT_EQ(new_R[0][2], 2);
    EXPECT_EQ(new_R[2][0], 2);
    EXPECT_EQ(new_R[2][2], 2);
}

TEST_F(Matrix_Methods, zoom_two_place_Test) {
    // Arrange
    Matrix<int> R(2,2);
    Matrix<int> new_R;

    // Act
    R.Fill(2);
    EXPECT_NO_THROW(new_R = R.zoom(2));

    // Assert
    EXPECT_EQ(new_R.getN(), 4);
    EXPECT_EQ(new_R.getM(), 4);

    EXPECT_EQ(new_R[0][0], 2);
    EXPECT_EQ(new_R[0][3], 2);
    EXPECT_EQ(new_R[3][0], 2);
    EXPECT_EQ(new_R[3][3], 2);
}

TEST_F(Matrix_Methods, zoom_wrong_place_zero_Test) {
    // Arrange
    Matrix<int> R(2, 2);

    // Act

    // Assert
    EXPECT_ANY_THROW(R.zoom(0));
}

TEST_F(Matrix_Methods, zoom_wrong_place_negative_Test) {
    // Arrange
    Matrix<int> R(2, 2);

    // Act

    // Assert
    EXPECT_ANY_THROW(R.zoom(-1));
}