#include <gtest/gtest.h>
#include "Neyron.h"
#include <fstream>

using namespace NN;

class Neyron_Methods : public ::testing::Test {
public:
    Neyron_Methods() { /* init protected members here */ }

    ~Neyron_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        A = Neyron<int>(3, 3, 5);
        B = Neyron<double>(3, 3, 6);
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
    Neyron<int> A;
    Neyron<double> B;
};

TEST(Neyron_Constructor, By_default_Test) {
    // Arrange

    // Act
    Neyron<int> m;

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
    EXPECT_EQ(m.GetD(), 0);
    EXPECT_EQ(m.GetWBias(), 0);
}

TEST(Neyron_Constructor, Initial_first_square_Test) {
    // Arrange
    int **arr = new int *[100];
    for (size_t i = 0; i < 100; i++) {
        arr[i] = new int[100];
        for (size_t j = 0; j < 100; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Neyron<int> m(arr, 100, 100, 5);

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

TEST(Neyron_Constructor, Initial_first_not_square_one_Test) {
    // Arrange
    int **arr = new int *[100];
    for (size_t i = 0; i < 100; i++) {
        arr[i] = new int[50];
        for (size_t j = 0; j < 50; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Neyron<int> m(arr, 100, 50, 3);

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

TEST(Neyron_Constructor, Initial_first_not_square_two_Test) {
    // Arrange
    int **arr = new int *[50];
    for (size_t i = 0; i < 50; i++) {
        arr[i] = new int[100];
        for (size_t j = 0; j < 100; j++) {
            arr[i][j] = i + j;
        }
    }

    // Act
    Neyron<int> m(arr, 50, 100, 8);

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

TEST(Neyron_Constructor, Initial_first_wrong_negative_size_Test) {
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
    EXPECT_ANY_THROW(Neyron<int> m(arr, -2, -2, 8));
}

TEST(Neyron_Constructor, Initial_second_square_Test) {
    // Arrange
    int *arr = new int[10000];
    for (size_t i = 0; i < 10000; i++) {
        arr[i] = 3;
    }

    // Act
    Neyron<int> m(arr, 100, 100);

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

TEST(Neyron_Constructor, Initial_second_not_square_one_Test) {
    // Arrange
    int *arr = new int[5000];
    for (size_t i = 0; i < 5000; i++) {
        arr[i] = 3;
    }

    // Act
    Neyron<int> m(arr, 100, 50, 5);

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

TEST(Neyron_Constructor, Initial_second_not_square_two_Test) {
    // Arrange
    int *arr = new int[5000];
    for (size_t i = 0; i < 5000; i++) {
        arr[i] = 3;
    }

    // Act
    Neyron<int> m(arr, 50, 100, 19);

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

TEST(Neyron_Constructor, Initial_second_wrong_negative_size_Test) {
    // Arrange
    int *arr = new int[10000];
    for (size_t i = 0; i < 10000; i++) {
        arr[i] = i;
    }

    // Act

    // Assert
    EXPECT_ANY_THROW(Neyron<int> m(arr, -2, -2));
}

TEST(Neyron_Constructor, Initial_third_square_Test) {
    // Arrange

    // Act
    Neyron<int> m(100, 100, 55);

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

TEST(Neyron_Constructor, Initial_third_not_square_one_Test) {
    // Arrange

    // Act
    Neyron<int> m(50, 100);

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

TEST(Neyron_Constructor, Initial_third_not_square_two_Test) {
    // Arrange

    // Act
    Neyron<int> m(100, 50);

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

TEST(Neyron_Constructor, Initial_third_wrong_negative_size_Test) {
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(Neyron<int> m(-2, -2));
}

TEST(Neyron_Constructor, Copy_Test) {
    // Arrange
    Neyron<int> t(100, 100, 9);
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            t[i][j] = i + j;
        }
    }

    // Act
    Neyron<int> m(t);

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

TEST_F(Neyron_Methods, outsrteam_operator){
    // Arrange
    std::ofstream file;
    std::ifstream fileIn;

    int n, m, d, w;
    int arr[9];

    // Act
    file.open("NeyronTest.txt");
    EXPECT_NO_THROW(file << A);
    file.close();
    fileIn.open("NeyronTest.txt");
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

TEST_F(Neyron_Methods, intsrteam_operator){
    // Arrange
    Neyron<int> M;
    std::ofstream file;
    std::ifstream fileIn;

    // Act
    file.open("NeyronTest.txt");
    file << A;
    file.close();
    fileIn.open("NeyronTest.txt");
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

TEST_F(Neyron_Methods, assignment_operator_zero_size_Test){
    // Arrange
    Neyron<int> D(0, 0, 3);

    // Act
    EXPECT_NO_THROW(D = A);


    // Assert
    EXPECT_EQ(D.getN(), 3);
    EXPECT_EQ(D.getM(), 3);
    EXPECT_EQ(D.GetD(), 10);
    EXPECT_EQ(D.GetWBias(), 5);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[i][j], i);
        }
    }
}

TEST_F(Neyron_Methods, assignment_operator_smaller_size_Test){
    // Arrange
    Neyron<int> D(2, 2, 5);
    D.Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A.getN(), 2);
    EXPECT_EQ(A.getM(), 2);
    EXPECT_EQ(A.GetD(), 0);
    EXPECT_EQ(A.GetWBias(), 5);
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            EXPECT_EQ(A[i][j], 5);
        }
    }
}

TEST_F(Neyron_Methods, assignment_operator_bigger_size_Test){
    // Arrange
    Neyron<int> D(4,4, 8);
    D.Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A.getN(), 4);
    EXPECT_EQ(A.getM(), 4);
    EXPECT_EQ(A.GetD(), 0);
    EXPECT_EQ(A.GetWBias(), 8);
    for(size_t i = 0; i < 4; i++){
        for(size_t j = 0; j < 4; j++){
            EXPECT_EQ(A[i][j], 5);
        }
    }
}

TEST_F(Neyron_Methods, Summator_with_wbias_Test){
    // Arrange
    Matrix<int> a(3,3);
    a.Fill(1);
    int summ;

    // Act
    EXPECT_NO_THROW(summ = A.Summator(a));


    // Assert
    EXPECT_EQ(summ, 14);
}

TEST_F(Neyron_Methods, Summator_without_wbias_Test){
    // Arrange
    Matrix<int> a(3,3);
    Neyron<int> n(3,3,0);
    n.Fill(1);
    a.Fill(1);
    int summ;

    // Act
    EXPECT_NO_THROW(summ = n.Summator(a));


    // Assert
    EXPECT_EQ(summ, 9);
}

TEST_F(Neyron_Methods, Summator_wrong_smaller_size_Test){
    // Arrange
    Matrix<int> a(2,2);
    a.Fill(1);
    int summ;

    // Act

    // Assert
    EXPECT_ANY_THROW(A.Summator(a));
}

TEST_F(Neyron_Methods, Summator_wrong_bigger_size_Test){
    // Arrange
    Matrix<int> a(4,4);
    a.Fill(1);
    int summ;

    // Act

    // Assert
    EXPECT_ANY_THROW(A.Summator(a));
}

TEST_F(Neyron_Methods, Summator_null_size_Test){
    // Arrange
    Matrix<int> a(0,0);
    Neyron<int> b(0,0);
    int summ;

    // Act

    // Assert
    EXPECT_ANY_THROW(A.Summator(a));
}

TEST_F(Neyron_Methods, FunkActiv_10_Test){
    // Arrange
    int summ = 10;
    int result;
    Relu<int> f(1);

    // Act
    result = Neyron<int>::FunkActiv(summ, f);

    // Assert
    EXPECT_EQ(result, 10);
}

TEST_F(Neyron_Methods, FunkActiv_1_Test){
    // Arrange
    int summ = 1;
    int result;
    Relu<int> f(1);

    // Act
    result = Neyron<int>::FunkActiv(summ, f);

    // Assert
    EXPECT_EQ(result, 1);
}

TEST_F(Neyron_Methods, FunkActiv_0_Test){
    // Arrange
    int summ = 0;
    int result;
    Relu<int> f(1);

    // Act
    result = Neyron<int>::FunkActiv(summ, f);

    // Assert
    EXPECT_EQ(result, 0);
}

TEST_F(Neyron_Methods, FunkActiv_negotiate_Test){
    // Arrange
    int summ = -1;
    int result;
    Relu<int> f(1);

    // Act
    result = Neyron<int>::FunkActiv(summ, f);

    // Assert
    EXPECT_EQ(result, 0);
}