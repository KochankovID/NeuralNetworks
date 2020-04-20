#include <gtest/gtest.h>
#include "NN/CNN/Filter/Filter.h"
#include <fstream>

using namespace NN;

#define MAT_TEST(X,Y) for(size_t iii = 0; iii < X.getN(); iii++){ for(size_t jjj = 0; jjj < X.getM(); jjj++){ EXPECT_EQ(X[iii][jjj], Y); }}


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
                A[0][i][j] = i;
                B[0][i][j] = j;
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
    EXPECT_EQ(m.getN(), 1);
    EXPECT_EQ(m.getM(), 1);
    EXPECT_EQ(m.getHeight(), 0);
    EXPECT_EQ(m.getWidth(), 0);
    EXPECT_EQ(m.getDepth(), 0);
}

TEST(Filter_Constructor, Initial_third_square_Test){
    // Arrange

    // Act
    Filter<int> m(100, 100);

    // Assert
    EXPECT_EQ(m[0].getN(), 100);
    EXPECT_EQ(m[0].getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], 0);
        }
    }
}

TEST(Filter_Constructor, Initial_third_not_square_one_Test){
    // Arrange

    // Act
    Filter<int> m(50, 100);

    // Assert
    EXPECT_EQ(m[0].getN(), 50);
    EXPECT_EQ(m[0].getM(), 100);
    for(size_t i = 0; i < 50; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], 0);
        }
    }
}

TEST(Filter_Constructor, Initial_third_not_square_two_Test){
    // Arrange

    // Act
    Filter<int> m(100, 50);

    // Assert
    EXPECT_EQ(m[0].getN(), 100);
    EXPECT_EQ(m[0].getM(), 50);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 50; j++){
            EXPECT_EQ(m[0][i][j], 0);
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
            t[0][i][j] = i+j;
        }
    }

    // Act
    Filter<int> m(t);

    // Assert
    EXPECT_EQ(m[0].getN(), 100);
    EXPECT_EQ(m[0].getM(), 100);
    for(size_t i = 0; i < 100; i++){
        for(size_t j = 0; j < 100; j++){
            EXPECT_EQ(m[0][i][j], i+j);
        }
    }
}

TEST_F(Filter_Methods, roate_180_Test){
    // Arrange

    // Act
    EXPECT_NO_THROW(A = A.roate_180());
    EXPECT_NO_THROW(B = B.roate_180());
    // Assert
    EXPECT_EQ(A[0].getN(), 3);
    EXPECT_EQ(A[0].getM(), 3);
    EXPECT_EQ(B[0].getN(), 3);
    EXPECT_EQ(B[0].getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(A[0][i][j], 2-i);
            EXPECT_EQ(B[0][i][j], 2-j);
        }
    }
}

TEST_F(Filter_Methods, roate_180_null_size_Test){
    // Arrange
    Filter<int> T(0,0);

    // Act
    EXPECT_NO_THROW(T = T.roate_180());
    // Assert
    EXPECT_EQ(T[0].getN(), 0);
    EXPECT_EQ(T[0].getM(), 0);
}

TEST_F(Filter_Methods, assignment_operator_bigger_size_Test){
    // Arrange
    Filter<int> D(4,4);
    D[0].Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A[0].getN(), 4);
    EXPECT_EQ(A[0].getM(), 4);
    for(size_t i = 0; i < 4; i++){
        for(size_t j = 0; j < 4; j++){
            EXPECT_EQ(A[0][i][j], 5);
        }
    }
}

TEST_F(Filter_Methods, assignment_operator_smaller_size_Test){
    // Arrange
    Filter<int> D(2, 2);
    D[0].Fill(5);

    // Act
    EXPECT_NO_THROW(A = D);


    // Assert
    EXPECT_EQ(A[0].getN(), 2);
    EXPECT_EQ(A[0].getM(), 2);
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            EXPECT_EQ(A[0][i][j], 5);
        }
    }
}

TEST_F(Filter_Methods, assignment_operator_zero_size_Test){
    // Arrange
    Filter<int> D(0, 0);

    // Act
    EXPECT_NO_THROW(D = A);


    // Assert
    EXPECT_EQ(D[0].getN(), 3);
    EXPECT_EQ(D[0].getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(D[0][i][j], i);
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
            EXPECT_EQ(arr[i*3+j], A[0][i][j]);
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
    EXPECT_EQ(M[0].getN(), 3);
    EXPECT_EQ(M[0].getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(M[0][i][j], A[0][i][j]);
        }
    }
}

TEST_F(Filter_Methods, padding_oneline_Test){
    // Arrange
    Matrix<int> M(1,1);

    // Act
    M[0][0] = 1;
    EXPECT_NO_THROW(M = Filter<int>::Padding(M,1));

    // Assert
    EXPECT_EQ(M.getN(), 3);
    EXPECT_EQ(M.getM(), 3);
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            if((i == 1) &&(j == 1)){
                EXPECT_EQ(M[i][j], 1);
            }else {
                EXPECT_EQ(M[i][j], 0);
            }
        }
    }
}

TEST_F(Filter_Methods, padding_twolines_Test){
    // Arrange
    Matrix<int> M(1,1);

    // Act
    M[0][0] = 1;
    EXPECT_NO_THROW(M =Filter<int>::Padding(M,2));

    // Assert
    EXPECT_EQ(M.getN(), 5);
    EXPECT_EQ(M.getM(), 5);
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            if((i == 2) &&(j == 2)){
                EXPECT_EQ(M[i][j], 1);
            }else {
                EXPECT_EQ(M[i][j], 0);
            }
        }
    }
}

TEST_F(Filter_Methods, padding_zero_size_Test){
    // Arrange
    Matrix<int> M(0,0);

    // Act
    EXPECT_NO_THROW(M =Filter<int>::Padding(M,2));

    // Assert
    EXPECT_EQ(M.getN(), 4);
    EXPECT_EQ(M.getM(), 4);
    for(size_t i = 0; i < 4; i++){
        for(size_t j = 0; j < 4; j++){
            EXPECT_EQ(M[i][j], 0);
        }
    }
}

TEST_F(Filter_Methods, maxpooling_3x3_Test){
    // Arrange
    Matrix<int> M(3,3);
    Matrix<int> max;

    // Act
    M.Fill(2);
    M[0][0] = 3;
    EXPECT_NO_THROW(max = Filter<int>::Pooling(M,3,3));

    // Assert
    EXPECT_EQ(max.getN(), 1);
    EXPECT_EQ(max.getM(), 1);
    EXPECT_EQ(max[0][0], 3);
}

TEST_F(Filter_Methods, maxpooling_2x2_Test){
    // Arrange
    Matrix<int> M(3,3);
    Matrix<int> max;

    // Act
    M.Fill(2);
    M[0][0] = 3;
    EXPECT_NO_THROW(max = Filter<int>::Pooling(M,2,2));

    // Assert
    EXPECT_EQ(max.getN(), 1);
    EXPECT_EQ(max.getM(), 1);

    EXPECT_EQ(max[0][0], 3);

}

TEST_F(Filter_Methods, maxpooling_2x2_4x4_Test){
    // Arrange
    Matrix<int> M(4,4);
    Matrix<int> max;

    // Act
    M.Fill(2);
    M[0][0] = 3;
    EXPECT_NO_THROW(max = Filter<int>::Pooling(M,2,2));

    // Assert
    EXPECT_EQ(max.getN(), 2);
    EXPECT_EQ(max.getM(), 2);

    EXPECT_EQ(max[0][0], 3);
    EXPECT_EQ(max[0][1], 2);
    EXPECT_EQ(max[1][0], 2);
    EXPECT_EQ(max[1][1], 2);

}

TEST_F(Filter_Methods, maxpooling_1x1_Test){
    // Arrange
    Matrix<int> M(2,2);
    Matrix<int> max;

    // Act
    M.Fill(2);
    EXPECT_NO_THROW(max = Filter<int>::Pooling(M,1,1));

    // Assert
    EXPECT_EQ(max.getN(), 2);
    EXPECT_EQ(max.getM(), 2);

    EXPECT_EQ(max[0][0], 2);
    EXPECT_EQ(max[0][1], 2);
    EXPECT_EQ(max[1][0], 2);
    EXPECT_EQ(max[1][1], 2);

}

TEST_F(Filter_Methods, maxpooling_wrong_size_of_kernel_negative_Test){
    // Arrange
    Matrix<int> M(2,2);

    // Act

    // Assert
    EXPECT_ANY_THROW(Filter<int>::Pooling(M, 0, 0));

}

TEST_F(Filter_Methods, maxpooling_wrong_size_of_kernel_biiger_Test){
    // Arrange
    Matrix<int> M(2,2);

    // Act

    // Assert
    EXPECT_ANY_THROW(Filter<int>::Pooling(M, 3, 3));

}

TEST_F(Filter_Methods, Svertka_with_step_2_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int> T(1, 1, 1);
    Matrix<int> out;

    // Act
    T[0] = Matrix<int>(1,1);
    T[0][0][0] = 1;
    U[0].Fill(1);
    T[0] = Filter<int>::Padding(T[0],1);
    EXPECT_NO_THROW(out = U.Svertka(T, 2));

    // Assert
    EXPECT_EQ(out.getN(), 1);
    EXPECT_EQ(out.getM(), 1);
    EXPECT_EQ(out[0][0], 1);
}

TEST_F(Filter_Methods, Svertka_with_step_1_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int> T(1, 1, 1);
    Matrix<int> out;

    // Act
    T[0] = Matrix<int>(2,2);
    U[0].Fill(1);
    T[0] = Filter<int>::Padding(T[0],1);
    T[0].Fill(2);
    EXPECT_NO_THROW(out = U.Svertka(T, 1));

    // Assert
    EXPECT_EQ(out.getN(), 3);
    EXPECT_EQ(out.getM(), 3);
    EXPECT_EQ(out[0][0], 8);
    EXPECT_EQ(out[0][1], 8);
    EXPECT_EQ(out[0][2], 8);
    EXPECT_EQ(out[1][0], 8);
    EXPECT_EQ(out[1][1], 8);
    EXPECT_EQ(out[1][2], 8);
    EXPECT_EQ(out[2][0], 8);
    EXPECT_EQ(out[2][1], 8);
    EXPECT_EQ(out[2][2], 8);
}

TEST_F(Filter_Methods, Svertka_with_step_1_dif_values_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int> T(1, 1, 1);
    Matrix<int> out;

    // Act
    U[0].Fill(1);
    T[0] = Matrix<int>(1,1);
    T[0][0][0] = 1;
    T[0] = Filter<int>::Padding(T[0],1);
    T[0][0][0] = 1;
    EXPECT_NO_THROW(out = U.Svertka(T, 1));

    // Assert
    EXPECT_EQ(out.getN(), 2);
    EXPECT_EQ(out.getM(), 2);
    EXPECT_EQ(out[0][0], 2);
    EXPECT_EQ(out[0][1], 1);
    EXPECT_EQ(out[1][0], 1);
    EXPECT_EQ(out[1][1], 1);
}

TEST_F(Filter_Methods, Svertka_wrong_step_small_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int> T(1, 1, 1);
    Matrix<int> out;

    // Act

    // Assert
    EXPECT_ANY_THROW(U.Svertka(T, 0));
}

TEST_F(Filter_Methods, Svertka_wrong_step_negative_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int> T(1, 1, 1);
    Matrix<int> out;

    // Act

    // Assert
    EXPECT_ANY_THROW(U.Svertka(T, -1));
}

TEST_F(Filter_Methods, Svertka_wrong_step_bigger_Test){
    // Arrange
    Filter<int> U(2,2);
    Tensor<int>  T(1, 1, 1);
    Matrix<int> out;

    // Act

    // Assert
    EXPECT_ANY_THROW(U.Svertka(T, 2));
}

TEST_F(Filter_Methods, Svertka_3x3_Test){
    // Arrange
    Filter<int> U(2,2, 3);
    Tensor<int>  T(3, 3, 3);
    Matrix<int> out;

    // Act
    T.Fill(1);
    U.Fill(1);
    EXPECT_NO_THROW(out = U.Svertka(T, 1));

    // Assert
    EXPECT_EQ(out.getN(), 2);
    EXPECT_EQ(out.getM(), 2);
    MAT_TEST(out, 12);
}

TEST_F(Filter_Methods, Svertka_3x3_with_dif_val_Test){
    // Arrange
    Filter<int> U(2,2, 3);
    Tensor<int>  T(3, 3, 3);
    Matrix<int> out;

    // Act
    T.Fill(1);
    U.Fill(1);
    T[0][0][0] = 100;
    EXPECT_NO_THROW(out = U.Svertka(T, 1));

    // Assert
    EXPECT_EQ(out.getN(), 2);
    EXPECT_EQ(out.getM(), 2);
    EXPECT_EQ(out[0][0], 111);
}

TEST_F(Filter_Methods, Svertka_3x3_static_Test){
    // Arrange
    Tensor<int> U(2,2, 3);
    Tensor<int>  T(3, 3, 3);
    Matrix<int> out;

    // Act
    T.Fill(1);
    U.Fill(1);
    EXPECT_NO_THROW(out = Filter<int>::Svertka(T,U, 1));

    // Assert
    EXPECT_EQ(out.getN(), 2);
    EXPECT_EQ(out.getM(), 2);
    MAT_TEST(out, 12);
}

TEST_F(Filter_Methods, Svertka_3x3_with_dif_val_static_Test){
    // Arrange
    Tensor<int> U(2,2, 3);
    Tensor<int>  T(3, 3, 3);
    Matrix<int> out;

    // Act
    T.Fill(1);
    U.Fill(1);
    T[0][0][0] = 100;
    EXPECT_NO_THROW(out = Filter<int>::Svertka(T,U, 1));

    // Assert
    EXPECT_EQ(out.getN(), 2);
    EXPECT_EQ(out.getM(), 2);
    EXPECT_EQ(out[0][0], 111);
}