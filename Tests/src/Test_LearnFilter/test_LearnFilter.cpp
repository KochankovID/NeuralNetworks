#include <opencv2/ts.hpp>
#include "LearnFilter.h"

using namespace ANN;

#define MAT_TEST(X,Y) for(size_t ii = 0; ii < X.getN(); ii++){ for(size_t jj = 0; jj < X.getM(); jj++){ EXPECT_EQ(X[ii][jj], Y); }}

TEST(LearnFilter_functions, BackPropagation_step_one_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D.Fill(1);
    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(D,F,1));

    // Assert
    EXPECT_EQ(out.getN(), 3);
    EXPECT_EQ(out.getM(), 3);

    EXPECT_EQ(out[0][0], 1);
    EXPECT_EQ(out[0][1], 2);
    EXPECT_EQ(out[0][2], 1);

    EXPECT_EQ(out[1][0], 2);
    EXPECT_EQ(out[1][1], 4);
    EXPECT_EQ(out[1][2], 2);

    EXPECT_EQ(out[2][0], 1);
    EXPECT_EQ(out[2][1], 2);
    EXPECT_EQ(out[2][2], 1);



}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 1;
    D[1][1] = 2;

    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(D,F,1));

    // Assert
    EXPECT_EQ(out.getN(), 3);
    EXPECT_EQ(out.getM(), 3);

    EXPECT_EQ(out[0][0], 1);
    EXPECT_EQ(out[0][1], 3);
    EXPECT_EQ(out[0][2], 2);

    EXPECT_EQ(out[1][0], 2);
    EXPECT_EQ(out[1][1], 6);
    EXPECT_EQ(out[1][2], 4);

    EXPECT_EQ(out[2][0], 1);
    EXPECT_EQ(out[2][1], 3);
    EXPECT_EQ(out[2][2], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D.Fill(1);
    F[0][0] = 1;
    F[0][1] = 2;
    F[1][0] = 3;
    F[1][1] = 4;
    EXPECT_NO_THROW(out = BackPropagation(D,F,1));

    // Assert
    EXPECT_EQ(out.getN(), 3);
    EXPECT_EQ(out.getM(), 3);

    EXPECT_EQ(out[0][0], 1);
    EXPECT_EQ(out[0][1], 3);
    EXPECT_EQ(out[0][2], 2);

    EXPECT_EQ(out[1][0], 4);
    EXPECT_EQ(out[1][1], 10);
    EXPECT_EQ(out[1][2], 6);

    EXPECT_EQ(out[2][0], 3);
    EXPECT_EQ(out[2][1], 7);
    EXPECT_EQ(out[2][2], 4);

}

TEST(LearnFilter_functions, BackPropagation_step_two_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D.Fill(1);
    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(D,F,2));

    // Assert
    MAT_TEST(out, 1);



}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 3;
    D[1][1] = 2;
    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(D,F,2));

    // Assert
    EXPECT_EQ(out[0][0], 1);
    EXPECT_EQ(out[0][1], 1);
    EXPECT_EQ(out[0][2], 2);
    EXPECT_EQ(out[0][3], 2);

    EXPECT_EQ(out[1][0], 1);
    EXPECT_EQ(out[1][1], 1);
    EXPECT_EQ(out[1][2], 2);
    EXPECT_EQ(out[1][3], 2);

    EXPECT_EQ(out[2][0], 3);
    EXPECT_EQ(out[2][1], 3);
    EXPECT_EQ(out[2][2], 2);
    EXPECT_EQ(out[2][3], 2);

    EXPECT_EQ(out[3][0], 3);
    EXPECT_EQ(out[3][1], 3);
    EXPECT_EQ(out[3][2], 2);
    EXPECT_EQ(out[3][3], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    F[0][0] = 1;
    F[0][1] = 2;

    F[1][0] = 3;
    F[1][1] = 2;

    D.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(D,F,2));

    // Assert
    EXPECT_EQ(out[0][0], 1);
    EXPECT_EQ(out[0][1], 2);
    EXPECT_EQ(out[0][2], 1);
    EXPECT_EQ(out[0][3], 2);

    EXPECT_EQ(out[1][0], 3);
    EXPECT_EQ(out[1][1], 2);
    EXPECT_EQ(out[1][2], 3);
    EXPECT_EQ(out[1][3], 2);

    EXPECT_EQ(out[2][0], 1);
    EXPECT_EQ(out[2][1], 2);
    EXPECT_EQ(out[2][2], 1);
    EXPECT_EQ(out[2][3], 2);

    EXPECT_EQ(out[3][0], 3);
    EXPECT_EQ(out[3][1], 2);
    EXPECT_EQ(out[3][2], 3);
    EXPECT_EQ(out[3][3], 2);

}

TEST(LearnFilter_functions, GradDes_step_one_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Matrix<double> M(3,3);
    Filter<double> F(2,2);
    SimpleGrad<double > G(1);

    // Act
    M.Fill(1);
    D.Fill(1);
    F.Fill(1);
    GradDes(G, M, D,F,1);

    // Assert
    EXPECT_EQ(F.getN(), 2);
    EXPECT_EQ(F.getM(), 2);

    MAT_TEST(F, -3);
}

TEST(LearnFilter_functions, GradDes_step_one_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Matrix<double> M(3,3);
    Filter<double> F(2,2);
    SimpleGrad<double > G(1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;
    D[1][0] = 3;
    D[1][1] = 4;
    M.Fill(1);
    F.Fill(1);
    GradDes(G, M, D,F,1);

    // Assert
    EXPECT_EQ(F.getN(), 2);
    EXPECT_EQ(F.getM(), 2);

    MAT_TEST(F, -9);
}

TEST(LearnFilter_functions, GradDes_step_one_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Matrix<double> M(3,3);
    Filter<double> F(2,2);
    SimpleGrad<double > G(1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;
    D[1][0] = 3;
    D[1][1] = 4;
    M.Fill(1);
    F.Fill(1);
    GradDes(G, M, D,F,1);

    // Assert
    EXPECT_EQ(F.getN(), 2);
    EXPECT_EQ(F.getM(), 2);

    MAT_TEST(F, -9);
}