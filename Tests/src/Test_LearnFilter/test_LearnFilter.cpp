#include <opencv2/ts.hpp>
#include "LearnFilter.h"

using namespace ANN;

#define MAT_TEST(X,Y) for(size_t ii = 0; ii < X.getN(); ii++){ for(size_t jj = 0; jj < X.getM(); jj++){ EXPECT_DOUBLE_EQ(X[ii][jj], Y); }}

TEST(LearnFilter_functions, BackPropagation_step_one_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Tensor<double> In(3,3,1);
    Tensor<double> out(3,3,1);

    // Act
    In.Fill(1);
    D.Fill(1);
    F[0].Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D, F, 1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 1);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 2);
    EXPECT_EQ(out[0][0][2], 1);

    EXPECT_EQ(out[0][1][0], 2);
    EXPECT_EQ(out[0][1][1], 4);
    EXPECT_EQ(out[0][1][2], 2);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 2);
    EXPECT_EQ(out[0][2][2], 1);

}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Tensor<double> In(3,3,1);
    Tensor<double> out(3, 3, 1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 1;
    D[1][1] = 2;

    F.Fill(1);
    In.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F, 1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 1);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 3);
    EXPECT_EQ(out[0][0][2], 2);

    EXPECT_EQ(out[0][1][0], 2);
    EXPECT_EQ(out[0][1][1], 6);
    EXPECT_EQ(out[0][1][2], 4);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 3);
    EXPECT_EQ(out[0][2][2], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Tensor<double> out(3, 3, 1);
    Tensor<double> In(3,3,1);

    // Act
    D.Fill(1);
    In.Fill(1);
    F[0][0][0] = 1;
    F[0][0][1] = 2;
    F[0][1][0] = 3;
    F[0][1][1] = 4;
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 1);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 3);
    EXPECT_EQ(out[0][0][2], 2);

    EXPECT_EQ(out[0][1][0], 4);
    EXPECT_EQ(out[0][1][1], 10);
    EXPECT_EQ(out[0][1][2], 6);

    EXPECT_EQ(out[0][2][0], 3);
    EXPECT_EQ(out[0][2][1], 7);
    EXPECT_EQ(out[0][2][2], 4);

}

TEST(LearnFilter_functions, BackPropagation_step_two_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2);
    Tensor<double> out(4,4,1);
    Tensor<double> In(4,4,1);

    // Act
    In.Fill(1);
    D.Fill(1);
    F.Fill(1);
    out = BackPropagation(In, D,F, 2);

    // Assert
    MAT_TEST(out[0], 1);



}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> In(4,4,1);
    Filter<double> F(2,2);
    Tensor<double> out(4,4,1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 3;
    D[1][1] = 2;
    F.Fill(1);
    In.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,2));

    // Assert
    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 1);
    EXPECT_EQ(out[0][0][2], 2);
    EXPECT_EQ(out[0][0][3], 2);

    EXPECT_EQ(out[0][1][0], 1);
    EXPECT_EQ(out[0][1][1], 1);
    EXPECT_EQ(out[0][1][2], 2);
    EXPECT_EQ(out[0][1][3], 2);

    EXPECT_EQ(out[0][2][0], 3);
    EXPECT_EQ(out[0][2][1], 3);
    EXPECT_EQ(out[0][2][2], 2);
    EXPECT_EQ(out[0][2][3], 2);

    EXPECT_EQ(out[0][3][0], 3);
    EXPECT_EQ(out[0][3][1], 3);
    EXPECT_EQ(out[0][3][2], 2);
    EXPECT_EQ(out[0][3][3], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> In(4,4,1);
    Filter<double> F(2,2);
    Tensor<double> out(4,4,1);

    // Act
    F[0][0][0] = 1;
    F[0][0][1] = 2;

    F[0][1][0] = 3;
    F[0][1][1] = 2;

    D.Fill(1);
    In.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,2));

    // Assert
    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 2);
    EXPECT_EQ(out[0][0][2], 1);
    EXPECT_EQ(out[0][0][3], 2);

    EXPECT_EQ(out[0][1][0], 3);
    EXPECT_EQ(out[0][1][1], 2);
    EXPECT_EQ(out[0][1][2], 3);
    EXPECT_EQ(out[0][1][3], 2);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 2);
    EXPECT_EQ(out[0][2][2], 1);
    EXPECT_EQ(out[0][2][3], 2);

    EXPECT_EQ(out[0][3][0], 3);
    EXPECT_EQ(out[0][3][1], 2);
    EXPECT_EQ(out[0][3][2], 3);
    EXPECT_EQ(out[0][3][3], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_one_all_one_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> In(3,3,2);
    Filter<double> F(2,2,2);
    Tensor<double> out(3,3,2);

    // Act
    D.Fill(1);
    F.Fill(1);
    In.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D, F, 1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 2);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 2);
    EXPECT_EQ(out[0][0][2], 1);

    EXPECT_EQ(out[0][1][0], 2);
    EXPECT_EQ(out[0][1][1], 4);
    EXPECT_EQ(out[0][1][2], 2);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 2);
    EXPECT_EQ(out[0][2][2], 1);

    EXPECT_EQ(out[1][0][0], 1);
    EXPECT_EQ(out[1][0][1], 2);
    EXPECT_EQ(out[1][0][2], 1);

    EXPECT_EQ(out[1][1][0], 2);
    EXPECT_EQ(out[1][1][1], 4);
    EXPECT_EQ(out[1][1][2], 2);

    EXPECT_EQ(out[1][2][0], 1);
    EXPECT_EQ(out[1][2][1], 2);
    EXPECT_EQ(out[1][2][2], 1);

}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_outs_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2,2);
    Tensor<double> In(3,3,2);
    Tensor<double> out(3, 3, 2);

    // Act
    In.Fill(1);
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 1;
    D[1][1] = 2;

    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F, 1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 2);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 3);
    EXPECT_EQ(out[0][0][2], 2);

    EXPECT_EQ(out[0][1][0], 2);
    EXPECT_EQ(out[0][1][1], 6);
    EXPECT_EQ(out[0][1][2], 4);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 3);
    EXPECT_EQ(out[0][2][2], 2);

    EXPECT_EQ(out[1][0][0], 1);
    EXPECT_EQ(out[1][0][1], 3);
    EXPECT_EQ(out[1][0][2], 2);

    EXPECT_EQ(out[1][1][0], 2);
    EXPECT_EQ(out[1][1][1], 6);
    EXPECT_EQ(out[1][1][2], 4);

    EXPECT_EQ(out[1][2][0], 1);
    EXPECT_EQ(out[1][2][1], 3);
    EXPECT_EQ(out[1][2][2], 2);
}

TEST(LearnFilter_functions, BackPropagation_step_one_different_values_filter_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2,2);
    Tensor<double> In(3,3,2);
    Tensor<double> out(3, 3, 2);

    // Act
    D.Fill(1);
    In.Fill(1);

    F[0][0][0] = 1;
    F[0][0][1] = 2;
    F[0][1][0] = 3;
    F[0][1][1] = 4;

    F[1][0][0] = 1;
    F[1][0][1] = 2;
    F[1][1][0] = 3;
    F[1][1][1] = 4;
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,1));

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 2);

    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 3);
    EXPECT_EQ(out[0][0][2], 2);

    EXPECT_EQ(out[0][1][0], 4);
    EXPECT_EQ(out[0][1][1], 10);
    EXPECT_EQ(out[0][1][2], 6);

    EXPECT_EQ(out[0][2][0], 3);
    EXPECT_EQ(out[0][2][1], 7);
    EXPECT_EQ(out[0][2][2], 4);

    EXPECT_EQ(out[1][0][0], 1);
    EXPECT_EQ(out[1][0][1], 3);
    EXPECT_EQ(out[1][0][2], 2);

    EXPECT_EQ(out[1][1][0], 4);
    EXPECT_EQ(out[1][1][1], 10);
    EXPECT_EQ(out[1][1][2], 6);

    EXPECT_EQ(out[1][2][0], 3);
    EXPECT_EQ(out[1][2][1], 7);
    EXPECT_EQ(out[1][2][2], 4);
}

TEST(LearnFilter_functions, BackPropagation_step_two_all_one_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2,2);
    Tensor<double> In(4,4,2);
    Tensor<double> out(4,4,2);

    // Act
    D.Fill(1);
    F.Fill(1);
    In.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F, 2));

    // Assert
    MAT_TEST(out[0], 1);
    MAT_TEST(out[1], 1);
}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_outs_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2,2);
    Tensor<double> In(4,4,2);
    Tensor<double> out(4,4,2);

    // Act
    In.Fill(1);
    D[0][0] = 1;
    D[0][1] = 2;

    D[1][0] = 3;
    D[1][1] = 2;
    F.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,2));

    // Assert
    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 1);
    EXPECT_EQ(out[0][0][2], 2);
    EXPECT_EQ(out[0][0][3], 2);

    EXPECT_EQ(out[0][1][0], 1);
    EXPECT_EQ(out[0][1][1], 1);
    EXPECT_EQ(out[0][1][2], 2);
    EXPECT_EQ(out[0][1][3], 2);

    EXPECT_EQ(out[0][2][0], 3);
    EXPECT_EQ(out[0][2][1], 3);
    EXPECT_EQ(out[0][2][2], 2);
    EXPECT_EQ(out[0][2][3], 2);

    EXPECT_EQ(out[0][3][0], 3);
    EXPECT_EQ(out[0][3][1], 3);
    EXPECT_EQ(out[0][3][2], 2);
    EXPECT_EQ(out[0][3][3], 2);

    EXPECT_EQ(out[1][0][0], 1);
    EXPECT_EQ(out[1][0][1], 1);
    EXPECT_EQ(out[1][0][2], 2);
    EXPECT_EQ(out[1][0][3], 2);

    EXPECT_EQ(out[1][1][0], 1);
    EXPECT_EQ(out[1][1][1], 1);
    EXPECT_EQ(out[1][1][2], 2);
    EXPECT_EQ(out[1][1][3], 2);

    EXPECT_EQ(out[1][2][0], 3);
    EXPECT_EQ(out[1][2][1], 3);
    EXPECT_EQ(out[1][2][2], 2);
    EXPECT_EQ(out[1][2][3], 2);

    EXPECT_EQ(out[1][3][0], 3);
    EXPECT_EQ(out[1][3][1], 3);
    EXPECT_EQ(out[1][3][2], 2);
    EXPECT_EQ(out[1][3][3], 2);

}

TEST(LearnFilter_functions, BackPropagation_step_two_different_values_filter_2x2_Test){
    // Arrange
    Matrix<double> D(2,2);
    Filter<double> F(2,2, 2);
    Tensor<double> In(4,4,2);
    Tensor<double> out(4,4,2);

    // Act
    In.Fill(1);
    F[0][0][0] = 1;
    F[0][0][1] = 2;

    F[0][1][0] = 3;
    F[0][1][1] = 2;

    F[1][0][0] = 1;
    F[1][0][1] = 2;

    F[1][1][0] = 3;
    F[1][1][1] = 2;

    D.Fill(1);
    EXPECT_NO_THROW(out = BackPropagation(In, D,F,2));

    // Assert
    EXPECT_EQ(out[0][0][0], 1);
    EXPECT_EQ(out[0][0][1], 2);
    EXPECT_EQ(out[0][0][2], 1);
    EXPECT_EQ(out[0][0][3], 2);

    EXPECT_EQ(out[0][1][0], 3);
    EXPECT_EQ(out[0][1][1], 2);
    EXPECT_EQ(out[0][1][2], 3);
    EXPECT_EQ(out[0][1][3], 2);

    EXPECT_EQ(out[0][2][0], 1);
    EXPECT_EQ(out[0][2][1], 2);
    EXPECT_EQ(out[0][2][2], 1);
    EXPECT_EQ(out[0][2][3], 2);

    EXPECT_EQ(out[0][3][0], 3);
    EXPECT_EQ(out[0][3][1], 2);
    EXPECT_EQ(out[0][3][2], 3);
    EXPECT_EQ(out[0][3][3], 2);

    EXPECT_EQ(out[1][0][0], 1);
    EXPECT_EQ(out[1][0][1], 2);
    EXPECT_EQ(out[1][0][2], 1);
    EXPECT_EQ(out[1][0][3], 2);

    EXPECT_EQ(out[1][1][0], 3);
    EXPECT_EQ(out[1][1][1], 2);
    EXPECT_EQ(out[1][1][2], 3);
    EXPECT_EQ(out[1][1][3], 2);

    EXPECT_EQ(out[1][2][0], 1);
    EXPECT_EQ(out[1][2][1], 2);
    EXPECT_EQ(out[1][2][2], 1);
    EXPECT_EQ(out[1][2][3], 2);

    EXPECT_EQ(out[1][3][0], 3);
    EXPECT_EQ(out[1][3][1], 2);
    EXPECT_EQ(out[1][3][2], 3);
    EXPECT_EQ(out[1][3][3], 2);

}

TEST(LearnFilter_functions, GradDes_step_one_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3,3);
    Filter<double> F(2,2,3);
    SGD<double > G(1);

    // Act
    M.Fill(1);
    D.Fill(1);
    F.Fill(1);
    BackPropagation(M,D,F,1);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 3);

    MAT_TEST(F[0], -3);
    MAT_TEST(F[1], -3);
    MAT_TEST(F[2], -3);
}

TEST(LearnFilter_functions, GradDes_step_one_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3, 3);
    Filter<double> F(2,2, 3);
    SGD<double > G(1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;
    D[1][0] = 3;
    D[1][1] = 4;

    M.Fill(1);
    F.Fill(1);
    BackPropagation(M,D,F,1);
    GradDes(G,F);

    // Assertv
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 3);

    MAT_TEST(F[0], -9);
    MAT_TEST(F[1], -9);
    MAT_TEST(F[2], -9);
}

TEST(LearnFilter_functions, GradDes_step_one_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3, 1);
    Filter<double> F(2,2, 1);
    SGD<double > G(1);

    // Act
    D.Fill(1);
    M.Fill(1);
    F[0][0][0] = 1;
    F[0][0][1] = 2;
    F[0][1][0] = 3;
    F[0][1][1] = 4;
    BackPropagation(M,D,F,1);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 1);

    EXPECT_EQ(F[0][0][0], -3);
    EXPECT_EQ(F[0][0][1], -2);
    EXPECT_EQ(F[0][1][0], -1);
    EXPECT_EQ(F[0][1][1], 0);
}

TEST(LearnFilter_functions, GradDes_step_one_different_values_input_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3,1);
    Filter<double> F(2,2, 1);
    SGD<double > G(1);

    // Act
    D.Fill(1);
    F.Fill(1);

    M[0][0][0] = 1;
    M[0][0][1] = 2;
    M[0][0][2] = 3;

    M[0][1][0] = 4;
    M[0][1][1] = 5;
    M[0][1][2] = 6;

    M[0][2][0] = 7;
    M[0][2][1] = 8;
    M[0][2][2] = 9;

    BackPropagation(M,D,F,1);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 1);

    EXPECT_EQ(F[0][0][0], -11);
    EXPECT_EQ(F[0][0][1], -15);
    EXPECT_EQ(F[0][1][0], -23);
    EXPECT_EQ(F[0][1][1], -27);
}

TEST(LearnFilter_functions, GradDes_step_two_all_one_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(4,4,2);
    Filter<double> F(2,2,2);
    SGD<double > G(1);

    // Act
    M.Fill(1);
    D.Fill(1);
    F.Fill(1);
    BackPropagation(M,D,F,2);
    GradDes(G, F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 2);

    MAT_TEST(F[0], -3);
    MAT_TEST(F[1], -3);
}

TEST(LearnFilter_functions, BackProp_wrong_size_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3,6);
    Filter<double> F(2,2,6);
    SGD<double > G(1);

    // Act
    M.Fill(1);
    D.Fill(1);
    F.Fill(1);

    // Assert
    EXPECT_ANY_THROW(BackPropagation(M,D,F,2));
}

TEST(LearnFilter_functions, GradDes_step_two_different_values_outs_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(4,4, 4);
    Filter<double> F(2,2, 4);
    SGD<double > G(1);

    // Act
    D[0][0] = 1;
    D[0][1] = 2;
    D[1][0] = 3;
    D[1][1] = 4;
    M.Fill(1);
    F.Fill(1);
    BackPropagation(M,D,F,2);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 4);

    MAT_TEST(F[0], -9);
    MAT_TEST(F[1], -9);
    MAT_TEST(F[2], -9);
    MAT_TEST(F[3], -9);
}

TEST(LearnFilter_functions, GradDes_step_two_different_values_filter_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(4,4,1);
    Filter<double> F(2,2,1);
    SGD<double > G(1);

    // Act
    D.Fill(1);
    M.Fill(1);
    F[0][0][0] = 1;
    F[0][0][1] = 2;
    F[0][1][0] = 3;
    F[0][1][1] = 4;
    BackPropagation(M,D,F,2);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 1);

    EXPECT_EQ(F[0][0][0], -3);
    EXPECT_EQ(F[0][0][1], -2);
    EXPECT_EQ(F[0][1][0], -1);
    EXPECT_EQ(F[0][1][1], 0);
}

TEST(LearnFilter_functions, GradDes_step_two_different_values_input_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(4,4,1);
    Filter<double> F(2,2,1);
    SGD<double > G(1);

    // Act
    D.Fill(1);
    F.Fill(1);

    M[0][0][0] = 1;
    M[0][0][1] = 2;
    M[0][0][2] = 3;
    M[0][0][3] = 4;

    M[0][1][0] = 5;
    M[0][1][1] = 6;
    M[0][1][2] = 7;
    M[0][1][3] = 8;

    M[0][2][0] = 9;
    M[0][2][1] = 10;
    M[0][2][2] = 11;
    M[0][2][3] = 12;

    M[0][3][0] = 13;
    M[0][3][1] = 14;
    M[0][3][2] = 15;
    M[0][3][3] = 16;
    BackPropagation(M,D,F,2);
    GradDes(G,F);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 1);

    EXPECT_EQ(F[0][0][0], -23);
    EXPECT_EQ(F[0][0][1], -27);
    EXPECT_EQ(F[0][1][0], -39);
    EXPECT_EQ(F[0][1][1], -43);
}

TEST(LearnFilter_functions, GradDes_history_Test){
    // Arrange
    Matrix<double> D(2,2);
    Tensor<double> M(3,3,3);
    Filter<double> F(2,2,3);
    Tensor<double> H(2,2,3);
    SGD_Momentum<double > G(1, 0.9);

    // Act
    M.Fill(1);
    D.Fill(1);
    F.Fill(1);
    H.Fill(1);
    BackPropagation(M,D,F,1);
    GradDes(G, F, H);

    // Assert
    EXPECT_EQ(F.getHeight(), 2);
    EXPECT_EQ(F.getWidth(), 2);
    EXPECT_EQ(F.getDepth(), 3);

    MAT_TEST(F[0], -0.3);
    MAT_TEST(F[1], -0.3);
    MAT_TEST(F[2], -0.3);
}

TEST(LearnFilter_functions, BackPropagation_pooling_Test){
    // Arrange
    Tensor<double> D(2,2,1);
    Tensor<double> M(4,4,1);
    Tensor<double> OUT(2,2,1);
    Tensor<double> o(4,4,1);

    // Act
    D.Fill(5);

    OUT[0][0][0] = 4;
    OUT[0][0][1] = 5;
    OUT[0][1][0] = 6;
    OUT[0][1][1] = 7;

    M[0][0][0] = 1;
    M[0][0][1] = 2;
    M[0][0][2] = 1;
    M[0][0][3] = 2;

    M[0][1][0] = 3;
    M[0][1][1] = 4;
    M[0][1][2] = 3;
    M[0][1][3] = 5;

    M[0][2][0] = 1;
    M[0][2][1] = 2;
    M[0][2][2] = 1;
    M[0][2][3] = 2;

    M[0][3][0] = 3;
    M[0][3][1] = 6;
    M[0][3][2] = 3;
    M[0][3][3] = 7;

    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));

    // Assert
    EXPECT_EQ(o.getHeight(), 4);
    EXPECT_EQ(o.getWidth(), 4);
    EXPECT_EQ(o.getDepth(), 1);

    EXPECT_EQ(o[0][0][0], 0);
    EXPECT_EQ(o[0][0][1], 0);
    EXPECT_EQ(o[0][0][2], 0);
    EXPECT_EQ(o[0][0][3], 0);

    EXPECT_EQ(o[0][1][0], 0);
    EXPECT_EQ(o[0][1][1], 5);
    EXPECT_EQ(o[0][1][2], 0);
    EXPECT_EQ(o[0][1][3], 5);

    EXPECT_EQ(o[0][2][0], 0);
    EXPECT_EQ(o[0][2][1], 0);
    EXPECT_EQ(o[0][2][2], 0);
    EXPECT_EQ(o[0][2][3], 0);

    EXPECT_EQ(o[0][3][0], 0);
    EXPECT_EQ(o[0][3][1], 5);
    EXPECT_EQ(o[0][3][2], 0);
    EXPECT_EQ(o[0][3][3], 5);
}

TEST(LearnFilter_functions, BackPropagation_pooling_different_values_Test){
    // Arrange
    Tensor<double> D(2,2,1);
    Tensor<double> M(4,4,1);
    Tensor<double> OUT(2,2,1);
    Tensor<double> o(4,4,1);

    // Act
    D.Fill(5);

    OUT[0][0][0] = 4;
    OUT[0][0][1] = 5;
    OUT[0][1][0] = 6;
    OUT[0][1][1] = 7;

    M[0][0][0] = 1;
    M[0][0][1] = 4;
    M[0][0][2] = 5;
    M[0][0][3] = 2;

    M[0][1][0] = 3;
    M[0][1][1] = 2;
    M[0][1][2] = 3;
    M[0][1][3] = 2;

    M[0][2][0] = 1;
    M[0][2][1] = 2;
    M[0][2][2] = 7;
    M[0][2][3] = 2;

    M[0][3][0] = 6;
    M[0][3][1] = 3;
    M[0][3][2] = 3;
    M[0][3][3] = 1;

    EXPECT_NO_THROW(o = BackPropagation<double>(M,OUT, D, 2, 2));

    // Assert
    EXPECT_EQ(o.getHeight(), 4);
    EXPECT_EQ(o.getWidth(), 4);
    EXPECT_EQ(o.getDepth(), 1);

    EXPECT_EQ(o[0][0][0], 0);
    EXPECT_EQ(o[0][0][1], 5);
    EXPECT_EQ(o[0][0][2], 5);
    EXPECT_EQ(o[0][0][3], 0);

    EXPECT_EQ(o[0][1][0], 0);
    EXPECT_EQ(o[0][1][1], 0);
    EXPECT_EQ(o[0][1][2], 0);
    EXPECT_EQ(o[0][1][3], 0);

    EXPECT_EQ(o[0][2][0], 0);
    EXPECT_EQ(o[0][2][1], 0);
    EXPECT_EQ(o[0][2][2], 5);
    EXPECT_EQ(o[0][2][3], 0);

    EXPECT_EQ(o[0][3][0], 5);
    EXPECT_EQ(o[0][3][1], 0);
    EXPECT_EQ(o[0][3][2], 0);
    EXPECT_EQ(o[0][3][3], 0);
}

TEST(LearnFilter_functions, BackPropagation_pooling_equal_Test){
    // Arrange
    Tensor<double> D(2,2,2);
    Tensor<double> M(4,4,2);
    Tensor<double> OUT(2,2,2);
    Tensor<double> o(4,4,2);

    // Act
    D.Fill(5);

    OUT[0][0][0] = 4;
    OUT[0][0][1] = 5;
    OUT[0][1][0] = 6;
    OUT[0][1][1] = 7;

    OUT[1][0][0] = 4;
    OUT[1][0][1] = 5;
    OUT[1][1][0] = 6;
    OUT[1][1][1] = 7;

    M[0][0][0] = 1;
    M[0][0][1] = 4;
    M[0][0][2] = 5;
    M[0][0][3] = 2;

    M[0][1][0] = 3;
    M[0][1][1] = 4;
    M[0][1][2] = 3;
    M[0][1][3] = 5;

    M[0][2][0] = 1;
    M[0][2][1] = 2;
    M[0][2][2] = 7;
    M[0][2][3] = 7;

    M[0][3][0] = 6;
    M[0][3][1] = 6;
    M[0][3][2] = 7;
    M[0][3][3] = 7;

    M[1][0][0] = 1;
    M[1][0][1] = 4;
    M[1][0][2] = 5;
    M[1][0][3] = 2;

    M[1][1][0] = 3;
    M[1][1][1] = 4;
    M[1][1][2] = 3;
    M[1][1][3] = 5;

    M[1][2][0] = 1;
    M[1][2][1] = 2;
    M[1][2][2] = 7;
    M[1][2][3] = 7;

    M[1][3][0] = 6;
    M[1][3][1] = 6;
    M[1][3][2] = 7;
    M[1][3][3] = 7;

    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));

    // Assert
    EXPECT_EQ(o.getHeight(), 4);
    EXPECT_EQ(o.getWidth(), 4);
    EXPECT_EQ(o.getDepth(), 2);

    EXPECT_EQ(o[0][0][0], 0);
    EXPECT_EQ(o[0][0][1], 5);
    EXPECT_EQ(o[0][0][2], 5);
    EXPECT_EQ(o[0][0][3], 0);

    EXPECT_EQ(o[0][1][0], 0);
    EXPECT_EQ(o[0][1][1], 0);
    EXPECT_EQ(o[0][1][2], 0);
    EXPECT_EQ(o[0][1][3], 0);

    EXPECT_EQ(o[0][2][0], 0);
    EXPECT_EQ(o[0][2][1], 0);
    EXPECT_EQ(o[0][2][2], 5);
    EXPECT_EQ(o[0][2][3], 0);

    EXPECT_EQ(o[0][3][0], 5);
    EXPECT_EQ(o[0][3][1], 0);
    EXPECT_EQ(o[0][3][2], 0);
    EXPECT_EQ(o[0][3][3], 0);

    EXPECT_EQ(o[1][0][0], 0);
    EXPECT_EQ(o[1][0][1], 5);
    EXPECT_EQ(o[1][0][2], 5);
    EXPECT_EQ(o[1][0][3], 0);

    EXPECT_EQ(o[1][1][0], 0);
    EXPECT_EQ(o[1][1][1], 0);
    EXPECT_EQ(o[1][1][2], 0);
    EXPECT_EQ(o[1][1][3], 0);

    EXPECT_EQ(o[1][2][0], 0);
    EXPECT_EQ(o[1][2][1], 0);
    EXPECT_EQ(o[1][2][2], 5);
    EXPECT_EQ(o[1][2][3], 0);

    EXPECT_EQ(o[1][3][0], 5);
    EXPECT_EQ(o[1][3][1], 0);
    EXPECT_EQ(o[1][3][2], 0);
    EXPECT_EQ(o[1][3][3], 0);
}

TEST(LearnFilter_functions, BackPropagation_matrix_Tests){
    // Arrange
    Tensor<double> D(2, 2, 10);
    Tensor<double> In(3, 3, 1);
    Matrix<Filter<double> > F(1, 10);
    Tensor<double> out(3,3,1);

    // Act
    D.Fill(1);
    In.Fill(1);
    for(size_t i = 0; i < F.getN(); i++){
        for(size_t j = 0; j < F.getM(); j++){
            F[i][j] = Filter<double>(2,2);
            F[i][j].Fill(1);
        }
    }
    out = BackPropagation(In, D,F,1);

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 1);

    EXPECT_EQ(out[0][0][0], 10);
    EXPECT_EQ(out[0][0][1], 20);
    EXPECT_EQ(out[0][0][2], 10);

    EXPECT_EQ(out[0][1][0], 20);
    EXPECT_EQ(out[0][1][1], 40);
    EXPECT_EQ(out[0][1][2], 20);

    EXPECT_EQ(out[0][2][0], 10);
    EXPECT_EQ(out[0][2][1], 20);
    EXPECT_EQ(out[0][2][2], 10);
}

TEST(LearnFilter_functions, BackPropagation_matrix_depth_3_Tests){
    // Arrange
    Tensor<double> D(2, 2, 20);
    Matrix<Filter<double> > F(1, 20);
    Tensor<double> In(3, 3, 3);
    Tensor<double> out(3, 3, 3);

    // Act
    D.Fill(1);
    In.Fill(1);
    for(size_t i = 0; i < F.getN(); i++){
        for(size_t j = 0; j < F.getM(); j++){
                F[i][j] = Filter<double>(2, 2, 3);
                F[i][j].Fill(1);
        }
    }
    out = BackPropagation(In, D,F,1);

    // Assert
    EXPECT_EQ(out.getHeight(), 3);
    EXPECT_EQ(out.getWidth(), 3);
    EXPECT_EQ(out.getDepth(), 3);

    for(size_t i = 0; i < 3; i++) {
        EXPECT_EQ(out[i].getN(), 3);
        EXPECT_EQ(out[i].getM(), 3);

        EXPECT_EQ(out[i][0][0], 20);
        EXPECT_EQ(out[i][0][1], 40);
        EXPECT_EQ(out[i][0][2], 20);

        EXPECT_EQ(out[i][1][0], 40);
        EXPECT_EQ(out[i][1][1], 80);
        EXPECT_EQ(out[i][1][2], 40);

        EXPECT_EQ(out[i][2][0], 20);
        EXPECT_EQ(out[i][2][1], 40);
        EXPECT_EQ(out[i][2][2], 20);
    }
}

TEST(LearnFilter_functions, GradDes_matrix_Test){
    // Arrange
    Tensor<double> D(2,2,10);
    Tensor<double> M(3,3,1);
    Matrix<Filter<double> > F(1,10);
    SGD<double > G(1);

    // Act
    M.Fill(1);
    D.Fill(1);
    for(size_t i = 0; i < F.getM(); i++){
        F[0][i] = Filter<double>(2, 2);
        F[0][i].Fill(1);
    }
    BackPropagation(M,D,F,1);
    EXPECT_NO_THROW(GradDes(G, F));

    // Assert
    for(size_t i = 0; i < F.getN(); i++) {
        for (size_t j = 0; j < F.getM(); j++) {

            EXPECT_EQ(F[i][j].getHeight(), 2);
            EXPECT_EQ(F[i][j].getWidth(), 2);
            EXPECT_EQ(F[i][j].getDepth(), 1);

            MAT_TEST(F[i][j][0], -3);
        }
    }
}

TEST(LearnFilter_functions, GradDes_matrix_two_matix_Test){
    // Arrange
    Tensor<double> D(2,2,20);
    Tensor<double> M(3,3,2);
    Matrix<Filter<double> > F(1,20);
    SGD<double > G(1);

    // Act
    D.Fill(1);
    M.Fill(1);

    for(size_t i = 0; i < F.getM(); i++){
        F[0][i] = Filter<double>(2,2,2);
        F[0][i].Fill(1);
    }
    BackPropagation(M,D,F,1);
    EXPECT_NO_THROW(GradDes(G, F));

    // Assert
    for (size_t j = 0; j < F.getM(); j++) {

        EXPECT_EQ(F[0][j].getHeight(), 2);
        EXPECT_EQ(F[0][j].getWidth(), 2);
        EXPECT_EQ(F[0][j].getDepth(), 2);

        MAT_TEST(F[0][j][0], -3);
        MAT_TEST(F[0][j][1], -3);
    }
}

TEST(LearnFilter_functions, BackPropagation_pooling_matrix_Test){
    // Arrange
    Tensor<double> D(2,2,10);
    Tensor<double> M(4,4,10);
    Tensor<double> OUT(2, 2,10);
    Tensor<double> o(4,4,10);

    // Act
    D.Fill(5);

    for(size_t j = 0; j < M.getDepth(); j++){

        M[j][0][0] = 1;
        M[j][0][1] = 2;
        M[j][0][2] = 1;
        M[j][0][3] = 2;

        M[j][1][0] = 3;
        M[j][1][1] = 4;
        M[j][1][2] = 3;
        M[j][1][3] = 5;

        M[j][2][0] = 1;
        M[j][2][1] = 2;
        M[j][2][2] = 1;
        M[j][2][3] = 2;

        M[j][3][0] = 3;
        M[j][3][1] = 6;
        M[j][3][2] = 3;
        M[j][3][3] = 7;

        OUT[j][0][0] = 4;
        OUT[j][0][1] = 5;
        OUT[j][1][0] = 6;
        OUT[j][1][1] = 7;
    }

    EXPECT_NO_THROW( o = BackPropagation(M, OUT, D, 2, 2));

    // Assert
    for(size_t i = 0; i < o.getDepth(); i++) {
            EXPECT_EQ(o[i].getN(), 4);
            EXPECT_EQ(o[i].getM(), 4);

            EXPECT_EQ(o[i][0][0], 0);
            EXPECT_EQ(o[i][0][1], 0);
            EXPECT_EQ(o[i][0][2], 0);
            EXPECT_EQ(o[i][0][3], 0);

            EXPECT_EQ(o[i][1][0], 0);
            EXPECT_EQ(o[i][1][1], 5);
            EXPECT_EQ(o[i][1][2], 0);
            EXPECT_EQ(o[i][1][3], 5);

            EXPECT_EQ(o[i][2][0], 0);
            EXPECT_EQ(o[i][2][1], 0);
            EXPECT_EQ(o[i][2][2], 0);
            EXPECT_EQ(o[i][2][3], 0);

            EXPECT_EQ(o[i][3][0], 0);
            EXPECT_EQ(o[i][3][1], 5);
            EXPECT_EQ(o[i][3][2], 0);
            EXPECT_EQ(o[i][3][3], 5);
    }
}