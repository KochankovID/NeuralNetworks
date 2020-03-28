#include <opencv2/ts.hpp>
#include "LearnFilter.h"

using namespace ANN;

#define MAT_TEST(X,Y) for(size_t ii = 0; ii < X.getN(); ii++){ for(size_t jj = 0; jj < X.getM(); jj++){ EXPECT_DOUBLE_EQ(X[ii][jj], Y); }}

TEST(LearnFilter_functions, BackPropagation_step_one_all_one_Test){
    // Arrange
    Tensor<double> D(2,2,1);
    Filter<double> F(2,2);
    Matrix<double> out;

    // Act
    D.Fill(1);
    F[0].Fill(1);
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

//TEST(LearnFilter_functions, BackPropagation_step_one_different_values_outs_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Filter<double> F(2,2);
//    Matrix<double> out;
//
//    // Act
//    D[0][0] = 1;
//    D[0][1] = 2;
//
//    D[1][0] = 1;
//    D[1][1] = 2;
//
//    F.Fill(1);
//    EXPECT_NO_THROW(out = BackPropagation(D,F,1));
//
//    // Assert
//    EXPECT_EQ(out.getN(), 3);
//    EXPECT_EQ(out.getM(), 3);
//
//    EXPECT_EQ(out[0][0], 1);
//    EXPECT_EQ(out[0][1], 3);
//    EXPECT_EQ(out[0][2], 2);
//
//    EXPECT_EQ(out[1][0], 2);
//    EXPECT_EQ(out[1][1], 6);
//    EXPECT_EQ(out[1][2], 4);
//
//    EXPECT_EQ(out[2][0], 1);
//    EXPECT_EQ(out[2][1], 3);
//    EXPECT_EQ(out[2][2], 2);
//
//}
//
//TEST(LearnFilter_functions, BackPropagation_step_one_different_values_filter_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Filter<double> F(2,2);
//    Matrix<double> out;
//
//    // Act
//    D.Fill(1);
//    F[0][0] = 1;
//    F[0][1] = 2;
//    F[1][0] = 3;
//    F[1][1] = 4;
//    EXPECT_NO_THROW(out = BackPropagation(D,F,1));
//
//    // Assert
//    EXPECT_EQ(out.getN(), 3);
//    EXPECT_EQ(out.getM(), 3);
//
//    EXPECT_EQ(out[0][0], 1);
//    EXPECT_EQ(out[0][1], 3);
//    EXPECT_EQ(out[0][2], 2);
//
//    EXPECT_EQ(out[1][0], 4);
//    EXPECT_EQ(out[1][1], 10);
//    EXPECT_EQ(out[1][2], 6);
//
//    EXPECT_EQ(out[2][0], 3);
//    EXPECT_EQ(out[2][1], 7);
//    EXPECT_EQ(out[2][2], 4);
//
//}
//
//TEST(LearnFilter_functions, BackPropagation_step_two_all_one_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Filter<double> F(2,2);
//    Matrix<double> out;
//
//    // Act
//    D.Fill(1);
//    F.Fill(1);
//    EXPECT_NO_THROW(out = BackPropagation(D,F,2));
//
//    // Assert
//    MAT_TEST(out, 1);
//
//
//
//}
//
//TEST(LearnFilter_functions, BackPropagation_step_two_different_values_outs_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Filter<double> F(2,2);
//    Matrix<double> out;
//
//    // Act
//    D[0][0] = 1;
//    D[0][1] = 2;
//
//    D[1][0] = 3;
//    D[1][1] = 2;
//    F.Fill(1);
//    EXPECT_NO_THROW(out = BackPropagation(D,F,2));
//
//    // Assert
//    EXPECT_EQ(out[0][0], 1);
//    EXPECT_EQ(out[0][1], 1);
//    EXPECT_EQ(out[0][2], 2);
//    EXPECT_EQ(out[0][3], 2);
//
//    EXPECT_EQ(out[1][0], 1);
//    EXPECT_EQ(out[1][1], 1);
//    EXPECT_EQ(out[1][2], 2);
//    EXPECT_EQ(out[1][3], 2);
//
//    EXPECT_EQ(out[2][0], 3);
//    EXPECT_EQ(out[2][1], 3);
//    EXPECT_EQ(out[2][2], 2);
//    EXPECT_EQ(out[2][3], 2);
//
//    EXPECT_EQ(out[3][0], 3);
//    EXPECT_EQ(out[3][1], 3);
//    EXPECT_EQ(out[3][2], 2);
//    EXPECT_EQ(out[3][3], 2);
//
//}
//
//TEST(LearnFilter_functions, BackPropagation_step_two_different_values_filter_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Filter<double> F(2,2);
//    Matrix<double> out;
//
//    // Act
//    F[0][0] = 1;
//    F[0][1] = 2;
//
//    F[1][0] = 3;
//    F[1][1] = 2;
//
//    D.Fill(1);
//    EXPECT_NO_THROW(out = BackPropagation(D,F,2));
//
//    // Assert
//    EXPECT_EQ(out[0][0], 1);
//    EXPECT_EQ(out[0][1], 2);
//    EXPECT_EQ(out[0][2], 1);
//    EXPECT_EQ(out[0][3], 2);
//
//    EXPECT_EQ(out[1][0], 3);
//    EXPECT_EQ(out[1][1], 2);
//    EXPECT_EQ(out[1][2], 3);
//    EXPECT_EQ(out[1][3], 2);
//
//    EXPECT_EQ(out[2][0], 1);
//    EXPECT_EQ(out[2][1], 2);
//    EXPECT_EQ(out[2][2], 1);
//    EXPECT_EQ(out[2][3], 2);
//
//    EXPECT_EQ(out[3][0], 3);
//    EXPECT_EQ(out[3][1], 2);
//    EXPECT_EQ(out[3][2], 3);
//    EXPECT_EQ(out[3][3], 2);
//
//}
//
//TEST(LearnFilter_functions, GradDes_step_one_all_one_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    M.Fill(1);
//    D.Fill(1);
//    F.Fill(1);
//    GradDes(G, M, D,F,1);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    MAT_TEST(F, -3);
//}
//
//TEST(LearnFilter_functions, GradDes_step_one_different_values_outs_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D[0][0] = 1;
//    D[0][1] = 2;
//    D[1][0] = 3;
//    D[1][1] = 4;
//    M.Fill(1);
//    F.Fill(1);
//    GradDes(G, M, D,F,1);
//
//    // Assertv
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    MAT_TEST(F, -9);
//}
//
//TEST(LearnFilter_functions, GradDes_step_one_different_values_filter_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D.Fill(1);
//    M.Fill(1);
//    F[0][0] = 1;
//    F[0][1] = 2;
//    F[1][0] = 3;
//    F[1][1] = 4;
//    GradDes(G, M, D,F,1);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    EXPECT_EQ(F[0][0], -3);
//    EXPECT_EQ(F[0][1], -2);
//    EXPECT_EQ(F[1][0], -1);
//    EXPECT_EQ(F[1][1], 0);
//}
//
//TEST(LearnFilter_functions, GradDes_step_one_different_values_input_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D.Fill(1);
//    F.Fill(1);
//
//    M[0][0] = 1;
//    M[0][1] = 2;
//    M[0][2] = 3;
//
//    M[1][0] = 4;
//    M[1][1] = 5;
//    M[1][2] = 6;
//
//    M[2][0] = 7;
//    M[2][1] = 8;
//    M[2][2] = 9;
//
//    GradDes(G, M, D,F,1);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    EXPECT_EQ(F[0][0], -11);
//    EXPECT_EQ(F[0][1], -15);
//    EXPECT_EQ(F[1][0], -23);
//    EXPECT_EQ(F[1][1], -27);
//}
//
//TEST(LearnFilter_functions, GradDes_step_two_all_one_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    M.Fill(1);
//    D.Fill(1);
//    F.Fill(1);
//    GradDes(G, M, D,F,2);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    MAT_TEST(F, -3);
//}
//
//TEST(LearnFilter_functions, GradDes_wrong_size_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    M.Fill(1);
//    D.Fill(1);
//    F.Fill(1);
//
//    // Assert
//    EXPECT_ANY_THROW(GradDes(G, M, D,F,2));
//}
//
//TEST(LearnFilter_functions, GradDes_step_two_different_values_outs_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D[0][0] = 1;
//    D[0][1] = 2;
//    D[1][0] = 3;
//    D[1][1] = 4;
//    M.Fill(1);
//    F.Fill(1);
//    GradDes(G, M, D,F,2);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    MAT_TEST(F, -9);
//}
//
//TEST(LearnFilter_functions, GradDes_step_two_different_values_filter_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D.Fill(1);
//    M.Fill(1);
//    F[0][0] = 1;
//    F[0][1] = 2;
//    F[1][0] = 3;
//    F[1][1] = 4;
//    GradDes(G, M, D,F,2);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    EXPECT_EQ(F[0][0], -3);
//    EXPECT_EQ(F[0][1], -2);
//    EXPECT_EQ(F[1][0], -1);
//    EXPECT_EQ(F[1][1], 0);
//}
//
//TEST(LearnFilter_functions, GradDes_step_two_different_values_input_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Filter<double> F(2,2);
//    SGD<double > G(1);
//
//    // Act
//    D.Fill(1);
//    F.Fill(1);
//
//    M[0][0] = 1;
//    M[0][1] = 2;
//    M[0][2] = 3;
//    M[0][3] = 4;
//
//    M[1][0] = 5;
//    M[1][1] = 6;
//    M[1][2] = 7;
//    M[1][3] = 8;
//
//    M[2][0] = 9;
//    M[2][1] = 10;
//    M[2][2] = 11;
//    M[2][3] = 12;
//
//    M[3][0] = 13;
//    M[3][1] = 14;
//    M[3][2] = 15;
//    M[3][3] = 16;
//
//    GradDes(G, M, D,F,2);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    EXPECT_EQ(F[0][0], -23);
//    EXPECT_EQ(F[0][1], -27);
//    EXPECT_EQ(F[1][0], -39);
//    EXPECT_EQ(F[1][1], -43);
//}
//
//TEST(LearnFilter_functions, GradDes_history_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(3,3);
//    Filter<double> F(2,2);
//    Filter<double> H(2,2);
//    SGD_Momentum<double > G(1, 0.9);
//
//    // Act
//    M.Fill(1);
//    D.Fill(1);
//    F.Fill(1);
//    H.Fill(1);
//    GradDes(G, M, D,F,1, H);
//
//    // Assert
//    EXPECT_EQ(F.getN(), 2);
//    EXPECT_EQ(F.getM(), 2);
//
//    MAT_TEST(F, -0.3);
//}
//
//TEST(LearnFilter_functions, BackPropagation_pooling_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Matrix<double> OUT(2,2);
//    Matrix<double> o;
//
//    // Act
//    D.Fill(5);
//
//    OUT[0][0] = 4;
//    OUT[0][1] = 5;
//    OUT[1][0] = 6;
//    OUT[1][1] = 7;
//
//    M[0][0] = 1;
//    M[0][1] = 2;
//    M[0][2] = 1;
//    M[0][3] = 2;
//
//    M[1][0] = 3;
//    M[1][1] = 4;
//    M[1][2] = 3;
//    M[1][3] = 5;
//
//    M[2][0] = 1;
//    M[2][1] = 2;
//    M[2][2] = 1;
//    M[2][3] = 2;
//
//    M[3][0] = 3;
//    M[3][1] = 6;
//    M[3][2] = 3;
//    M[3][3] = 7;
//
//    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));
//
//    // Assert
//    EXPECT_EQ(o.getN(), 4);
//    EXPECT_EQ(o.getM(), 4);
//
//    EXPECT_EQ(o[0][0], 0);
//    EXPECT_EQ(o[0][1], 0);
//    EXPECT_EQ(o[0][2], 0);
//    EXPECT_EQ(o[0][3], 0);
//
//    EXPECT_EQ(o[1][0], 0);
//    EXPECT_EQ(o[1][1], 5);
//    EXPECT_EQ(o[1][2], 0);
//    EXPECT_EQ(o[1][3], 5);
//
//    EXPECT_EQ(o[2][0], 0);
//    EXPECT_EQ(o[2][1], 0);
//    EXPECT_EQ(o[2][2], 0);
//    EXPECT_EQ(o[2][3], 0);
//
//    EXPECT_EQ(o[3][0], 0);
//    EXPECT_EQ(o[3][1], 5);
//    EXPECT_EQ(o[3][2], 0);
//    EXPECT_EQ(o[3][3], 5);
//}
//
//TEST(LearnFilter_functions, BackPropagation_pooling_different_values_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Matrix<double> OUT(2,2);
//    Matrix<double> o;
//
//    // Act
//    D.Fill(5);
//
//    OUT[0][0] = 4;
//    OUT[0][1] = 5;
//    OUT[1][0] = 6;
//    OUT[1][1] = 7;
//
//    M[0][0] = 1;
//    M[0][1] = 4;
//    M[0][2] = 5;
//    M[0][3] = 2;
//
//    M[1][0] = 3;
//    M[1][1] = 2;
//    M[1][2] = 3;
//    M[1][3] = 2;
//
//    M[2][0] = 1;
//    M[2][1] = 2;
//    M[2][2] = 7;
//    M[2][3] = 2;
//
//    M[3][0] = 6;
//    M[3][1] = 3;
//    M[3][2] = 3;
//    M[3][3] = 1;
//
//    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));
//
//    // Assert
//    EXPECT_EQ(o.getN(), 4);
//    EXPECT_EQ(o.getM(), 4);
//
//    EXPECT_EQ(o[0][0], 0);
//    EXPECT_EQ(o[0][1], 5);
//    EXPECT_EQ(o[0][2], 5);
//    EXPECT_EQ(o[0][3], 0);
//
//    EXPECT_EQ(o[1][0], 0);
//    EXPECT_EQ(o[1][1], 0);
//    EXPECT_EQ(o[1][2], 0);
//    EXPECT_EQ(o[1][3], 0);
//
//    EXPECT_EQ(o[2][0], 0);
//    EXPECT_EQ(o[2][1], 0);
//    EXPECT_EQ(o[2][2], 5);
//    EXPECT_EQ(o[2][3], 0);
//
//    EXPECT_EQ(o[3][0], 5);
//    EXPECT_EQ(o[3][1], 0);
//    EXPECT_EQ(o[3][2], 0);
//    EXPECT_EQ(o[3][3], 0);
//}
//
//TEST(LearnFilter_functions, BackPropagation_pooling_equal_Test){
//    // Arrange
//    Matrix<double> D(2,2);
//    Matrix<double> M(4,4);
//    Matrix<double> OUT(2,2);
//    Matrix<double> o;
//
//    // Act
//    D.Fill(5);
//
//    OUT[0][0] = 4;
//    OUT[0][1] = 5;
//    OUT[1][0] = 6;
//    OUT[1][1] = 7;
//
//    M[0][0] = 1;
//    M[0][1] = 4;
//    M[0][2] = 5;
//    M[0][3] = 2;
//
//    M[1][0] = 3;
//    M[1][1] = 4;
//    M[1][2] = 3;
//    M[1][3] = 5;
//
//    M[2][0] = 1;
//    M[2][1] = 2;
//    M[2][2] = 7;
//    M[2][3] = 7;
//
//    M[3][0] = 6;
//    M[3][1] = 6;
//    M[3][2] = 7;
//    M[3][3] = 7;
//
//    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));
//
//    // Assert
//    EXPECT_EQ(o.getN(), 4);
//    EXPECT_EQ(o.getM(), 4);
//
//    EXPECT_EQ(o[0][0], 0);
//    EXPECT_EQ(o[0][1], 5);
//    EXPECT_EQ(o[0][2], 5);
//    EXPECT_EQ(o[0][3], 0);
//
//    EXPECT_EQ(o[1][0], 0);
//    EXPECT_EQ(o[1][1], 0);
//    EXPECT_EQ(o[1][2], 0);
//    EXPECT_EQ(o[1][3], 0);
//
//    EXPECT_EQ(o[2][0], 0);
//    EXPECT_EQ(o[2][1], 0);
//    EXPECT_EQ(o[2][2], 5);
//    EXPECT_EQ(o[2][3], 0);
//
//    EXPECT_EQ(o[3][0], 5);
//    EXPECT_EQ(o[3][1], 0);
//    EXPECT_EQ(o[3][2], 0);
//    EXPECT_EQ(o[3][3], 0);
//}
//
//TEST(LearnFilter_functions, BackPropagation_matrix_Tests){
//    // Arrange
//    Matrix<Matrix<double> > D(1, 10);
//    Matrix<Filter<double> > F(1, 10);
//    Matrix<Matrix<double> > out(1, 10);
//
//    // Act
//    for(size_t i = 0; i < D.getN(); i++){
//        for(size_t j = 0; j < D.getM(); j++){
//            D[i][j] = Matrix<double>(2,2);
//            D[i][j].Fill(1);
//            F[i][j] = Filter<double>(2,2);
//            F[i][j].Fill(1);
//        }
//    }
//    out = BackPropagation(D,F,1);
//
//    // Assert
//    EXPECT_EQ(out.getN(), 1);
//    EXPECT_EQ(out.getM(), 1);
//
//    for(size_t i = 0; i < out.getN(); i++){
//        for(size_t j = 0; j < out.getM(); j++){
//
//            EXPECT_EQ(out[i][j].getN(), 3);
//            EXPECT_EQ(out[i][j].getM(), 3);
//
//            EXPECT_EQ(out[i][j][0][0], 10);
//            EXPECT_EQ(out[i][j][0][1], 20);
//            EXPECT_EQ(out[i][j][0][2], 10);
//
//            EXPECT_EQ(out[i][j][1][0], 20);
//            EXPECT_EQ(out[i][j][1][1], 40);
//            EXPECT_EQ(out[i][j][1][2], 20);
//
//            EXPECT_EQ(out[i][j][2][0], 10);
//            EXPECT_EQ(out[i][j][2][1], 20);
//            EXPECT_EQ(out[i][j][2][2], 10);
//        }
//    }
//}
//
//TEST(LearnFilter_functions, BackPropagation_matrix_two_matrix_Tests){
//    // Arrange
//    Matrix<Matrix<double> > D(1, 20);
//    Matrix<Filter<double> > F(1, 10);
//    Matrix<Matrix<double> > out(2, 10);
//
//    // Act
//    for(size_t i = 0; i < D.getN(); i++){
//        for(size_t j = 0; j < D.getM(); j++){
//            D[i][j] = Matrix<double>(2,2);
//            D[i][j].Fill(1);
//            if(j < 10) {
//                F[i][j] = Filter<double>(2, 2);
//                F[i][j].Fill(1);
//            }
//        }
//    }
//    out = BackPropagation(D,F,1);
//
//    // Assert
//    EXPECT_EQ(out.getN(), 1);
//    EXPECT_EQ(out.getM(), 2);
//
//    for(size_t i = 0; i < out.getN(); i++){
//        for(size_t j = 0; j < out.getM(); j++){
//
//            EXPECT_EQ(out[i][j].getN(), 3);
//            EXPECT_EQ(out[i][j].getM(), 3);
//
//            EXPECT_EQ(out[i][j][0][0], 10);
//            EXPECT_EQ(out[i][j][0][1], 20);
//            EXPECT_EQ(out[i][j][0][2], 10);
//
//            EXPECT_EQ(out[i][j][1][0], 20);
//            EXPECT_EQ(out[i][j][1][1], 40);
//            EXPECT_EQ(out[i][j][1][2], 20);
//
//            EXPECT_EQ(out[i][j][2][0], 10);
//            EXPECT_EQ(out[i][j][2][1], 20);
//            EXPECT_EQ(out[i][j][2][2], 10);
//        }
//    }
//}
//
//TEST(LearnFilter_functions, GradDes_matrix_Test){
//    // Arrange
//    Matrix<Matrix<double> > D(1,10);
//    Matrix<Matrix<double> > M(1,1);
//    Matrix<Filter<double> > F(1,10);
//    SGD<double > G(1);
//
//    // Act
//    M[0][0] = Matrix<double>(3,3);
//    M[0][0].Fill(1);
//    for(size_t i = 0; i < D.getN(); i++){
//        for(size_t j = 0; j < D.getM(); j++){
//            D[i][j] = Matrix<double>(2,2);
//            D[i][j].Fill(1);
//
//            F[i][j] = Filter<double>(2, 2);
//            F[i][j].Fill(1);
//        }
//    }
//    EXPECT_NO_THROW(GradDes(G, M, D,F,1));
//
//    // Assert
//    for(size_t i = 0; i < F.getN(); i++) {
//        for (size_t j = 0; j < F.getM(); j++) {
//
//            EXPECT_EQ(F[i][j].getN(), 2);
//            EXPECT_EQ(F[i][j].getM(), 2);
//
//            MAT_TEST(F[i][j], -3);
//        }
//    }
//}
//
//TEST(LearnFilter_functions, GradDes_matrix_two_matix_Test){
//    // Arrange
//    Matrix<Matrix<double> > D(1,20);
//    Matrix<Matrix<double> > M(1,2);
//    Matrix<Filter<double> > F(1,10);
//    SGD<double > G(1);
//
//    // Act
//    for(size_t i = 0; i < D.getN(); i++){
//        for(size_t j = 0; j < D.getM(); j++){
//            D[i][j] = Matrix<double>(2,2);
//            D[i][j].Fill(1);
//            if(j < 10) {
//                if( j < 2) {
//                    M[i][j] = Matrix<double>(3, 3);
//                    M[i][j].Fill(1);
//                }
//
//                F[i][j] = Filter<double>(2, 2);
//                F[i][j].Fill(1);
//            }
//        }
//    }
//    EXPECT_NO_THROW(GradDes(G, M, D,F,1));
//
//    // Assert
//    for(size_t i = 0; i < F.getN(); i++) {
//        for (size_t j = 0; j < F.getM(); j++) {
//
//            EXPECT_EQ(F[i][j].getN(), 2);
//            EXPECT_EQ(F[i][j].getM(), 2);
//
//            MAT_TEST(F[i][j], -7);
//        }
//    }
//}
//
//TEST(LearnFilter_functions, BackPropagation_pooling_matrix_Test){
//    // Arrange
//    Matrix<Matrix<double>> D(1,10);
//    Matrix<Matrix<double>> M(1,10);
//    Matrix<Matrix<double>> OUT(1,10);
//    Matrix<Matrix<double>> o(1,1);
//
//    // Act
//    for(size_t i = 0; i < D.getN(); i++){
//        for(size_t j = 0; j < D.getM(); j++) {
//            D[i][j] = Matrix<double>(2,2);
//            D[i][j].Fill(5);
//
//            M[i][j] = Matrix<double>(4,4);
//            M[i][j][0][0] = 1;
//            M[i][j][0][1] = 2;
//            M[i][j][0][2] = 1;
//            M[i][j][0][3] = 2;
//
//            M[i][j][1][0] = 3;
//            M[i][j][1][1] = 4;
//            M[i][j][1][2] = 3;
//            M[i][j][1][3] = 5;
//
//            M[i][j][2][0] = 1;
//            M[i][j][2][1] = 2;
//            M[i][j][2][2] = 1;
//            M[i][j][2][3] = 2;
//
//            M[i][j][3][0] = 3;
//            M[i][j][3][1] = 6;
//            M[i][j][3][2] = 3;
//            M[i][j][3][3] = 7;
//
//            OUT[i][j] = Matrix<double>(2,2);
//
//            OUT[i][j][0][0] = 4;
//            OUT[i][j][0][1] = 5;
//            OUT[i][j][1][0] = 6;
//            OUT[i][j][1][1] = 7;
//        }
//    }
//
//    EXPECT_NO_THROW(o = BackPropagation(M,OUT, D, 2, 2));
//
//    // Assert
//    for(size_t i = 0; i < o.getN(); i++) {
//        for (size_t j = 0; j < o.getM(); j++) {
//            EXPECT_EQ(o[i][j].getN(), 4);
//            EXPECT_EQ(o[i][j].getM(), 4);
//
//            EXPECT_EQ(o[i][j][0][0], 0);
//            EXPECT_EQ(o[i][j][0][1], 0);
//            EXPECT_EQ(o[i][j][0][2], 0);
//            EXPECT_EQ(o[i][j][0][3], 0);
//
//            EXPECT_EQ(o[i][j][1][0], 0);
//            EXPECT_EQ(o[i][j][1][1], 5);
//            EXPECT_EQ(o[i][j][1][2], 0);
//            EXPECT_EQ(o[i][j][1][3], 5);
//
//            EXPECT_EQ(o[i][j][2][0], 0);
//            EXPECT_EQ(o[i][j][2][1], 0);
//            EXPECT_EQ(o[i][j][2][2], 0);
//            EXPECT_EQ(o[i][j][2][3], 0);
//
//            EXPECT_EQ(o[i][j][3][0], 0);
//            EXPECT_EQ(o[i][j][3][1], 5);
//            EXPECT_EQ(o[i][j][3][2], 0);
//            EXPECT_EQ(o[i][j][3][3], 5);
//        }
//    }
//}