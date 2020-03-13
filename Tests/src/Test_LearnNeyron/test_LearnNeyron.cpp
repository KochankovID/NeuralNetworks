#include <opencv2/ts.hpp>
#include "LearnNeyron.h"
#include "Neyrons.h"
#include "Functors.h"
#include "Gradients.h"
#include <fstream>

using namespace ANN;

#define MAT_TEST(X,Y) for(size_t ii = 0; ii < X.getN(); ii++){ for(size_t jj = 0; jj < X.getM(); jj++){ EXPECT_EQ(X[ii][jj], Y); }}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_squared_Test){
    // Arrange
    I_Neyron n(3,3);
    Matrix<I_Neyron> m(3,3);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(m[i][j].GetD(), 50);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_not_squared_Test){
    // Arrange
    I_Neyron n(2,5);
    Matrix<I_Neyron> m(2,5);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            EXPECT_EQ(m[i][j].GetD(), 50);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_singl_Test){
    // Arrange
    I_Neyron n(1,1);
    Matrix<I_Neyron> m(1,1);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    EXPECT_EQ(m[0][0].GetD(), 50);

}


TEST(LearnNeyron_functions, BackPropagation_one_neyron_wrong_size_Test){
    // Arrange
    I_Neyron n(2,1);
    Matrix<I_Neyron> m(1,1);

    // Act
    n.Fill(5);
    n.GetD() = 10;

    // Assert
    EXPECT_ANY_THROW(BackPropagation(m, n));

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_squared_Test){
    // Arrange
    Matrix<I_Neyron> n(3,3);
    Matrix<I_Neyron> m(3,3);

    // Act
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neyron(3,3);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(m[i][j].GetD(), 450);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_not_squared_Test){
    // Arrange
    Matrix<I_Neyron> n(2,5);
    Matrix<I_Neyron> m(2,5);

    // Act
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            n[i][j] = I_Neyron(2,5);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            EXPECT_EQ(m[i][j].GetD(), 500);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_singe_Test){
    // Arrange
    Matrix<I_Neyron> n(1,1);
    Matrix<I_Neyron> m(1,1);

    // Act
    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 1; j++){
            n[i][j] = I_Neyron(1, 1);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    EXPECT_EQ(m[0][0].GetD(), 50);

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_zero_Test){
    // Arrange
    Matrix<I_Neyron> n;
    Matrix<I_Neyron> m;

    // Act
    EXPECT_NO_THROW(BackPropagation(m, n));

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(n.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
    EXPECT_EQ(n.getM(), 0);

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_wrong_size_Test){
    // Arrange
    Matrix<I_Neyron> n(2,1);
    Matrix<I_Neyron> m(1,1);

    // Act
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 1; j++){
            n[i][j] = I_Neyron(3,3);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }

    // Assert
    EXPECT_ANY_THROW(BackPropagation(m, n));
}

TEST(LearnNeyron_functions, GradDes_Test){
    // Arrange
    ReluD<int> F(1);
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    n.Fill(1);
    n.GetD() = 3;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    MAT_TEST(n, -2)

}

TEST(LearnNeyron_functions, GradDes_zero_der_Test){
    // Arrange
    ReluD<int> F(1);
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    n.Fill(1);
    n.GetD() = 0;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    MAT_TEST(n, 1)

}

TEST(LearnNeyron_functions, GradDes_negative_der_Test){
    // Arrange
    ReluD<int> F(1);
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    n.Fill(1);
    n.GetD() = -3;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    MAT_TEST(n, 4)

}

TEST(LearnNeyron_functions, GradDes_wrong_size_Test){
    // Arrange
    ReluD<int> F(1);
    I_Neyron n(4,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    n.Fill(1);
    n.GetD() = -3;
    m.Fill(1);

    // Assert
    EXPECT_ANY_THROW(GradDes(G,n,m,F));

}

TEST(LearnNeyron_functions, GradDes_find_mimnun_Test){
    // Arrange
    Relu<int> F(1);
    ReluD<int> F_D(1);
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(0.0002);
    int result = 10, sum, error = 10;

    // Act
    n.Fill(100000);
    m.Fill(1);

    while (error !=0) {
        sum = n.Summator(m);
        result = Neyron<int>::FunkActiv(sum, F);
        error = 2 * (result - 0);
        n.GetD() = error;
        EXPECT_NO_THROW(GradDes(G,n,m,F));
    }

    // Assert
    EXPECT_EQ(error, 0);

}

TEST(LearnNeyron_functions, GradDes_matrix_Test){
    // Arrange
    ReluD<int> F(1);
    Matrix<I_Neyron> n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neyron(3,3);
            n[i][j].Fill(1);
            n[i][j].GetD() = 3;
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], -2)
        }
    }
}

TEST(LearnNeyron_functions, GradDes_matrix_zero_der_Test){
    // Arrange
    ReluD<int> F(1);
    Matrix<I_Neyron> n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neyron(3,3);
            n[i][j].Fill(1);
            n[i][j].GetD() = 0;
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], 1)
        }
    }

}

TEST(LearnNeyron_functions, GradDes_matrix_negative_der_Test){
    // Arrange
    ReluD<int> F(1);
    Matrix<I_Neyron> n(3,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neyron(3,3);
            n[i][j].Fill(1);
            n[i][j].GetD() = -3;
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,F));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], 4)
        }
    }

}

TEST(LearnNeyron_functions, GradDes_matrix_wrong_size_Test){
    // Arrange
    ReluD<int> F(1);
    Matrix<I_Neyron> n(4,3);
    Matrix<int> m(3,3);
    SimpleGrad<int> G(1);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neyron(3,4);
            n[i][j].Fill(1);
            n[i][j].GetD() = -3;
        }
    }
    m.Fill(1);

    // Assert
    EXPECT_ANY_THROW(GradDes(G,n,m,F));

}

TEST(LearnNeyron_functions, loss_function_Test){
    // Arrange
    RMS_error<int> R;
    Matrix<int> result(1,4);
    Matrix<int> correct(1,4);
    Matrix<int> err;

    // Act
    for(size_t i = 0; i < 4; i++){
        result[0][i] = 1;
        correct[0][i] = 0;
    }
    EXPECT_NO_THROW(err = loss_function(R, result, correct));

    // Assert
    EXPECT_EQ(err[0][0], 1);

}

TEST(LearnNeyron_functions, loss_function_wrong_size_Test){
    // Arrange
    RMS_error<int> R;
    Matrix<int> result(1, 5);
    Matrix<int> correct(1, 4);
    Matrix<int> err;

    // Act
    for(size_t i = 0; i < 4; i++){
        result[0][i] = 1;
        correct[0][i] = 0;
    }

    // Assert
    EXPECT_ANY_THROW(err = loss_function(R, result, correct));

}

TEST(LearnNeyron_functions, metric_function_Test){
    // Arrange
    RMS_error<double > R;
    Matrix<double> result(1, 4);
    Matrix<double> correct(1, 4);
    double err;

    // Act
    for(size_t i = 0; i < 4; i++){
        result[0][i] = 1;
        correct[0][i] = 0;
    }
    EXPECT_NO_THROW(err = metric_function(R, result, correct));

    // Assert
    EXPECT_EQ(err, 1);

}

TEST(LearnNeyron_functions, metric_function_wrong_size_Test){
    // Arrange
    RMS_error<double > R;
    Matrix<double > result(1,5);
    Matrix<double > correct(1,4);
    double err;

    // Act
    for(size_t i = 0; i < 4; i++){
        result[0][i] = 1;
        correct[0][i] = 0;
    }

    // Assert
    EXPECT_ANY_THROW(err = metric_function(R, result, correct));

}

TEST(LearnNeyron_functions, retract_Test){
    // Arrange
    Neyron<double > n(5, 5);
    int d = 1;

    // Act
    n.Fill(10);
    EXPECT_NO_THROW(retract(n, d));

    // Assert
    MAT_TEST(n, 9.9);

}

TEST(LearnNeyron_functions, retract_negative_Test){
    // Arrange
    Neyron<double > n(5, 5);
    int d = 1;

    // Act
    n.Fill(-10);
    EXPECT_NO_THROW(retract(n, d));

    // Assert
    MAT_TEST(n, -9.9);

}

TEST(LearnNeyron_functions, retract_matrix_Test){
    // Arrange
    Matrix<D_Neyron> m(5,5);
    int d = 1;

    // Act
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            m[i][j] = D_Neyron(5,5);
            m[i][j].Fill(10);

        }
    }
    EXPECT_NO_THROW(retract(m, d));

    // Assert
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            for(size_t ii = 0; ii < 5; ii++) {
                for (size_t jj = 0; jj < 5; jj++) {
                    EXPECT_EQ(m[i][j][ii][jj], 9.9);
                }
            }
        }
    }

}

TEST(LearnNeyron_functions, retract_matrix_negative_Test){
    // Arrange
    Matrix<D_Neyron> m(5,5);
    int d = 1;

    // Act
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            m[i][j] = D_Neyron(5,5);
            m[i][j].Fill(-10);

        }
    }
    EXPECT_NO_THROW(retract(m, d));

    // Assert
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            for(size_t ii = 0; ii < 5; ii++) {
                for (size_t jj = 0; jj < 5; jj++) {
                    EXPECT_EQ(m[i][j][ii][jj], -9.9);
                }
            }
        }
    }

}

TEST(LearnNeyron_functions, SimpleLearning_dont_recognize_Test){
    // Arrange
    int a = 1;
    int y = 0;
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    double speed = 1;

    // Act
    n.Fill(0);
    m.Fill(1);
    m[1][1] = 0;

    EXPECT_NO_THROW(SimpleLearning(a,y,n,m,speed));

    // Assert
    EXPECT_EQ(n[0][0], 1);
    EXPECT_EQ(n[0][1], 1);
    EXPECT_EQ(n[0][2], 1);

    EXPECT_EQ(n[1][0], 1);
    EXPECT_EQ(n[1][1], 0);
    EXPECT_EQ(n[1][2], 1);

    EXPECT_EQ(n[2][0], 1);
    EXPECT_EQ(n[2][1], 1);
    EXPECT_EQ(n[2][2], 1);

}

TEST(LearnNeyron_functions, SimpleLearning_recognize_wrong_Test){
    // Arrange
    int a = 0;
    int y = 1;
    I_Neyron n(3,3);
    Matrix<int> m(3,3);
    double speed = 1;

    // Act
    n.Fill(1);
    m.Fill(1);
    m[1][1] = 0;

    EXPECT_NO_THROW(SimpleLearning(a,y,n,m,speed));

    // Assert
    EXPECT_EQ(n[0][0], 0);
    EXPECT_EQ(n[0][1], 0);
    EXPECT_EQ(n[0][2], 0);

    EXPECT_EQ(n[1][0], 0);
    EXPECT_EQ(n[1][1], 1);
    EXPECT_EQ(n[1][2], 0);

    EXPECT_EQ(n[2][0], 0);
    EXPECT_EQ(n[2][1], 0);
    EXPECT_EQ(n[2][2], 0);

}

TEST(LearnNeyron_functions, SimpleLearning_wrong_size_Test){
    // Arrange
    int a = 0;
    int y = 1;
    I_Neyron n(3,3);
    Matrix<int> m(4,3);
    double speed = 1;

    // Act
    n.Fill(1);
    m.Fill(1);
    m[1][1] = 0;

    // Assert
    EXPECT_ANY_THROW(SimpleLearning(a,y,n,m,speed));

}