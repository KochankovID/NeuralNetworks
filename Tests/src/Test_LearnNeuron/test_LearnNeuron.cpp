#include <gtest/gtest.h>
#include "LearnNeuron.h"
#include <fstream>

using namespace NN;

#define MAT_TEST(X,Y) for(size_t ii = 0; ii < X.getN(); ii++){ for(size_t jj = 0; jj < X.getM(); jj++){ EXPECT_EQ(X[ii][jj], Y); }}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_squared_Test){
    // Arrange
    I_Neuron n(3, 3);
    Matrix<I_Neuron> m(3, 3);
    I_Matrix der(3,3);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    der.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(m[i][j].GetD(), 50);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_not_squared_Test){
    // Arrange
    I_Neuron n(2, 5);
    Matrix<I_Neuron> m(2, 5);
    I_Matrix der(2, 5);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    der.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            EXPECT_EQ(m[i][j].GetD(), 50);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_singl_Test){
    // Arrange
    I_Neuron n(1, 1);
    Matrix<I_Neuron> m(1, 1);
    I_Matrix derr(1,1);

    // Act
    n.Fill(5);
    n.GetD() = 10;
    derr.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, derr));

    // Assert
    EXPECT_EQ(m[0][0].GetD(), 50);

}

TEST(LearnNeyron_functions, BackPropagation_one_neyron_wrong_size_Test){
    // Arrange
    I_Neuron n(2, 1);
    Matrix<I_Neuron> m(1, 1);
    I_Matrix derr(1,1);


    // Act
    n.Fill(5);
    derr.Fill(1);
    n.GetD() = 10;

    // Assert
    EXPECT_ANY_THROW(BackPropagation(m, n, derr));

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_squared_Test){
    // Arrange
    Matrix<I_Neuron> n(3, 3);
    Matrix<I_Neuron> m(3, 3);
    I_Matrix der(3,3);

    // Act
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(m[i][j].GetD(), 450);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_not_squared_Test){
    // Arrange
    Matrix<I_Neuron> n(2, 5);
    Matrix<I_Neuron> m(2, 5);
    I_Matrix der(2, 5);

    // Act
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            n[i][j] = I_Neuron(2, 5);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            EXPECT_EQ(m[i][j].GetD(), 500);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_singe_Test){
    // Arrange
    Matrix<I_Neuron> n(1, 1);
    Matrix<I_Neuron> m(1, 1);
    I_Matrix der(1,1);

    // Act
    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 1; j++){
            n[i][j] = I_Neuron(1, 1);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    EXPECT_EQ(m[0][0].GetD(), 50);

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_zero_Test){
    // Arrange
    Matrix<I_Neuron> n;
    Matrix<I_Neuron> m;
    I_Matrix der;

    // Act
    EXPECT_NO_THROW(BackPropagation(m, n, der));

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(n.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
    EXPECT_EQ(n.getM(), 0);

}

TEST(LearnNeyron_functions, BackPropagation_matrix_neyron_wrong_size_Test){
    // Arrange
    Matrix<I_Neuron> n(2, 1);
    Matrix<I_Neuron> m(1, 1);
    I_Matrix der(1, 1);

    // Act
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 1; j++){
            n[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);

    // Assert
    EXPECT_ANY_THROW(BackPropagation(m, n, der));
}

TEST(LearnNeyron_functions, BackPropagation_matrix_squared_Test){
    // Arrange
    Matrix<I_Neuron> n(3, 3);
    Matrix<int> m(3,3);
    I_Matrix der(3, 3);

    // Act
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    m.Fill(1);
    EXPECT_NO_THROW(BackPropagation(n, m, der));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            EXPECT_EQ(n[i][j].GetD(), 11);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_not_squared_Test){
    // Arrange
    Matrix<I_Neuron> n(2, 5);
    Matrix<int > m(2,5);
    I_Matrix der(2, 5);

    // Act
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            n[i][j] = I_Neuron(2, 5);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    m.Fill(1);
    EXPECT_NO_THROW(BackPropagation(n, m, der));

    // Assert
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 5; j++){
            EXPECT_EQ(n[i][j].GetD(), 11);
        }
    }

}

TEST(LearnNeyron_functions, BackPropagation_matrix_singe_Test){
    // Arrange
    Matrix<I_Neuron> n(1, 1);
    Matrix<int> m(1,1);
    I_Matrix der(1,1);

    // Act
    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 1; j++){
            n[i][j] = I_Neuron(1, 1);
            n[i][j].Fill(5);
            n[i][j].GetD() = 10;
        }
    }
    der.Fill(1);
    m.Fill(1);
    EXPECT_NO_THROW(BackPropagation(n, m, der));

    // Assert
    EXPECT_EQ(n[0][0].GetD(), 11);

}

TEST(LearnNeyron_functions, BackPropagation_matrix__zero_Test){
    // Arrange
    Matrix<I_Neuron> n;
    Matrix<int > m;
    I_Matrix der;

    // Act
    EXPECT_NO_THROW(BackPropagation(n, m, der));

    // Assert
    EXPECT_EQ(m.getN(), 0);
    EXPECT_EQ(n.getN(), 0);
    EXPECT_EQ(m.getM(), 0);
    EXPECT_EQ(n.getM(), 0);

}

//TEST(LearnNeyron_functions, BackPropagation_matrix_wrong_size_Test){
//    // Arrange
//    Matrix<I_Neyron> n(2,1);
//    Matrix<int> m(1,1);
//    I_Matrix der(2, 1);
//
//    // Act
//    for(size_t i = 0; i < 2; i++){
//        for(size_t j = 0; j < 1; j++){
//            n[i][j] = I_Neyron(3,3);
//            n[i][j].Fill(5);
//            n[i][j].GetD() = 10;
//        }
//    }
//    der.Fill(1);
//
//    // Assert
//    EXPECT_ANY_THROW(BackPropagation(n, m, der));
//}

TEST(LearnNeyron_functions, GradDes_Test){
    // Arrange
    I_Neuron n(3, 3);
    I_Neuron h(3, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    n.Fill(1);
    n.GetD() = 3;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    MAT_TEST(n, -2)

}

TEST(LearnNeyron_functions, GradDes_zero_der_Test){
    // Arrange
    I_Neuron n(3, 3);
    I_Neuron h(3, 3);

    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    n.Fill(1);
    n.GetD() = 0;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    MAT_TEST(n, 1)

}

TEST(LearnNeyron_functions, GradDes_negative_der_Test){
    // Arrange
    I_Neuron n(3, 3);
    I_Neuron h(3, 3);

    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    n.Fill(1);
    n.GetD() = -3;
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    MAT_TEST(n, 4)

}

TEST(LearnNeyron_functions, GradDes_wrong_size_Test){
    // Arrange
    I_Neuron n(4, 3);
    I_Neuron h(4, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    n.Fill(1);
    n.GetD() = -3;
    m.Fill(1);

    // Assert
    EXPECT_ANY_THROW(GradDes(G,n,m,h));

}

TEST(LearnNeyron_functions, GradDes_matrix_Test){
    // Arrange
    Matrix<I_Neuron> n(3, 3);
    Matrix<I_Neuron> h(3, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(1);
            n[i][j].GetD() = 3;
            h[i][j] = I_Neuron(3, 3);
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], -2)
        }
    }
}

TEST(LearnNeyron_functions, GradDes_matrix_zero_der_Test){
    // Arrange
    Matrix<I_Neuron> n(3, 3);
    Matrix<I_Neuron> h(3, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1, 0);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 3);
            h[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(1);
            n[i][j].GetD() = 0;
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], 1)
        }
    }

}

TEST(LearnNeyron_functions, GradDes_matrix_negative_der_Test){
    // Arrange
    Matrix<I_Neuron> n(3, 3);
    Matrix<I_Neuron> h(3, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1 ,0);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 3);
            h[i][j] = I_Neuron(3, 3);
            n[i][j].Fill(1);
            n[i][j].GetD() = -3;
        }
    }
    m.Fill(1);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            MAT_TEST(n[i][j], 4)
        }
    }

}

TEST(LearnNeyron_functions, GradDes_matrix_wrong_size_Test){
    // Arrange
    Matrix<I_Neuron> n(4, 3);
    Matrix<I_Neuron> h(4, 3);
    Matrix<int> m(3,3);
    SGD<int> G(1,0);

    // Act
    for(size_t i =0; i < 3; i++){
        for(size_t j = 0; j < 3; j++){
            n[i][j] = I_Neuron(3, 4);
            h[i][j] = I_Neuron(3, 4);
            n[i][j].Fill(1);
            n[i][j].GetD() = -3;
        }
    }
    m.Fill(1);

    // Assert
    EXPECT_ANY_THROW(GradDes(G,n,m,h));

}

TEST(LearnNeyron_functions, GradDes_with_history_Test){
    // Arrange
    D_Neuron n(1, 1);
    D_Neuron h(1, 1);
    Matrix<double> m(1,1);
    SGD<double> G(1, 0.9);

    // Act
    n.Fill(1);
    n.GetD() = 3;
    m.Fill(1);
    h.Fill(0);
    EXPECT_NO_THROW(GradDes(G,n,m,h));

    // Assert
    EXPECT_DOUBLE_EQ(n[0][0], 0.7);

}

TEST(LearnNeyron_functions, loss_function_Test){
    // Arrange
    RMS_error<int> R;
    Matrix<int> result(1,4);
    Matrix<int> correct(1,4);
    Matrix<double> err;

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
    Matrix<double> err;

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
    Neuron<double > n(5, 5);
    int d = 1;

    // Act
    n.Fill(10);
    EXPECT_NO_THROW(retract(n, d));

    // Assert
    MAT_TEST(n, 9.9);

}

TEST(LearnNeyron_functions, retract_negative_Test){
    // Arrange
    Neuron<double > n(5, 5);
    int d = 1;

    // Act
    n.Fill(-10);
    EXPECT_NO_THROW(retract(n, d));

    // Assert
    MAT_TEST(n, -9.9);

}

TEST(LearnNeyron_functions, retract_matrix_Test){
    // Arrange
    Matrix<D_Neuron> m(5, 5);
    int d = 1;

    // Act
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            m[i][j] = D_Neuron(5, 5);
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
    Matrix<D_Neuron> m(5, 5);
    int d = 1;

    // Act
    for(size_t i = 0; i < 5; i++){
        for(size_t j = 0; j < 5; j++){
            m[i][j] = D_Neuron(5, 5);
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
    I_Neuron n(3, 3);
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
    I_Neuron n(3, 3);
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
    I_Neuron n(3, 3);
    Matrix<int> m(4,3);
    double speed = 1;

    // Act
    n.Fill(1);
    m.Fill(1);
    m[1][1] = 0;

    // Assert
    EXPECT_ANY_THROW(SimpleLearning(a,y,n,m,speed));

}

TEST(LearnNeyron_functions, BackPropeteion_full_Test){
    // Arrange
    Sigm<double> f(1);
    SigmD<double> fd(1);
    D_Matrix input(2,2);
    SGD<double> G(1, 0);
    Matrix<D_Neuron> layer1(1, 1);
    Matrix<D_Neuron> H1(1, 1);
    Matrix<D_Neuron> layer2(1, 1);
    Matrix<D_Neuron> H2(1, 1);
    D_Matrix der1(1,1);
    D_Matrix der2(1,1);
    D_Matrix out1(1,1), out2(1,1);
    D_Matrix error;
    D_Matrix correct(1, 1);
    RMS_errorD<double> rms;

    // Act
    input[0][0] = 0;
    input[0][1] = 1;
    input[1][0] = 0;
    input[1][1] = 0;

    layer1[0][0] = D_Neuron(2, 2);
    H1[0][0] = D_Neuron(2, 2);
    layer2[0][0] = D_Neuron(1, 1);
    H2[0][0] = D_Neuron(1, 1);

    layer1[0][0].Fill(0);
    layer2[0][0].Fill(0);

    out1[0][0] = D_Neuron::FunkActiv(layer1[0][0].Summator(input), f);
    der1[0][0] = fd(layer1[0][0].Summator(input));
    out2[0][0] = D_Neuron::FunkActiv(layer2[0][0].Summator(out1), f);
    der2[0][0] = fd(layer2[0][0].Summator(out1));

    correct[0][0] = 1;

    error = loss_function(rms, out2, correct);

    BackPropagation(layer2, error,der2 );
    BackPropagation(layer1, layer2,der1);

    GradDes(G, layer1, input, H1);
    GradDes(G, layer2, out1, H2);

    // Assert
    EXPECT_EQ(layer1[0][0][0][0], 0);
    EXPECT_EQ(layer1[0][0][0][1], 0);
    EXPECT_EQ(layer1[0][0][1][0], 0);
    EXPECT_EQ(layer1[0][0][1][1], 0);

    EXPECT_EQ(layer2[0][0][0][0], 0.125);

    EXPECT_EQ(der1[0][0], 0.25);
    EXPECT_EQ(der2[0][0], 0.25);

    EXPECT_EQ(layer1[0][0].GetD(), 0);
    EXPECT_EQ(layer2[0][0].GetD(), -0.25);

    EXPECT_EQ(out1[0][0], 0.5);
    EXPECT_EQ(out2[0][0], 0.5);
}