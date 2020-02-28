#include "opencv2/ts.hpp"
#include <fstream>
#include "Matrix.h"

Matrix<int> A(3,3);
Matrix<double> B(3,3);

TEST(Matrix_, Constructor){

    int tt = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j] = tt;
            B[i][j] = tt++;
        }
    }

    Matrix<int> R;
    EXPECT_EQ(R.getN(), 0);
    EXPECT_EQ(R.getM(), 1);

    double** t;
    t = new double*[2];
    t[0] = new double[2];
    t[1] = new double[2];

    t[0][0] = 0;
    t[0][1] = 1;
    t[1][0] = 2;
    t[1][1] = 3;

    Matrix<double> D(t, 2, 2);
    EXPECT_EQ(D.getN(), 2);
    EXPECT_EQ(D.getM(), 2);

    EXPECT_EQ(D[0][0], 0);
    EXPECT_EQ(D[0][1], 1);
    EXPECT_EQ(D[1][0], 2);
    EXPECT_EQ(D[1][1], 3);

    double* ty;
    ty = new double[4];
    ty[0] = 0;
    ty[1] = 1;
    ty[2] = 2;
    ty[3] = 3;

    Matrix<double> DD(ty, 2, 2);
    EXPECT_EQ(DD.getN(), 2);
    EXPECT_EQ(DD.getM(), 2);

    EXPECT_EQ(DD[0][0], 0);
    EXPECT_EQ(DD[0][1], 1);
    EXPECT_EQ(DD[1][0], 2);
    EXPECT_EQ(DD[1][1], 3);

    Matrix<int> G(2, 3);
    EXPECT_EQ(G.getN(), 2);
    EXPECT_EQ(G.getM(), 3);

    EXPECT_EQ(G[0][0], 0);
    EXPECT_EQ(G[0][1], 0);
    EXPECT_EQ(G[0][2], 0);
    EXPECT_EQ(G[1][0], 0);
    EXPECT_EQ(G[1][1], 0);
    EXPECT_EQ(G[1][2], 0);

    Matrix<double> P(D);
    EXPECT_EQ(P.getN(), 2);
    EXPECT_EQ(P.getM(), 2);

    EXPECT_EQ(P[0][0], 0);
    EXPECT_EQ(P[0][1], 1);
    EXPECT_EQ(P[1][0], 2);
    EXPECT_EQ(P[1][1], 3);
}

TEST(Matrix, Constructor_wrong_arguments) {
    EXPECT_ANY_THROW(Matrix<int> t(-1, -2));

    double** t;
    t = new double*[2];
    t[0] = new double[2];
    t[1] = new double[2];

    EXPECT_ANY_THROW(Matrix<double> W(t, -1, -2));
}

TEST(Matrix_, GetN){
    EXPECT_EQ(A.getN(), 3);
}

TEST(Matrix_, GetM){
    EXPECT_EQ(A.getM(), 3);
}

TEST(Matrix_, Max_arr){
    double** t;
    t = new double*[2];
    t[0] = new double[2];
    t[1] = new double[2];

    t[0][0] = 0;
    t[0][1] = 1;
    t[1][0] = 2;
    t[1][1] = 3;

    EXPECT_EQ(Matrix<double>::Max(t, 2, 2), 3);
}

TEST(Matrix_, Max_matrix){

    EXPECT_EQ(A.Max(), 8);
}

TEST(Matrix_, GetCopy){

    int** a;
    a = A.getCopy();

    EXPECT_EQ(a[0][0], 0);
    EXPECT_EQ(a[0][1], 1);
    EXPECT_EQ(a[0][2], 2);
    EXPECT_EQ(a[1][0], 3);
    EXPECT_EQ(a[1][1], 4);
    EXPECT_EQ(a[1][2], 5);
    EXPECT_EQ(a[2][0], 6);
    EXPECT_EQ(a[2][1], 7);
    EXPECT_EQ(a[2][2], 8);
}

TEST(Matrix_, GetPodmatrix){
    Matrix<double> T;
    T = B.getPodmatrix(1, 1, 2, 2);

    EXPECT_EQ(T[0][0], 4);
    EXPECT_EQ(T[0][1], 5);
    EXPECT_EQ(T[1][0], 7);
    EXPECT_EQ(T[1][1], 8);
}

TEST(Matrix_, GetPodmatrix_wrond_arguments){
    EXPECT_ANY_THROW(A.getPodmatrix(-1, 100, 2, 2));
    EXPECT_ANY_THROW(A.getPodmatrix(1, 1, 3, 3););
}

TEST(Matrix_, Operator_assignment){
    Matrix<int> E;
    E = A;
    EXPECT_EQ(E.getN(), 3);
    EXPECT_EQ(E.getM(), 3);

    EXPECT_EQ(E[0][0], 0);
    EXPECT_EQ(E[0][1], 1);
    EXPECT_EQ(E[0][2], 2);
    EXPECT_EQ(E[1][0], 3);
    EXPECT_EQ(E[1][1], 4);
    EXPECT_EQ(E[1][2], 5);
    EXPECT_EQ(E[2][0], 6);
    EXPECT_EQ(E[2][1], 7);
    EXPECT_EQ(E[2][2], 8);
}

TEST(Matrix_, Operator_addition){
    Matrix<double> Y(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Y[i][j] = 1;
        }
    }
    Y = Y + B;

    EXPECT_EQ(Y.getN(), 3);
    EXPECT_EQ(Y.getM(), 3);

    EXPECT_EQ(Y[0][0], 1);
    EXPECT_EQ(Y[0][1], 2);
    EXPECT_EQ(Y[0][2], 3);
    EXPECT_EQ(Y[1][0], 4);
    EXPECT_EQ(Y[1][1], 5);
    EXPECT_EQ(Y[1][2], 6);
    EXPECT_EQ(Y[2][0], 7);
    EXPECT_EQ(Y[2][1], 8);
    EXPECT_EQ(Y[2][2], 9);
}

TEST(Matrix_, Operator_addition_wrong_arguments){
    Matrix<int> UU(1, 1);
    EXPECT_ANY_THROW(A + UU);
}

TEST(Matrix_, Operator_matrix_multiplication){
    Matrix<double> Y(3, 3);
    Matrix<double> O(3, 1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 1; j++) {
            Y[i][j] = 0;
        }
    }

    O = Y * O;
    EXPECT_EQ(O.getN(), 3);
    EXPECT_EQ(O.getM(), 1);

    EXPECT_EQ(O[0][0], 0);
    EXPECT_EQ(O[1][0], 0);
    EXPECT_EQ(O[2][0], 0);
}

TEST(Matrix_, Operator_matrix_multiplication_wrong_arguments){
    Matrix<int> UUU(1, 1);
    EXPECT_ANY_THROW(A * UUU);
}

TEST(Matrix_, Operator_matrix_on_num_multiplication){
    Matrix<double> U(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[i][j] = 1;
        }
    }
    U = U * 8;

    EXPECT_EQ(U[0][0], 8);
    EXPECT_EQ(U[0][1], 8);
    EXPECT_EQ(U[0][2], 8);
    EXPECT_EQ(U[1][0], 8);
    EXPECT_EQ(U[1][1], 8);
    EXPECT_EQ(U[1][2], 8);
    EXPECT_EQ(U[2][0], 8);
    EXPECT_EQ(U[2][1], 8);
    EXPECT_EQ(U[2][2], 8);

    U = 0 * U;

    EXPECT_EQ(U[0][0], 0);
    EXPECT_EQ(U[0][1], 0);
    EXPECT_EQ(U[0][2], 0);
    EXPECT_EQ(U[1][0], 0);
    EXPECT_EQ(U[1][1], 0);
    EXPECT_EQ(U[1][2], 0);
    EXPECT_EQ(U[2][0], 0);
    EXPECT_EQ(U[2][1], 0);
    EXPECT_EQ(U[2][2], 0);
}

TEST(Matrix_, Operator_output_input) {
    Matrix<int> M;

    std::ofstream file;
    file.open("MatrixTest.txt");
    file << A;
    file.close();
    std::ifstream fileIn;
    fileIn.open("MatrixTest.txt");
    fileIn >> M;
    fileIn.close();

    EXPECT_EQ(M.getN(), 3);
    EXPECT_EQ(M.getM(), 3);

    EXPECT_EQ(M[0][0], 0);
    EXPECT_EQ(M[0][1], 1);
    EXPECT_EQ(M[0][2], 2);
    EXPECT_EQ(M[1][0], 3);
    EXPECT_EQ(M[1][1], 4);
    EXPECT_EQ(M[1][2], 5);
    EXPECT_EQ(M[2][0], 6);
    EXPECT_EQ(M[2][1], 7);
    EXPECT_EQ(M[2][2], 8);
}

TEST(Matrix_, Operator_index) {
    EXPECT_EQ(A[0][0], 0);
}

TEST(Matrix_, Operator_compare) {
    EXPECT_EQ(A[0][0], 0);
}

TEST(Matrix_, InInRange_wrong_work) {
    EXPECT_ANY_THROW(A[10][10]);
}

