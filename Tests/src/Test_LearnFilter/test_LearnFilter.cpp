#include <opencv2/ts.hpp>
#include "LearnFilter.h"

using namespace ANN;

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
