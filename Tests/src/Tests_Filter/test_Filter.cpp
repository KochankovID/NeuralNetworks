#include "opencv2/ts.hpp"
#include <fstream>
#include "Filter.h"

TEST(Filter_, Rotate_180){
    Filter<int> A(3,3);

    int t = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j] = t++;
        }
    }

    std::ofstream out("FilterTest.txt");
    out << A;

    Filter<int> Y;
    Y = A.roate_180();
    EXPECT_EQ(Y.getN(), A.getN());
    EXPECT_EQ(Y.getM(), A.getM());
    int n = A.getN()-1, m = A.getM()-1;

    for (int i = n; i >= 0; i--) {
        for (int j = m; j >= 0; j--) {
            EXPECT_EQ(Y[i][j], A[n - i][m - j]);
        }
    }

    out << Y;
    out.close();
}
