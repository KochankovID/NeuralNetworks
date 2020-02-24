#include "opencv2/ts.hpp"
#include <fstream>
#include "Matrix.h"
Matrix<double>A(1500, 1500);
PERF_TEST(Matrix_, GetPodmatrix_perf)
{
	Matrix<double> T;

	PERF_SAMPLE_BEGIN()
		for(int i = 0; i<20; i++)
		T = A.getPodmatrix(1, 1, 80, 80);
	PERF_SAMPLE_END()

		SANITY_CHECK_NOTHING();
}