#include <opencv2/ts.hpp>
#include "Matrix.h"
#include "NeyronCnn.h"
#include <fstream>

PERF_TEST (NeyronCnn_, Svertka_perf)
{
	NeyronCnn<double> A;
	Filter<double> y(3, 3);
	double **f = new double*[1];
	f[0] = new double[1];
	f[0][0] = 5;
	Matrix<double> FF(f, 1, 1);
	A.Padding(FF);
	PERF_SAMPLE_BEGIN()
		FF = A.Svertka(y, FF);
	PERF_SAMPLE_END()
		delete[](f[0]);
	    delete[](f);
		SANITY_CHECK_NOTHING();
}