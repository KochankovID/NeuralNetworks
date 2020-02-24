#include <opencv2/ts.hpp>
#include "Matrix.h"
#include "NeyronCnn.h"
#include <fstream>

PERF_TEST (NeyronCnn_, Svertka_perf)
{
	NeyronCnn<double> A;
	Filter<double> y(10, 10);

	for(int i = 0; i < 5; i++){
	    for(int j = 0; j < 5; j++){
	        y[i][j] = 4;
	    }
	}

	Matrix<double> f(100,100);

	for(int  i =0; i < 100; i++){
	    for (int j = 0; j < 100; j++){
	        f[i][j] = i+j*0.5;
	    }
	}

	PERF_SAMPLE_BEGIN()
		A.Svertka(y, f);
	PERF_SAMPLE_END()


	SANITY_CHECK_NOTHING();
}