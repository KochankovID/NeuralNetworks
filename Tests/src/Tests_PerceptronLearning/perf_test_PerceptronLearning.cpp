#include "opencv2/ts.hpp"
#include <fstream>
#include "PLearns.h"
#include "Functors.h"


PERF_TEST(PercetronLearning, BackPropagation_perf)
{
	D_Leaning C;
	Matrix<double> UUU(2, 2);
	UUU[0][0] = 1;
	UUU[0][1] = 1;
	UUU[1][0] = 0;
	UUU[1][1] = 1;

	Matrix<Weights<double>> II(2, 2);
	Weights<double> IIu(2, 2);
	Matrix<double> III(2, 2);
	III[0][0] = 1;
	III[0][1] = 1;
	III[1][0] = 0;
	III[1][1] = 1;

	IIu[0][0] = 1;
	IIu[0][1] = 1;
	IIu[1][0] = 0;
	IIu[1][1] = 0;

	IIu.GetD() = 0.58;
	PERF_SAMPLE_BEGIN()

	C.BackPropagation(II, IIu);

	PERF_SAMPLE_END()

		SANITY_CHECK_NOTHING();
}

