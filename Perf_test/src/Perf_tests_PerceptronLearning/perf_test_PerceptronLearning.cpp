#include "opencv2/ts.hpp"
#include <fstream>
#include "PLearns.h"
#include "Functors.h"


PERF_TEST(PercetronLearning, BackPropagation_perf)
{
	D_Leaning C;
	Matrix<Weights<double>> II(100, 100);
    Weights<double > m(20,20);
    m.Fill(20);
    II.Fill(m);

	Weights<double> IIu(100, 100);

	IIu.GetD() = 0.58;
	PERF_SAMPLE_BEGIN()

	C.BackPropagation(II, IIu);

	PERF_SAMPLE_END()

	SANITY_CHECK_NOTHING();
}

