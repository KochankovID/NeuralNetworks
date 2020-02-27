#include "opencv2/ts.hpp"
#include <fstream>
#include "PLearns.h"
#include "Functors.h"

class Sigm : public D_Func
{
public:
    Sigm(const double& a_) : D_Func(), a(a_) {};
    double a;

    double operator()(const double& x) {
        double f = 1;
        const double e = 2.7182818284;
        if (x >= 0) {
            f = pow(1/e, x*a);
        }
        else {
            f = pow(e, abs(a*x));
        }
        f++;
        return 1 / f;
    }
    ~Sigm() {};
};

// Производная сигмоиды
class SigmD : public Sigm
{
public:
    SigmD(const double& a_) : Sigm(a_) {};
    double operator()(const double& x) {
        double f = 1;
        f = Sigm::operator()(x)*(1 - Sigm::operator()(x));
        return f;
    }
    ~SigmD() {};
};

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

PERF_TEST(PercetronLearning, GradDes_perf)
{
    D_Leaning C;
    Weights<double> II(100, 100);
    II.Fill(5);
    Matrix<double > m(100,100);
    m.Fill(20);
    SigmD g(1);
    II.GetD() = 0.58;
    PERF_SAMPLE_BEGIN()

        C.GradDes(II, m, g, 1);

    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}