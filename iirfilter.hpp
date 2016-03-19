#ifndef __IIRFILTER__
#define __IIRFILTER__

#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
namespace cv
{
namespace fir_iirfilter
{
enum TWindowType {wtFIRSTWINDOW, wtNONE, wtKAISER, wtSINC, wtHANNING,
				   wtHAMMING, wtBLACKMAN, wtFLATTOP, wtBLACKMAN_HARRIS,
				   wtBLACKMAN_NUTTALL, wtNUTTALL, wtKAISER_BESSEL, wtTRAPEZOID,
                   wtGAUSS, wtSINE, wtTEST };

enum TTransFormType {FORWARD, INVERSE};

enum TFIRPassTypes {firLPF, firHPF, firBPF, firNOTCH, ftNOT_FIR};

#define PI	    M_PI
#define TWOPI	    (2.0 * PI)
#define MAXORDER    10
#define MAXPOLES    (2*MAXORDER)    /* to allow for doubling of poles in BP filter */


#define opt_be 0x0001	/* -Be		Bessel cheracteristic	       */
#define opt_bu 0x0002	/* -Bu		Butterworth characteristic     */
#define opt_ch 0x0004	/* -Ch		Chebyshev characteristic       */

#define opt_lp 0x0008	/* -Lp		low-pass		       */
#define opt_hp 0x0010	/* -Hp		high-pass		       */
#define opt_bp 0x0020	/* -Bp		band-pass		       */

#define opt_a  0x0040	/* -a		alpha value		       */
#define opt_e  0x0100	/* -e		execute filter		       */
#define opt_l  0x0200	/* -l		just list filter parameters    */
#define opt_o  0x0400	/* -o		order of filter		       */
#define opt_p  0x0800	/* -p		specified poles only	       */
#define opt_w  0x1000	/* -w		don't pre-warp		       */

#define PARKS_BIG 1100          // Per the original code, this must be > 8 * MaxNumTaps = 8 * 128 = 1024
#define PARKS_SMALL 256         // This needs to be greater than or equal to MAX_NUM_PARKS_TAPS
#define MAX_NUM_PARKS_TAPS 127  // This was the limit set in the original code.
#define ITRMAX 50               // Max Number of Iterations. Some Notch and BPF are running ~ 43
#define MIN_TEST_VAL 1.0E-6     // Min value used in LeGrangeInterp and GEE

    /* H(z) =\frac{Y(z)}{X(z)}= \frac{\sum_{k=0}^{N}b_k z^{-k}}{\sum_{k=0}^{M}a_k z^{-k}} */
    /* y(n)=b_0x(n)+...b_N x(n-N)-a_1 y(n-1)-...-a_M y(n-M) */
class /*CV_EXPORTS_W*/  FIR_IIRFilter
{
private:
    int order, numpoles;
    double raw_alpha1, raw_alpha2;
    std::complex<double>  dc_gain, fc_gain, hf_gain;
    unsigned int opts;	/* option flag bits */

    double warped_alpha1, warped_alpha2, chebrip;
    unsigned int polemask;
    bool optsok;

    std::complex<double>  spoles[MAXPOLES];
    std::complex<double>  zpoles[MAXPOLES], zzeros[MAXPOLES];
    double xcoeffs[MAXPOLES+1], ycoeffs[MAXPOLES+1];
    std::vector <double> a; // denominator
    std::vector <double> b; // Numerator
    int n;                  // Filter order y= bx-ay Y=B/A

    int HalfTapCount, ExchangeIndex[PARKS_SMALL];
    double LeGrangeD[PARKS_SMALL], Alpha[PARKS_SMALL], CosOfGrid[PARKS_SMALL], DesPlus[PARKS_SMALL];
    double Coeff[PARKS_SMALL], Edge[PARKS_SMALL], BandMag[PARKS_SMALL], InitWeight[PARKS_SMALL];
    double DesiredMag[PARKS_BIG], Grid[PARKS_BIG], Weight[PARKS_BIG];

    double fLow,fHigh;

    void CalcParkCoeff2(int NumBands, int TapCount, double *FirCoeff);
    int Remez2(int GridIndex);
    double LeGrangeInterp2(int K, int N, int M);// D
    double GEE2(int K, int N);
    bool ErrTest(int k, int Nut, double Comp, double *Err);
    void CalcCoefficients(void);
    void WindowData(double *Data, int N, TWindowType WindowType, double Alpha, double Beta, bool UnityGain);
    // This gets used with the Kaiser window.
    double Bessel(double x);
    double Sinc(double x);

    std::complex<double>   eval(std::complex<double>   coeffs[], int np, std::complex<double>   z);

    std::complex<double>  evaluate(std::complex<double>  topco[],std::complex<double>  botco[], int np, std::complex<double>  z);
    void choosepole(std::complex<double>  z);
    void  compute_s() ;

    void normalize();
    void compute_z();
    void multin(std::complex<double>  w, std::complex<double>  coeffs[]);
    void expand(std::complex<double>  pz[],std::complex<double>   coeffs[]);
    void expandpoly();
    void printrecurrence(); /* given (real) Z-plane poles & zeros, compute & print recurrence relation */

public :
    /*** IIR filter */
    FIR_IIRFilter(std::string filterType,int order,double fs,std::vector<double> frequency);
    /** FIR filter using Parks-McClellan algorithm*/
    FIR_IIRFilter(double *FirCoeff, int NumTaps, TFIRPassTypes PassType, double OmegaC, double BW, double ParksWidth, TWindowType WindowType, double WinBeta);

    Mat McClellanTransform(Mat fir1d,std::vector<double> abcde,int nbPts=256);
    Mat UnitaryFilter(Mat l);
    std::vector<Mat> OptimizeUnitaryFilter(double cufoffFrequency,int nbTap);

};
}
}

#endif
