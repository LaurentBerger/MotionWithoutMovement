#include "opencv2/opencv.hpp"
#include <iostream>
#include <strstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#define  _USE_MATH_DEFINES
#include <math.h>
#include "iirfilter.hpp"


/* The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence 13 (9). pp. 891–906.  */
class SeparableSteerableFilter {
    std::vector<cv::Mat> fSepX,fSepY;
    std::vector<cv::Mat> hilbertX,hilbertY;
    std::vector<cv::Mat> filterRes;
    std::vector<cv::Mat> hilbertRes;
    cv::Mat c1,c2,c3;
    std::vector<double> norme2fSepX;
    std::vector<double> norme2fSepY;
    cv::Point anchor;
    int derivativeOrder;

    cv::Mat l0;
    cv::Mat h0;
    cv::Mat l1;
    cv::Mat h1;
    cv::Mat l0sym;
    cv::Mat l1sym;
    double l0CutoffFrequency;
    double l1CutoffFrequency;

private :
    /* d/dx2 gaussian*/
    double G2a(double x, double y)
    {
        return 0.9213*(2 * x*x - 1)*exp(-(x*x+y*y));
    }
    double G2b(double x, double y)
    {
        return 1.843*x*y*exp(-(x*x+y*y));
    }
    double G2c(double x, double y)
    {
        return 0.9213*(2 * y*y - 1)*exp(-(x*x+y*y));
    }
    /* Hilbert d/dx2 gaussian*/
    double H2a(double x, double y)
    {
        return 0.978*x*(-2.254+ x*x)*exp(-(x*x+y*y));
    }
    double H2b(double x, double y)
    {
        return 0.978*y*(-0.7515+ x*x)*exp(-(x*x+y*y));
    }
    double H2c(double x, double y)
    {
        return 0.978*x*(-0.7515+ y*y)*exp(-(x*x+y*y));
    }
    double H2d(double x, double y)
    {
        return 0.978*y*(-2.254+ y*y)*exp(-(x*x+y*y));
    }

    /* d/dx4 gaussian*/
    double G4a(double x, double y)
    {
        return 1.246*(0.75 + x*x*(-3+ +x*x))*exp(-(x*x+y*y));
    }
    double G4b(double x, double y)
    {
        return 1.246*x*(-1.5+x*x)*y*exp(-(x*x+y*y));
    }
    double G4c(double x, double y)
    {
        return 1.246*( y*y - 0.5)*( x*x - 0.5)*exp(-(x*x+y*y));
    }
    double G4d(double x, double y)
    {
        return 1.246*y*(-1.5+y*y)*x*exp(-(x*x+y*y));
    }
    double G4e(double x, double y)
    {
        return 1.246*(0.75 + y*y*(-3+ +y*y))*exp(-(x*x+y*y));
    }
    /* Hilbert de d/dx4 gaussian*/
    double H4a(double x, double y)
    {
        return 0.3975*x*(7.189+x*x*(x*x-7.501))*exp(-(x*x+y*y));
    }
    double H4b(double x, double y)
    {
        return 0.3975*(1.438+x*x*(x*x-4.501))*y*exp(-(x*x+y*y));
    }
    double H4c(double x, double y)
    {
        return 0.3975*x*(x*x-2.225)*(y*y-0.663)*exp(-(x*x+y*y));
    }
    double H4d(double x, double y)
    {
        return 0.3975*y*(y*y-2.225)*(x*x-0.663)*exp(-(x*x+y*y));
    }
    double H4e(double x, double y)
    {
        return 0.3975*(1.438+y*y*(y*y-4.501))*x*exp(-(x*x+y*y));
    }
    double H4f(double x, double y)
    {
        return 0.3975*y*(7.189+y*y*(y*y-7.501))*exp(-(x*x+y*y));
    }
    
	

public :
    SeparableSteerableFilter(int n,int height=9,double step=0.67);
    cv::Mat L0(){return l0;};
    cv::Mat L1(){return l1;};
    cv::Mat H0(){return h0;};
    cv::Mat H1(){return h1;};
    cv::Mat Filter(cv::Mat,double angle);
    cv::Mat Filter(cv::Mat,int );
    cv::Mat InvFilter(cv::Mat src,double angle);
    cv::Mat InvFilter(cv::Mat src,int  );
    cv::Mat FilterL1(cv::Mat src);
    cv::Mat FilterH1(cv::Mat src);
    cv::Mat FilterL0(cv::Mat src);
    cv::Mat InvFilterL0(cv::Mat src);
    cv::Mat FilterH0(cv::Mat src);
    cv::Mat InvFilterH0(cv::Mat src);
    cv::Mat InvFilterH1(cv::Mat src);
    cv::Mat InvFilterL1(cv::Mat src);

    cv::Mat FilterHilbert(cv::Mat,double angle);
    std::vector<cv::Mat> GetFilterXY(double );
	std::vector<cv::Mat> GetFilterHilbertXY(double);
    double InterpolationFunction(int, double);
	double InterpolationFunctionHilbert(int, double);
	void     EstimateL0L1(int nbTap,int);
    std::vector<cv::Mat> LocalOrientation(cv::Mat );

    void DisplayFilter();

};


using namespace std;
using namespace cv;


void DisplayImage(Mat x,string s)
{
	vector<Mat> sx;
	split(x, sx);
	vector<double> minVal(3), maxVal(3);
	for (int i = 0; i < sx.size(); i++)
	{
		minMaxLoc(sx[i], &minVal[i], &maxVal[i]);

	}
	maxVal[0] = *max_element(maxVal.begin(), maxVal.end());
	minVal[0] = *min_element(minVal.begin(), minVal.end());
	Mat uc;
	x.convertTo(uc, CV_8U,255/(maxVal[0]-minVal[0]),-255*minVal[0]/(maxVal[0]-minVal[0]));
	imshow(s, uc);
	waitKey(); 

}

void SeparableSteerableFilter::DisplayFilter()
{
    ostrstream s;
    s << "//The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence 13 (9). pp. 891–906.  \n";
    for (int i = 0; i < fSepX.size(); i++)
    {
        s << "//******************  Filter " << i << " ******************\n";
        s << "Gx"<<i<<"="<< fSepX[i]/norme2fSepX[i]<<";"<<endl;
        s << "Gy"<<i<<"="<< fSepY[i].t()/norme2fSepY[i]<<";"<<endl;
    }
    s << "\n//***********************************************************\n";
    s << "//****  QUADRATURE FILTER ***********\n";
    for (int i = 0; i < hilbertX.size(); i++)
    {
        s << "//******************  Filter " << i << " ******************\n";
        s << "Hx"<<i<<"="<< hilbertX[i]<<";"<<endl;
        s << "yFilter"<<i<<"="<< hilbertY[i].t()<<";"<<endl;
    }
    s << "Frequency_cut_off= " << this->l0CutoffFrequency <<";"<< "\n";
    s << "L0="<<l0<<";"<<endl;;
    s << "H0="<<h0<<";"<<endl;;
    s << "Frequency_cut_off= " << this->l1CutoffFrequency <<";"<< "\n";
	s << "L1="<<l1<<";"<<endl;
	s << "H1="<<h1<<";"<<endl;
    string a(s.str(),s.pcount());
    cout << a;
    ofstream f("filter.sce");
    f<<a;
    f.flush();
    s.freeze(false);
    waitKey();
}



double SeparableSteerableFilter::InterpolationFunction(int n, double angle)
{
	if (n > derivativeOrder)
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "n must be less than polyOrder!";
	}
	switch (derivativeOrder)
	{
	case 2:
		switch (n) {
		case 0:
			return pow(cos(angle), 2.0);
			break;
		case 1:
			return -2 * cos(angle)*sin(angle);
			break;
		case 2:
			return sin(angle)*sin(angle);
			break;
		}
		break;
	case 4:
		switch (n) {
		case 0:
			return pow(cos(angle), 4.);
			break;
		case 1:
			return -4 * pow(cos(angle), 3.)*sin(angle);
			break;
		case 2:
			return 6 * pow(cos(angle)*sin(angle), 2.0);
			break;
		case 3:
			return -4*cos(angle)*pow(sin(angle), 3.);
			break;
		case 4:
			return pow(sin(angle), 4.);
			break;
		}
		break;
	default:
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "n must be less than polyOrder!";
	}
	}
    return 0 ;
}

double SeparableSteerableFilter::InterpolationFunctionHilbert(int n, double angle)
{
	if (n > derivativeOrder+1)
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "n must be less than polyOrder+1!";
	}
	switch (derivativeOrder)
	{
	case 2:
		switch (n) {
		case 0:
			return pow(cos(angle), 3.0);
			break;
		case 1:
			return -3 *cos(angle)* cos(angle)*sin(angle);
			break;
		case 2:
			return 3*cos(angle)*sin(angle)*sin(angle);
			break;
		case 3:
			return -pow(sin(angle), 3.);
			break;
		}
		break;
	case 4:
		switch (n) {
		case 0:
			return pow(cos(angle), 5.);
			break;
		case 1:
			return -5 * pow(cos(angle), 4.)*sin(angle);
			break;
		case 2:
			return 10 * pow(cos(angle)*sin(angle), 2.0)*cos(angle);
			break;
		case 3:
			return -10 * sin(angle)*pow(sin(angle)*cos(angle), 2.);
			break;
		case 4:
			return 5*cos(angle)*pow(sin(angle), 4.);
			break;
		case 5:
			return -pow(sin(angle), 5.);
			break;
		}
		break;
	default:
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "n must be less than polyOrder!";
	}
	}
return 0;
}



SeparableSteerableFilter::SeparableSteerableFilter(int n,int height,double stepxy)
{
    float step=static_cast<float>(stepxy);
    if (n!=2 && n!=4)
	{	
		cv::Exception e;
        e.code = -2;
        e.msg = "polynomial order not implemented!";
    }
    if (height%2==0)
    {
        cv::Exception e;
        e.code = -2;
        e.msg = "filter size must be even";
    }
	derivativeOrder =n;
    Mat filter(height,height,CV_32F);
    Mat u,v,s;
    Mat filterX,filterY;
    vector<Mat> freq(2);
    vector<Mat> z;
    Mat         freqMod,zeros,imCplx;
    double minV,maxV;
	// Decomposition SVD du filtre K=USV' diag(s) sqrt(diags(s))u(:,1) vertical kernel and sqrt(diags(s))v(:,1)' horizontal kernel (
	// http://www.cs.toronto.edu/~urtasun/courses/CV/lecture02.pdf
    switch (n){
    case 2:
        zeros = Mat::zeros(height,height,CV_32FC1);
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G2a(x,y));
            }
        }
        cout << "G0="<<filter<<endl;
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)/u.at<float>(height/2,0));
        fSepX.push_back(v.row(0)*u.at<float>(height/2,0)*s.at<float>(0,0));
        freq[0]=filter;
        freq[1]=zeros;
        merge(freq,imCplx);
        dft(imCplx,freqMod);
        split(freqMod,z);
        magnitude(z[0], z[1], freqMod);
        minMaxLoc(freqMod,&minV,&maxV);
        norme2fSepX.push_back((maxV));
        norme2fSepY.push_back(1);
       for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G2b(x,y));
            }
        }
        cout << "G1="<<filter<<endl;
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        freq[0]=filter;
        freq[1]=zeros;
        merge(freq,imCplx);
        dft(imCplx,freqMod);
        split(freqMod,z);
        magnitude(z[0], z[1], freqMod);
        minMaxLoc(freqMod,&minV,&maxV);
        norme2fSepX.push_back((maxV));
        norme2fSepY.push_back(1);
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G2c(x,y));
            }
        }
        cout << "G2="<<filter<<endl;
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*v.at<float>(0,height/2)*s.at<float>(0,0));
        fSepX.push_back(v.row(0)/v.at<float>(0,height/2));
        freq[0]=filter;
        freq[1]=zeros;
        merge(freq,imCplx);
        dft(imCplx,freqMod);
        split(freqMod,z);
        magnitude(z[0], z[1], freqMod);
        minMaxLoc(freqMod,&minV,&maxV);
        norme2fSepX.push_back((maxV));
        norme2fSepY.push_back(1);
// Hilbert transform
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H2a(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H2b(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H2c(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H2d(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        break;
    case 4:
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G4a(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G4b(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G4c(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G4d(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(G4e(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
// Hilbert transform
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4a(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4b(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4c(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4d(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4e(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= static_cast<float>(H4f(x,y));
            }
        }
        SVD::compute(filter,s,u,v);
        hilbertY.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertX.push_back(v.row(0)*sqrt(s.at<float>(0,0)));

    }
    anchor = Point(-height/2,-height/2);

}

Mat SeparableSteerableFilter::InvFilterL0(Mat src)
{
    Mat f,iv0;

    if (l0.rows==1)
    {
        flip(l0,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,l0);
    }

    return f;


}

Mat SeparableSteerableFilter::FilterL0(Mat src)
{
    Mat f,iv0;

    if (l0.rows==1)
    {
        sepFilter2D(src,f,CV_32F,l0,l0);
    }
    else
    {
        filter2D(src,f,CV_32F,l0);
    }

    return f;


}

Mat SeparableSteerableFilter::InvFilterL1(Mat src)
{
    Mat f,iv0;

    if (l1.rows==1)
    {
        flip(l1,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,l1);
    }
    return f;


}


Mat SeparableSteerableFilter::FilterL1(Mat src)
{
    Mat f,iv0;

    if (l1.rows==1)
    {
        sepFilter2D(src,f,CV_32F,l1,l1);
    }
    else
    {
        filter2D(src,f,CV_32F,l1);
    }
    return f;


}


Mat SeparableSteerableFilter::InvFilterH1(Mat src)
{
    Mat f,iv0;

    if (h1.rows==1)
    {
        flip(h1,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,h1);
    }
    return f;


}


Mat SeparableSteerableFilter::FilterH1(Mat src)
{
    Mat f,iv0;

    if (h1.rows==1)
    {
        flip(h1,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,h1);
    }
    return f;


}

Mat SeparableSteerableFilter::InvFilterH0(Mat src)
{
    Mat f,iv0;

    if (h1.rows==1)
    {
        flip(h0,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,h0);
    }
    return f;


}


Mat SeparableSteerableFilter::FilterH0(Mat src)
{
    Mat f,iv0;

    if (h0.rows==1)
    {
        flip(h0,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,h0);
    }
    return f;


}




/* http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2014/bsb663/slides/w08-pyramids.pdf */
Mat SeparableSteerableFilter::InvFilter(Mat src,double angle)
{
    vector<Mat> v=GetFilterXY(angle);
    Mat f,iv0,iv1;

    flip(v[0],iv0,1);
    flip(v[1],iv1,1);
    sepFilter2D(src,f,CV_32F,iv0,iv1,anchor);
    return f;


}

/* http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2014/bsb663/slides/w08-pyramids.pdf */
Mat SeparableSteerableFilter::InvFilter(Mat src,int i)
{
    Mat f,iv0,iv1;

    flip(fSepX[i]/norme2fSepX[i],iv0,1);
    flip(fSepY[i]/norme2fSepY[i],iv1,-1);
    sepFilter2D(src,f,CV_32F,iv0,iv1,anchor);
//    sepFilter2D(src,f,CV_32F,fSepY[i],fSepX[i],anchor);
    return f;


}

Mat SeparableSteerableFilter::Filter(Mat src,double angle)
{
    vector<Mat> v=GetFilterXY(angle);
    Mat f;

    sepFilter2D(src,f,CV_32F,v[0],v[1],anchor);
    return f;


}

Mat SeparableSteerableFilter::Filter(Mat src,int i)
{
    Mat f;

    sepFilter2D(src,f,CV_32F,fSepX[i]/norme2fSepX[i],fSepY[i].t()/norme2fSepY[i]);
    return f;


}

Mat SeparableSteerableFilter::FilterHilbert(Mat src,double angle)
{
    vector<Mat> v=GetFilterHilbertXY(angle);
    Mat f;

    sepFilter2D(src,f,CV_32F,v[0],v[1],anchor);
    return f;


}

vector<Mat> SeparableSteerableFilter::GetFilterXY(double angle)
{
	Mat x,y;

    x=InterpolationFunction(0,angle)*fSepX[0];
    y=InterpolationFunction(0,angle)*fSepY[0];
    for (int i = 1; i < fSepX.size(); i++)
    {
        double k=InterpolationFunction(i,angle);
        x = x +k*fSepX[i];
        y = y +k*fSepY[i];

    }
    vector<Mat> v(2);
    v[0]=x;
    v[1]=y;

	return v;
}

vector<Mat> SeparableSteerableFilter::GetFilterHilbertXY(double angle)
{
	// A REVOIR INCOHERENCE ENTRE HILBERT ET FSEPX
	Mat x,y;

    x=InterpolationFunctionHilbert(0,angle)*hilbertX[0];
    y=InterpolationFunctionHilbert(0,angle)*hilbertY[0];
    for (int i = 1; i < hilbertX.size(); i++)
    {
        double k=InterpolationFunctionHilbert(i,angle);
        x = x +k*hilbertX[i];
        y = y +k*hilbertY[i];

    }
    vector<Mat> v(2);
    v[0]=x;
    v[1]=y;

	return v;
}

vector<cv::Mat> SeparableSteerableFilter::LocalOrientation(cv::Mat mc)
{
	Mat m,m1;
	if (L0().rows == 1)
		sepFilter2D(mc, m1, CV_32F,L0(), L0());
	else
		filter2D(mc, m1, CV_32F, L0());
    DisplayImage(m1, "L0 filter");
	if (L1().rows == 1)
		sepFilter2D(m1, m, CV_32F,L1(), L1());
	else
		filter2D(m1, m, CV_32F, L1());
    DisplayImage(m, "L1*L0 filter");
 	if (h1.rows == 1)
		sepFilter2D(m1, m, CV_32F,h1, h1);
	else
		filter2D(m1, m, CV_32F,h1);
    //m=m1-m;
    DisplayImage(m, "BP(L1*L0) filter");
    filterRes.clear();
    hilbertRes.clear();
    if (derivativeOrder != 2)
    {
		cv::Exception e;
		e.code = -2;
		e.msg = "Local orientation supported for derivativeOrder=2 only!";
    }
    for (int i=0;i<fSepX.size();i++)
    {
        filterRes.push_back(Filter(m,i));
        DisplayImage(filterRes[i], format("g2%d",i));
    }
    for (int i=0;i<hilbertX.size();i++)
    {
        hilbertRes.push_back(FilterHilbert(m,i));
    }
    Mat g2a2=filterRes[0].mul(filterRes[0]);
    Mat g2b2=filterRes[1].mul(filterRes[1]);
    Mat g2c2=filterRes[2].mul(filterRes[2]);
    Mat h2a2=hilbertRes[0].mul(hilbertRes[0]);
    Mat h2b2=hilbertRes[1].mul(hilbertRes[1]);
    Mat h2c2=hilbertRes[2].mul(hilbertRes[2]);
    Mat h2d2=hilbertRes[3].mul(hilbertRes[3]);
    Mat h2ac=hilbertRes[0].mul(hilbertRes[2]);
    Mat h2bd=hilbertRes[1].mul(hilbertRes[3]);

    c1 = 0.5*g2b2 + 0.25*filterRes[0].mul(filterRes[2]);
    c1 = c1 + 0.375*(g2a2+g2c2);
    c1 = c1 + 0.3125*(h2a2+h2d2);
    c1 = c1 + 0.5625*(h2b2+h2c2);
    c1 = c1 + 0.375*(h2ac+h2bd);
    c2 = 0.5*(g2a2 - g2c2) + 0.46875*(h2a2 - h2d2) + 0.28125*(h2b2-h2c2);
    c2+=0.1875*(h2ac-h2bd);
    c3 = -filterRes[0].mul(filterRes[1]) - filterRes[1].mul(filterRes[2]);
    c3 += - 0.9375*(hilbertRes[2].mul(hilbertRes[3])+hilbertRes[0].mul(hilbertRes[1]));
    c3 += - 1.6875*hilbertRes[1].mul(hilbertRes[2])-0.1875*hilbertRes[0].mul(hilbertRes[3]);

	vector<Mat> r;
	Mat modulus = c2.mul(c2) + c3.mul(c3);
	sqrt(modulus, modulus);
	r.push_back(modulus);
	return r;
}




void     SeparableSteerableFilter::EstimateL0L1(int nbTapL0,int nbTapL1)
{
	if (fSepX.size() == 0)
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "No filter initialize!";

	}
	int i = max(3, 4);
	Mat x = Mat::zeros(1, getOptimalDFTSize(max(max(nbTapL0,nbTapL1) *8, static_cast<int> (fSepX[0].cols * 4))), CV_32F);
	Mat y = Mat::zeros(1, getOptimalDFTSize(max(max(nbTapL0,nbTapL1) *8, static_cast<int> (fSepX[0].cols * 4))), CV_32F);
	
	Rect r(0, 0, fSepX[0].cols, 1);
	fSepX[0].copyTo(x(r));
	Mat s;
	vector<Mat> p = { x,y };
	merge(p, s);

	Mat fftx;
	dft(s,fftx);
	split(fftx, p);
	Mat m;
	magnitude(p[0],p[1],m);
	double minVal, maxVal;
    Point minPos,maxPos;
	minMaxLoc(m, &minVal, &maxVal,&minPos,&maxPos);
    if (maxPos.x>m.cols/2)
        maxPos.x=m.cols-maxPos.x;
	Mat bp;
	bp = m > maxVal /1.5;
    // low frequency cut off
    i=maxPos.x;
    while (i >= 0 && bp.at<uchar>(0,i)==255)
        i--;
    if (i==-1)
	{
		cv::Exception e;
		e.code = -3;
		e.msg = "Cannot find low pass filter L1!";
	}
    l1CutoffFrequency=static_cast<float>(i)/bp.cols;
    i=maxPos.x;
    while (i <bp.cols/2 && bp.at<uchar>(0,i)==255)
        i++;
    if (i==bp.cols/2)
	{
		cv::Exception e;
		e.code = -3;
		e.msg = "Cannot find low pass filter L0!";
	}
    l0CutoffFrequency=static_cast<float>(i)/bp.cols;
    vector<double> firCoefL0(nbTapL0);
    cv::fir_iirfilter::TFIRPassTypes passType=cv::fir_iirfilter::firLPF;
    double omegac=l0CutoffFrequency;
    double bw=0;
    double parksWidth=0.1;
    cv::fir_iirfilter::TWindowType windowType=cv::fir_iirfilter::wtKAISER;


    cv::fir_iirfilter::FIR_IIRFilter f(firCoefL0.data(), nbTapL0, passType, omegac*2, bw,  1./nbTapL0, windowType, 2.41);
    vector<Mat> pp = f.OptimizeUnitaryFilter(l0CutoffFrequency,nbTapL0);
    Mat l01d = Mat(firCoefL0);
    l01d = l01d / sum(l01d)[0];
    vector<double> al = {-0.5,0.5,0.5, 0.25, 0.25};
    vector<double> ah = {-0.5,0.5,0.5, 0.25, 0.25};
    cout<<"l0="<<pp[0]<<endl;
    cout<<"h0="<<pp[1]<<endl;
    l0= f.McClellanTransform(pp[0].t(),al);
    h0= f.McClellanTransform(pp[1].t(),ah);

    omegac=l1CutoffFrequency;
    cout<<"l0="<<pp[0]<<endl;
    cout<<"h0="<<pp[1]<<endl;
    pp = f.OptimizeUnitaryFilter(l1CutoffFrequency,nbTapL1);
    cout<<"l1="<<pp[0]<<endl;
    cout<<"h1="<<pp[1]<<endl;
    l1=f.McClellanTransform(pp[0].t(),al);
    h1=f.McClellanTransform(pp[1].t(),ah);
    cout << h1 << "\n";

}



int main(int argc, char **argv)
{
 

    
    
    {
//    Mat mc=imread("f:/lib/einstein.pgm",CV_LOAD_IMAGE_GRAYSCALE),dst1,dst2;
    Mat mc=imread("f:/lib/opencv/samples/data/lena.jpg",CV_LOAD_IMAGE_GRAYSCALE);
/*    mc = Mat::zeros(512,512,CV_8SC1);
    for (int i = 0; i < 512; i++)
    {
        uchar *uc = mc.ptr(i);
        for (int j = 0; j < 512; j++,uc++)
        {
            double d = (i - 256)*(i - 256) + (j-256)*(j-256);
            d = sqrt(d)/1;
            if (d==0)
                *uc=127;
            else
                *uc = static_cast<uchar>(sin(d)/d*127);
        }
    }*/
    double minValmc,maxValmc;
    minMaxIdx(mc,&minValmc,&maxValmc);
    int order=2;
    SeparableSteerableFilter g(order,9,0.67);
    vector<vector<Mat> > level(6);
    Mat m;
	g.EstimateL0L1(9,9);



	g.DisplayFilter();
    mc.convertTo(m,CV_32F);
    Mat mcH;
    mcH = g.InvFilterH1(g.FilterH1(m))+g.InvFilterL1(g.FilterL1(m));
    Mat me;
    mcH.convertTo(me,CV_8U);
    DisplayImage(m, "original");
    DisplayImage(mcH, "original gL1");
    cout << "PSNR = " << PSNR(mc,me)<<"\n";
    mcH = g.InvFilterL0(g.FilterL0(m))+g.InvFilterH0(g.FilterH0(m));
    mcH.convertTo(me,CV_8U);
    DisplayImage(mcH, "original gL0");
    DisplayImage(g.FilterL0(m), "L0");
    DisplayImage(g.FilterL1(m), "L1");
    cout << "PSNR = " << PSNR(mc,me)<<"\n";
    double minVal,maxVal;
    absdiff(m,mcH,m);
    minMaxIdx(m,&minVal,&maxVal);
    cout << minVal << " "<<maxVal<<" "<<mean(m)[0]<<endl;
    Mat mHigh,mLow;
    for (int i=0;i<=order;i++)
    {
        DisplayImage(g.Filter(m,i), format("g2%d",i));
    }
	vector<Mat> w = g.LocalOrientation(mc);

	DisplayImage(w[0], "LO");


	g.DisplayFilter();

    mc.convertTo(m,CV_32F);
    minMaxIdx(m,&minVal,&maxVal);
    cout <<"Original "<< minVal << "\t"<<maxVal<<endl;
    mLow =g.FilterL0(m);
    minMaxIdx(mLow,&minVal,&maxVal);
    cout << minVal << "\t"<<maxVal<<endl;
    mHigh=g.FilterH0(m);
    for (int i = 0; i < level.size(); i++)
    {
        Mat mH,mL;
        mL=g.FilterL1(mLow);
        mH=g.FilterH1(mLow);
        level[i].push_back(g.InvFilter(mH,0));
        level[i].push_back(g.InvFilter(mH,1));
        level[i].push_back(g.InvFilter(mH,2));
        if (i==0)
            level[i].push_back(mHigh);
        resize(mL, mLow, Size(),0.5,0.5,INTER_NEAREST);
        minMaxIdx(mLow,&minVal,&maxVal);
        cout << "Level "<<i<<"\t"<<minVal << "\t"<<maxVal<<endl;
        DisplayImage(mLow, "Reduce");
    }
    minMaxIdx(mLow,&minVal,&maxVal);
    cout << minVal << "\t"<<maxVal<<endl;
    for (int i =  static_cast<int>(level.size())-1; i >=0 ; i--)
    {
        minMaxIdx(mLow,&minVal,&maxVal);
        cout << "Level "<<i<<"\t"<<minVal << "\t"<<maxVal<<endl;
        resize(mLow, mLow, Size(),2,2,INTER_NEAREST);
        // sepFilter2D(mLow,mLow,CV_32F,g.L1(),g.L1());
        mLow=g.InvFilterL1(mLow);
        Mat s=g.InvFilterH1( g.Filter(level[i][0],0)+ g.Filter(level[i][1],1)+g.Filter(level[i][2],2));
        mLow=(mLow+s);
        DisplayImage(mLow, "collapse");
    }
    //mLow = g.InvFilterL0(mLow);
    mLow = mLow + level[0][3];
    minMaxIdx(mLow,&minVal,&maxVal);
    DisplayImage(mLow, "collapse");
    Mat mu;
    mLow.convertTo(mu,CV_8U,maxValmc/(maxVal-minVal),-minValmc/(maxVal-minVal));
    imshow("collapse U",mu);
    absdiff(mu,mc,mu);
    minMaxIdx(mu,&minVal,&maxVal);
    cout << "error "<<minVal << "\t"<<maxVal<<endl;
    imshow("Error",mu);
    imshow("Original",mc);

    waitKey();
    }





    SeparableSteerableFilter g(4,15);
    SeparableSteerableFilter g2(4,11);
    g.DisplayFilter();
//    Mat mc=imread("f:/lib/opencv/samples/data/detect_blob.png",CV_LOAD_IMAGE_COLOR);
    Mat mc=imread("f:/lib/opencv/samples/data/lena.jpg",CV_LOAD_IMAGE_COLOR);
    Mat mask=Mat::zeros(mc.size(),CV_8U);
    Rect r(0,0,mask.cols,mask.rows/2);
    mask(r)=255;

    Mat mask2;

    bitwise_not(mask,mask2);

    Mat mHSV;

    cvtColor(mc,mHSV,COLOR_BGR2YUV);
    vector<Mat> plan;

    split(mHSV,plan);
    Mat m = plan[0];

    double angle=0.7;
    double angle2=-angle;
    Mat f = g.Filter(m,angle+M_PI/2);
    Mat e = g.Filter(m,angle);
    Mat o = g.FilterHilbert(m,angle);
    Mat f2 = g2.Filter(m,angle2+M_PI/2);
    Mat e2 = g2.Filter(m,angle2);
    Mat o2 = g2.FilterHilbert(m,angle2);
    double T=5;
    double t=0;
    double minVal,maxVal;

    char c=0;
    while (c!=27)
    {
        
        Mat uc;

        Mat d = cos(2*M_PI/T*t)*e + sin(2*M_PI/T*t)*o-f;
        Mat d2 = cos(6*M_PI/T*t)*e2 + sin(6*M_PI/T*t)*o2-f2;
        d2(r).copyTo(d(r));
        minMaxIdx(d,&minVal,&maxVal);

        d.convertTo(uc,CV_8U,255/(maxVal-minVal),-255*minVal/(maxVal-minVal));
        plan[0]=uc;
        merge(plan,mHSV);
        cvtColor(mHSV,mc,COLOR_YUV2BGR);
        imshow("Motion without movement",mc);
        t=t+0.25;
        c= waitKey(50);
        if (c == '+')
            T *=1.05;
        if (c == '-')
            T /=1.05;
    }


}