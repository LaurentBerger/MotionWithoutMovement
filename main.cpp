#include "opencv2/opencv.hpp"
#include <iostream>
#include <algorithm>
#define  _USE_MATH_DEFINES
#include<math.h>
#include "NewParksMcClellan.h"

/* The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence 13 (9). pp. 891–906.  */
class SeparableSteerableFilter {
    std::vector<cv::Mat> fSepX,fSepY;
    std::vector<cv::Mat> hilbertX,hilbertY;
    std::vector<double> norme2fSepX;
    std::vector<double> norme2fSepY;
    cv::Point anchor;
    int derivativeOrder;

    cv::Mat l0;
    cv::Mat l1;
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
    cv::Mat Filter(cv::Mat,double angle);
    cv::Mat Filter(cv::Mat,int );
    cv::Mat InvFilter(cv::Mat src,double angle);
    cv::Mat InvFilter(cv::Mat src,int  );
    cv::Mat InvFilterL0(cv::Mat src);
    cv::Mat FilterL1(cv::Mat src);
    cv::Mat InvFilterL1(cv::Mat src);

    cv::Mat FilterHilbert(cv::Mat,double angle);
    std::vector<cv::Mat> GetFilterXY(double );
	std::vector<cv::Mat> GetFilterHilbertXY(double);
    double InterpolationFunction(int, double);
	double InterpolationFunctionHilbert(int, double);
	void     EstimateL0L1(int nbTap,int);

    void DisplayFilter();

};


using namespace std;
using namespace cv;

void SeparableSteerableFilter::DisplayFilter()
{
    cout << "//The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence 13 (9). pp. 891–906.  \n";
    for (int i = 0; i < fSepX.size(); i++)
    {
        cout << "//******************  Filter " << i << " ******************\n";
        cout << "Gx"<<i<<"="<< fSepX[i]/norme2fSepX[i]<<endl;
        cout << "Gy"<<i<<"="<< fSepY[i].t()/norme2fSepY[i]<<endl;
    }
    cout << "\n//***********************************************************\n";
    cout << "//****  QUADRATURE FILTER ***********\n";
    for (int i = 0; i < hilbertX.size(); i++)
    {
        cout << "//******************  Filter " << i << " ******************\n";
        cout << "Hx"<<i<<"="<< hilbertX[i]<<endl;
        cout << "yFilter"<<i<<"="<< hilbertY[i].t()<<endl;
    }
    cout << "Frequency cut off " << this->l0CutoffFrequency << "\n";
    cout << "L0="<<l0<<endl;;
    cout << "Frequency cut off " << this->l1CutoffFrequency << "\n";
	cout << "L1="<<l1<<endl;
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
}



SeparableSteerableFilter::SeparableSteerableFilter(int n,int height,double step)
{
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
	// Decomposition SVD du filtre K=USV' diag(s) sqrt(diags(s))u(:,1) vertical kernel and sqrt(diags(s))v(:,1)' horizontal kernel (
	// http://www.cs.toronto.edu/~urtasun/courses/CV/lecture02.pdf
    switch (n){
    case 2:
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= G2a(x,y);
            }
        }
        SVD::compute(filter,s,u,v);

        fSepY.push_back(u.col(0)/u.at<float>(height/2,0));
        fSepX.push_back(v.row(0)*u.at<float>(height/2,0)*s.at<float>(0,0));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
       for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= G2b(x,y);
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
                filter.at<float >(i,j)= G2c(x,y);
            }
        }
        SVD::compute(filter,s,u,v);
        fSepY.push_back(u.col(0)*v.at<float>(0,height/2)*s.at<float>(0,0));
        fSepX.push_back(v.row(0)/v.at<float>(0,height/2));
        norme2fSepX.push_back(norm(fSepX[fSepX.size()-1])*norm(fSepX[fSepX.size()-1]));
        norme2fSepY.push_back(norm(fSepY[fSepY.size()-1])*norm(fSepY[fSepY.size()-1]));
// Hilbert transform
        for (int i = 0; i < height; i++)
        {
            float y=static_cast<float>(i-height/2)*step;
            for (int j = 0; j < height; j++)
            {
                float x=static_cast<float>(j-height/2)*step;
                filter.at<float >(i,j)= H2a(x,y);
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
                filter.at<float >(i,j)= H2b(x,y);
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
                filter.at<float >(i,j)= H2c(x,y);
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
                filter.at<float >(i,j)= H2d(x,y);
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
                filter.at<float >(i,j)= G4a(x,y);
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
                filter.at<float >(i,j)= G4b(x,y);
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
                filter.at<float >(i,j)= G4c(x,y);
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
                filter.at<float >(i,j)= G4d(x,y);
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
                filter.at<float >(i,j)= G4e(x,y);
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
                filter.at<float >(i,j)= H4a(x,y);
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
                filter.at<float >(i,j)= H4b(x,y);
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
                filter.at<float >(i,j)= H4c(x,y);
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
                filter.at<float >(i,j)= H4d(x,y);
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
                filter.at<float >(i,j)= H4e(x,y);
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
                filter.at<float >(i,j)= H4f(x,y);
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
        flip(l1,iv0,1);
        sepFilter2D(src,f,CV_32F,iv0,iv0);
    }
    else
    {
        filter2D(src,f,CV_32F,l1);
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

    sepFilter2D(src,f,CV_32F,fSepX[i]/norme2fSepX[i],fSepY[i]/norme2fSepY[i],anchor);
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
    for (int i = 1; i < fSepX.size(); i++)
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
	bp = m > maxVal /2;
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
    l1CutoffFrequency=static_cast<double>(i)/bp.cols;
    i=maxPos.x;
    while (i <bp.cols/2 && bp.at<uchar>(0,i)==255)
        i++;
    if (i==bp.cols/2)
	{
		cv::Exception e;
		e.code = -3;
		e.msg = "Cannot find low pass filter L0!";
	}
    l0CutoffFrequency=static_cast<double>(i)/bp.cols;
    double *firCoef=new double[max(nbTapL0,nbTapL1)];
    TFIRPassTypes passType=firLPF;
    double omegac=l0CutoffFrequency;
    double bw=0;
    double parksWidth=0.1;
    TWindowType windowType=wtHANNING;

    NewParksMcClellan(firCoef, nbTapL0, passType, omegac*2, bw, parksWidth, windowType, 0);
    l0 = Mat::zeros(1,nbTapL0,CV_32FC1);
 
    for (int i=0;i<nbTapL0;i++)
        l0.at<float>(0, i) = firCoef[i];
    cout<< "l0="<<l0<<endl;
    l0 = Mat::zeros(nbTapL0,nbTapL0,CV_32FC1);
    for (int i=0;i<nbTapL0;i++)
        for (int j = 0; j < nbTapL0; j++)
        {
            int d = sqrt((i-nbTapL0/2)*(i-nbTapL0/2)+(j-nbTapL0/2)*(j-nbTapL0/2));
            if (d>nbTapL0/2)
                l0.at<float>(i, j) = 0;
            else
                l0.at<float>(i, j) = firCoef[d+nbTapL0/2];

        }

    omegac=l1CutoffFrequency;
    NewParksMcClellan(firCoef, nbTapL1, passType, omegac*2, bw, parksWidth, windowType, 0);
    l1 = Mat::zeros(1,nbTapL1,CV_32FC1);
    for (int i=0;i<nbTapL1;i++)
        l1.at<float>(0, i) = firCoef[i];
    l1 = Mat::zeros(nbTapL1,nbTapL1,CV_32FC1);
    for (int i=0;i<nbTapL1;i++)
        for (int j = 0; j < nbTapL1; j++)
        {
            int d = sqrt((i-nbTapL1/2)*(i-nbTapL1/2)+(j-nbTapL1/2)*(j-nbTapL1/2));
            if (d>nbTapL1/2)
                l1.at<float>(i, j) = 0;
            else
                l1.at<float>(i, j) = firCoef[d+nbTapL1/2];

        }
    delete firCoef;
// Analyse spectrale du système
    vector<Mat> kernel;
    if (l0.rows==1)
        kernel.push_back(l0.t()* l0);
    else
        kernel.push_back(l0);
    if (l1.rows==1)
        kernel.push_back(l1.t()* l1);
    else
        kernel.push_back(l1);
    for (int i = 0; i<fSepX.size();i++)
        kernel.push_back(fSepX[i].t()* fSepY[i].t());
    vector<Mat> spectre;
    spectre.resize(kernel.size());
    Mat response;
    for (int i = 0; i < kernel.size(); i++)
    {
	    Mat x = Mat::zeros(256, 256, CV_32F);
	    Mat y = Mat::zeros(256, 256, CV_32F);
	    Mat z;

	    Rect r(0, 0, kernel[i].cols, kernel[i].rows);
	    kernel[i].copyTo(x(r));
    FileStorage fs(format("kernel%d.yml",i), FileStorage::WRITE);

    fs << "Image" << kernel[i];
        vector<Mat> cplxPlane = {x,y};
        merge(cplxPlane,z);
        dft(z,spectre[i]);
        vector<Mat> p;
	    split(spectre[i],p);
	    Mat m;
	    magnitude(p[0],p[1],m);
	    minMaxLoc(m, &minVal, &maxVal,&minPos,&maxPos);
        cout << "spectrum "<<i<<"="<<minVal << " " << maxVal << "\n";
        
        if (i > 0)
        {
            if (i>=2)
            {
                maxVal *=1;
                norme2fSepX[i-2]=sqrt(maxVal);
                norme2fSepY[i-2]=sqrt(maxVal);
            }
            if (i == 1)
            {

                l1=l1/maxVal;
                m = m/maxVal;
            }
            if (response.empty())
                response=m;
            else
                response+=m/maxVal;

        }
        else
            l0=l0/maxVal;
        DisplayImage(m,format("Full spectrum %d",i));

    }
    FileStorage fs("fullSpectrume.yml.yml", FileStorage::WRITE);

    fs << "Image" << response;
	    minMaxLoc(response, &minVal, &maxVal,&minPos,&maxPos);
         cout << "Full spectrum "<<minVal << " " << maxVal << "\n";
       DisplayImage(response,"Full spectrum ");

}


int main(int argc, char **argv)
{
    {
    Mat mc=imread("f:/lib/opencv/samples/data/lena.jpg",CV_LOAD_IMAGE_GRAYSCALE),dst1,dst2;
    double minValmc,maxValmc;
    minMaxIdx(mc,&minValmc,&maxValmc);
    SeparableSteerableFilter g(2,9,0.35);
    vector<vector<Mat> > level(6);
    Mat m;
	g.EstimateL0L1(9,9);
    g.DisplayFilter();
    mc.convertTo(m,CV_32F);
    double minVal,maxVal;
    Mat mHigh,mLow;

    minMaxIdx(m,&minVal,&maxVal);
    cout <<"Original "<< minVal << "\t"<<maxVal<<endl;
    if (g.L0().rows==1)
        sepFilter2D(m,mLow,CV_32F,g.L0(),g.L0());
    else
        filter2D(m, mLow, CV_32F, g.L0());
    minMaxIdx(mLow,&minVal,&maxVal);
    cout << minVal << "\t"<<maxVal<<endl;
    mHigh=m-mLow;
    for (int i = 0; i < level.size(); i++)
    {
        Mat mH,mL;
        mL=g.FilterL1(mLow);
        mH=mLow-mL;
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
    for (int i =  level.size()-1; i >=0 ; i--)
    {
        minMaxIdx(mLow,&minVal,&maxVal);
        cout << "Level "<<i<<"\t"<<minVal << "\t"<<maxVal<<endl;
        resize(mLow, mLow, Size(),2,2,INTER_NEAREST);
        // sepFilter2D(mLow,mLow,CV_32F,g.L1(),g.L1());
        mLow=g.InvFilterL1(mLow);
        Mat s=( g.Filter(level[i][0],0)+ g.Filter(level[i][1],1)+g.Filter(level[i][2],2));
//        Mat s=level[i][0]+level[i][1]+level[i][2];
        mLow=(mLow+1.4*s);
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
    imshow("Original",mc);

    waitKey();
    return 0;
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