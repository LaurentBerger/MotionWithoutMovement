#include "opencv2/opencv.hpp"
#include<iostream>
#define  _USE_MATH_DEFINES
#include<math.h>


/* The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence 13 (9). pp. 891–906.  */
class SeparableSteerableFilter {
    std::vector<cv::Mat> fSepX,fSepY;
    std::vector<cv::Mat> hilbertX,hilbertY;
    cv::Point anchor;
    int derivativeOrder;

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
    cv::Mat Filter(cv::Mat,double angle);
    cv::Mat FilterHilbert(cv::Mat,double angle);
    std::vector<cv::Mat> GetFilter(double );
	std::vector<cv::Mat> GetFilterHilbert(double);
    double InterpolationFunction(int, double);
	double InterpolationFunctionHilbert(int, double);

};


using namespace std;
using namespace cv;

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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
        cout << hilbertX[1]<<endl;
        cout << hilbertY[1]<<endl;
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        fSepX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        fSepY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));
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
        hilbertX.push_back(u.col(0)*sqrt(s.at<float>(0,0)));
        hilbertY.push_back(v.row(0)*sqrt(s.at<float>(0,0)));

    }
    anchor = Point(-height/2,-height/2);

}


Mat SeparableSteerableFilter::Filter(Mat src,double angle)
{
    vector<Mat> v=GetFilter(angle);
    Mat f;

    sepFilter2D(src,f,CV_32F,v[0],v[1],anchor);
    return f;


}

Mat SeparableSteerableFilter::FilterHilbert(Mat src,double angle)
{
    vector<Mat> v=GetFilterHilbert(angle);
    Mat f;

    sepFilter2D(src,f,CV_32F,v[0],v[1],anchor);
    return f;


}

vector<Mat> SeparableSteerableFilter::GetFilter(double angle)
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

vector<Mat> SeparableSteerableFilter::GetFilterHilbert(double angle)
{
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


int main(int argc, char **argv)
{
    SeparableSteerableFilter g(4);
    Mat mc=imread("f:/lib/opencv/samples/data/detect_blob.png",CV_LOAD_IMAGE_COLOR);
    Mat mHSV;

    cvtColor(mc,mHSV,COLOR_BGR2YUV);
    vector<Mat> plan;

    split(mHSV,plan);
    Mat m = plan[0];

    double angle=0.7;
    Mat f = g.Filter(m,angle+M_PI/2);
    Mat e = g.Filter(m,angle);
    Mat o = g.FilterHilbert(m,angle);
    double T=5;
    double t=0;
    double minVal,maxVal;

    char c=0;
    while (c!=27)
    {
        
        Mat uc;

        Mat d = cos(2*M_PI/T*t)*e + sin(2*M_PI/T*t)*o-f;
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