#pragma once
#include <cmath>
/*****************************************
* CubicHermiteSpline class
* 
* Cubic Hermite Spline Interpolation
* 三次エルミート補間を実現する
* ただし、一階微分値に関しては、二次精度の値を用いる
* 
********************************************/


class FifthSpline {
public: 
    enum class EDGE_DIFF_TYPE :int
    {
        ALWAYS_ZERO = 0,            //dv/dx[0] is regarded as 0
        DIFF_TO_2ND_POINT = 1,     //dv/dx[0] is regarded as (v[1]-v[0])/dx
    };

private:
    int m_num_grids;
    double m_x_begin;
    double m_x_end;
    double* values = nullptr;
    EDGE_DIFF_TYPE m_edge_difference_mode;

    double ValueOnPoint(int i) {
        if (i < 0) {
            return values[0];
        } else if (i >= m_num_grids) {
            return values[m_num_grids - 1];
        } else {
            return values[i];
        }
    }

    // dv/dx * dx 
    double DifferenceOnPoint(int i) {
        if (i <= 0) {
            if (m_edge_difference_mode == EDGE_DIFF_TYPE::ALWAYS_ZERO) {
                return 0.0;
            }else{
                return (values[1] - values[0]);
            }
        } else if (i >= m_num_grids-1) {
            return values[m_num_grids - 1] - values[m_num_grids - 2];
        } else {
            return (values[i+1] - values[i-1])/2.0;
        }
    }
    // d2v/dx2 * dx*dx 
    double SecondOnPoint(int i) {
        if (i <= 0) {
            if (m_edge_difference_mode == EDGE_DIFF_TYPE::ALWAYS_ZERO) {
                return 0.0;
            } else {
                return (values[2] -2.0* values[1] + values[0]);
            }
        } else if (i >= m_num_grids - 1) {
            return values[m_num_grids - 1] - 2.0*values[m_num_grids - 2]+ values[m_num_grids - 3];
        } else {
            return (values[i + 1] -2.0* values[i] + values[i - 1]);
        }
    }

public:
    
    ~FifthSpline(){
        delete[] values; 
    }


    template<class FUNC>
    void Initialize(int N, double x_begin, double x_end, EDGE_DIFF_TYPE edge_difference_mode, FUNC func) {

        m_num_grids = N;
        m_x_begin = x_begin;
        m_x_end = x_end;
        m_edge_difference_mode = edge_difference_mode;
        values = new double[N];


        const double dx = (m_x_end - m_x_begin) / (double)(m_num_grids - 1);
        for (int i = 0; i < m_num_grids; ++i) {
            double x = dx * (double)(i)+m_x_begin;    
            values[i] = func(x);
        }
    }

    double Interpolate(double x, double* res_dp_dx) {
        const double dx = (m_x_end - m_x_begin) / (double)(m_num_grids - 1);
        const double nx = (x - m_x_begin)/dx;
        const int i = (int)floor(nx);
        const double t = nx - floor(nx);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double t4 = t2 * t2;
        const double t5 = t2 * t3;

        const double p0 = ValueOnPoint(i);
        const double p1 = ValueOnPoint(i+1);
        const double m0 = DifferenceOnPoint(i);  //m0 = dp0/dx * dx;
        const double m1 = DifferenceOnPoint(i+1);
        const double s0 = SecondOnPoint(i);  //m0 = dp0/dx * dx;
        const double s1 = SecondOnPoint(i + 1);

        //Fifth Hermite Interpolation//
        const double h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
        const double h01 = 1.0 - h00;
        const double h10 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
        const double h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
        const double h20 = (t2 - 3.0*t3 + 3.0*t4-t5)/2.0;
        const double h21 = (t3 - 2.0 * t4 + t5) / 2.0;

        const double inter_p = h00 * p0 + h10 * m0 + h20 * s0 + h01 * p1 + h11 * m1+ h21 * s1;

        if (res_dp_dx) {
            *res_dp_dx = ((-30.0*t2 +60.0*t3-30.0*t4) * (p0 - p1)
                + (1.0 - 18.0*t2 + 32.0*t3 - 15.0*t4) * m0
                + (-12.0*t2 + 28.0*t3 - 15.0*t4) * m1
                + (2.0*t - 9.0 * t2 + 12.0 * t3 - 5.0 * t4)/2.0 * s0
                + (3.0 * t2 - 8.0 * t3 + 5.0 * t4) / 2.0 * s1 ) / dx;
        }
        return inter_p;

    }


};
