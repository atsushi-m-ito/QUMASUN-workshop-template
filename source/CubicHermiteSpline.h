#pragma once
/*****************************************
* CubicHermiteSpline class
* CubicHermiteSpline_diff2nd class
* 
* Cubic Hermite Spline Interpolation
* 三次エルミート補間を実現する
* ただし、CubicHermiteSpline_diff2ndでは一階微分値に関しては、二次精度の値を用いる
* 
********************************************/


class CubicHermiteSpline {
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
    double* dvdx = nullptr;
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
        if (i < 0) {
            return dvdx[0];
        } else if (i >= m_num_grids) {
            return dvdx[m_num_grids - 1];
        } else {
            return dvdx[i];
        }
        
    }

public:
    
    ~CubicHermiteSpline(){
        delete[] values;
        //delete[] dvdx;
    }


    template<class FUNC>
    void Initialize(int N, double x_begin, double x_end, EDGE_DIFF_TYPE edge_difference_mode, FUNC func) {

        m_num_grids = N;
        m_x_begin = x_begin;
        m_x_end = x_end;
        m_edge_difference_mode = edge_difference_mode;
        values = new double[N * 2];
        dvdx = values + N;

        const double dx = (m_x_end - m_x_begin) / (double)(m_num_grids - 1);
        for (int i = 0; i < m_num_grids; ++i) {
            double x = dx * (double)(i)+m_x_begin;    
            double dv_dx;
            values[i] = func(x, &dv_dx);
            dvdx[i] = dv_dx * dx;
        }
    }

    double Interpolate(double x, double* res_dp_dx) {
        const double dx = (m_x_end - m_x_begin) / (double)(m_num_grids - 1);
        const double nx = (x - m_x_begin)/dx;
        const int i = (int)floor(nx);
        const double t = nx - floor(nx);
        const double t2 = t * t;
        const double t3 = t2 * t;

        const double p0 = ValueOnPoint(i);
        const double p1 = ValueOnPoint(i+1);
        const double m0 = DifferenceOnPoint(i);  //m0 = dp0/dx * dx;
        const double m1 = DifferenceOnPoint(i+1);

#if 0
        //Cubic Hermite Interpolation//
        const double a = (2.0 * (p0 - p1) + m0 + m1);
        const double b = (3.0 * (-p0 + p1) - 2.0 * m0 - m1);
        const double inter_p = a * t3 + b * t2 + m0 * t + p0;

        if (res_dp_dx) {
            *res_dp_dx = (3.0*a * t2 + 2.0*b * t + m0)/dx;
        }

        return inter_p;
#else
        //Cubic Hermite Interpolation//
        const double h01 = 3.0 * t2 - 2.0 * t3;
        const double h00 = 1.0 - h01;
        const double h11 = -t2 + t3;
        const double h10 = t - t2 + h11;
        const double inter_p = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;

        if (res_dp_dx) {
            *res_dp_dx = ((6.0*t - 6.0*t2) * (p1-p0) + (3.0*t2-4.0*t+1.0) * m0 + (-2.0*t+3.0*t2) * m1) / dx;
        }
        return inter_p;
#endif   
    }


};



class CubicHermiteSpline_diff2nd {
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
            } else {
                return (values[1] - values[0]);
            }
        } else if (i >= m_num_grids - 1) {
            return values[m_num_grids - 1] - values[m_num_grids - 2];
        } else {
            return (values[i + 1] - values[i - 1]) / 2.0;
        }
    }

public:

    ~CubicHermiteSpline_diff2nd() {
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
        const double nx = (x - m_x_begin) / dx;
        const int i = (int)floor(nx);
        const double t = nx - floor(nx);
        const double t2 = t * t;
        const double t3 = t2 * t;

        const double p0 = ValueOnPoint(i);
        const double p1 = ValueOnPoint(i + 1);
        const double m0 = DifferenceOnPoint(i);  //m0 = dp0/dx * dx;
        const double m1 = DifferenceOnPoint(i + 1);

#if 0
        //Cubic Hermite Interpolation//
        const double a = (2.0 * (p0 - p1) + m0 + m1);
        const double b = (3.0 * (-p0 + p1) - 2.0 * m0 - m1);
        const double inter_p = a * t3 + b * t2 + m0 * t + p0;

        if (res_dp_dx) {
            *res_dp_dx = (3.0 * a * t2 + 2.0 * b * t + m0) / dx;
        }

        return inter_p;
#else
        //Cubic Hermite Interpolation//
        const double h01 = 3.0 * t2 - 2.0 * t3;
        const double h00 = 1.0 - h01;
        const double h11 = -t2 + t3;
        const double h10 = t - t2 + h11;
        const double inter_p = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;

        if (res_dp_dx) {
            *res_dp_dx = ((6.0 * t - 6.0 * t2) * (p1 - p0) + (3.0 * t2 - 4.0 * t + 1.0) * m0 + (-2.0 * t + 3.0 * t2) * m1) / dx;
        }
        return inter_p;
#endif   
    }


};
