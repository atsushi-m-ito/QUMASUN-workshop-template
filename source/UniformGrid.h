#pragma once


//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>


/*
等間隔グリッドと補間

他のクラスに参照させて、メモリは共有で使いまわす
*/
class UniformGrid{
public:
	const int num_grids;
	
	const double xi_min;
	//const double xi_max;
	const double xi_delta;//コンストラクタ初期化の順番の為、必ずxi_min,xi_maxより後に定義すること//
	
    static constexpr int EVEN = 1;
    static constexpr int ZERO = 0;
    static constexpr int ODD = -1;

    UniformGrid(const int num_grids_a, const double xi_min_a, const double xi_delta_a) :
		num_grids(num_grids_a),
		xi_min(xi_min_a),
		//xi_max(xi_max_a),
		xi_delta(xi_delta_a)		
	{
	};


	double GetValue(const double xi, const double* values, int even_odd_zero) const {
		return GetValue(xi, values, this->num_grids, this->xi_min, this->xi_delta, even_odd_zero);
	}

	inline
	static double GetValue(const double xi, const double* values, int num_grids, double xi_min, double xi_delta, int even_odd_zero) {
		const double relative_xi = xi - xi_min;
		const double delta = xi_delta;
		double floor_xi = floor(relative_xi / delta);
		double x = (relative_xi / delta) - floor_xi;
		int i = (int)(floor_xi);

		
        auto Get = [&](int i) {
            if (i < 0) {
                return values[-i] * (double)even_odd_zero;
            } else if (i >= num_grids) {
                return 0.0;
            }
            return values[i];
            };

		//xi空間での三次スプライン補完//
		{//二次精度//
            
            const double fm2 = Get(i-2);
            const double fm1 = Get(i-1);
			const double f0 = Get(i);
			const double f1 = Get(i + 1);
			const double f2 = Get(i + 2);
			const double f3 = Get(i + 3);
			
			const double d1f0 = 2.0 / 3.0 * (f1 - fm1) - 1.0 / 12.0 * (f2 - fm2);
			const double d1f1 = 2.0 / 3.0 * (f2 - f0) - 1.0 / 12.0 * (f3 - fm1);
			const double d2f0 = 4.0 / 3.0 * (f1 + fm1) - 1.0 / 12.0 * (f2 + fm2) - 5.0 / 2.0 * f0;
			const double d2f1 = 4.0 / 3.0 * (f2 + f0) - 1.0 / 12.0 * (f3 + fm1) - 5.0 / 2.0 * f1;
			
			auto c3 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return 10.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) - 4.0 * (d1p1 - (d1p0 + d2p0)) + 1.0 / 2.0 * (d2p1 - d2p0);
			};
			auto c4 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return -15.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) + 7.0 * (d1p1 - (d1p0 + d2p0)) - (d2p1 - d2p0);
			};
			auto c5 = [&c3,&c4](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return p1 - p0 - d1p0 - d2p0 / 2.0 - c3(p0, d1p0, d2p0, p1, d1p1, d2p1) - c4(p0, d1p0, d2p0, p1, d1p1, d2p1);
			};
			auto K = [&c3, &c4, &c5](double x, double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return (p0)+x * ((d1p0)+x * ((d2p0 / 2.0) + x * (c3(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * (c4(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * c5(p0, d1p0, d2p0, p1, d1p1, d2p1)))));
			};

			const double x = (relative_xi / delta) - (double)i;
			return K(x, f0, d1f0, d2f0, f1, d1f1, d2f1);

		}
		
		
	};


	/*
	* Pay attention this is not diveded yet.
	*/
	double GetLaplacian(const double xi, const double* values, const int L, int even_odd_zero) const {
		return GetLaplacian(xi, values, L, this->num_grids, this->xi_min, this->xi_delta, even_odd_zero);

	}

	inline 
	static double GetLaplacian(const double xi, const double* values, const int L, int num_grids, double xi_min, double xi_delta, int even_odd_zero)  {
		const double relative_xi = xi - xi_min;
		const double delta = xi_delta;
		double floor_xi = floor(relative_xi / delta);
		double x = (relative_xi / delta) - floor_xi;
		int i = (int)(floor_xi);
		const double L_L_plus_1 = (double)(L * (L + 1));

        auto Get = [&](int i) {
            if (i < 0) {
                return values[-i] * (double)even_odd_zero;
            } else if (i >= num_grids) {
                return 0.0;
            }
            return values[i];
            };


		//xi空間での三次スプライン補完//
		{//二次精度//

            const double fm2 = Get(i - 2);
            const double fm1 = Get(i - 1);
            const double f0 = Get(i);
            const double f1 = Get(i + 1);
            const double f2 = Get(i + 2);
            const double f3 = Get(i + 3);

			const double d1f0 = 2.0 / 3.0 * (f1 - fm1) - 1.0 / 12.0 * (f2 - fm2);
			const double d1f1 = 2.0 / 3.0 * (f2 - f0) - 1.0 / 12.0 * (f3 - fm1);
			const double d2f0 = 4.0 / 3.0 * (f1 + fm1) - 1.0 / 12.0 * (f2 + fm2) - 5.0 / 2.0 * f0;
			const double d2f1 = 4.0 / 3.0 * (f2 + f0) - 1.0 / 12.0 * (f3 + fm1) - 5.0 / 2.0 * f1;

			auto c3 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return 10.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) - 4.0 * (d1p1 - (d1p0 + d2p0)) + 1.0 / 2.0 * (d2p1 - d2p0);
			};
			auto c4 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return -15.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) + 7.0 * (d1p1 - (d1p0 + d2p0)) - (d2p1 - d2p0);
			};
			auto c5 = [&c3, &c4](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return p1 - p0 - d1p0 - d2p0 / 2.0 - c3(p0, d1p0, d2p0, p1, d1p1, d2p1) - c4(p0, d1p0, d2p0, p1, d1p1, d2p1);
			};
			auto K = [&c3, &c4, &c5](double x, double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return (p0)+x * ((d1p0)+x * ((d2p0 / 2.0) + x * (c3(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * (c4(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * c5(p0, d1p0, d2p0, p1, d1p1, d2p1)))));
			};
			auto dKdx = [&c3, &c4, &c5](double x, double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return (d1p0)+x * ((d2p0) + x * (3.0*c3(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * (4.0*c4(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * 5.0*c5(p0, d1p0, d2p0, p1, d1p1, d2p1))));
			};
			auto d2Kdx2 = [&c3, &c4, &c5](double x, double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
				return ((d2p0)+x * (6.0 * c3(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * (12.0 * c4(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * 20.0 * c5(p0, d1p0, d2p0, p1, d1p1, d2p1))));
			};

			const double v = K(x, f0, d1f0, d2f0, f1, d1f1, d2f1);
			const double dv_dxi = dKdx(x, f0, d1f0, d2f0, f1, d1f1, d2f1) / delta;
			const double d2v_dxi2 = d2Kdx2(x, f0, d1f0, d2f0, f1, d1f1, d2f1) / (delta * delta);

			double Laplace_Psi = (d2v_dxi2 + (2.0/xi)* dv_dxi - 2.0*(L_L_plus_1 * v)/(xi* xi));//
			return -0.5 * Laplace_Psi;

		} 
			
		return 0.0;
		
	};


    inline
    static double GetDifferential1(const double xi, const double* values, int num_grids, double xi_min, double xi_delta, int even_odd_zero) {
        const double relative_xi = xi - xi_min;
        const double delta = xi_delta;
        double floor_xi = floor(relative_xi / delta);
        double x = (relative_xi / delta) - floor_xi;
        int i = (int)(floor_xi);
        //const double L_L_plus_1 = (double)(L * (L + 1));

        auto Get = [&](int i) {
            if (i < 0) {
                return values[-i] * (double)even_odd_zero;
            } else if (i >= num_grids) {
                return 0.0;
            }
            return values[i];
            };


        //xi空間での三次スプライン補完//
        {//二次精度//

            const double fm2 = Get(i - 2);
            const double fm1 = Get(i - 1);
            const double f0 = Get(i);
            const double f1 = Get(i + 1);
            const double f2 = Get(i + 2);
            const double f3 = Get(i + 3);

            const double d1f0 = 2.0 / 3.0 * (f1 - fm1) - 1.0 / 12.0 * (f2 - fm2);
            const double d1f1 = 2.0 / 3.0 * (f2 - f0) - 1.0 / 12.0 * (f3 - fm1);
            const double d2f0 = 4.0 / 3.0 * (f1 + fm1) - 1.0 / 12.0 * (f2 + fm2) - 5.0 / 2.0 * f0;
            const double d2f1 = 4.0 / 3.0 * (f2 + f0) - 1.0 / 12.0 * (f3 + fm1) - 5.0 / 2.0 * f1;

            auto c3 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
                return 10.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) - 4.0 * (d1p1 - (d1p0 + d2p0)) + 1.0 / 2.0 * (d2p1 - d2p0);
                };
            auto c4 = [](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
                return -15.0 * (p1 - (p0 + d1p0 + d2p0 / 2.0)) + 7.0 * (d1p1 - (d1p0 + d2p0)) - (d2p1 - d2p0);
                };
            auto c5 = [&c3, &c4](double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
                return p1 - p0 - d1p0 - d2p0 / 2.0 - c3(p0, d1p0, d2p0, p1, d1p1, d2p1) - c4(p0, d1p0, d2p0, p1, d1p1, d2p1);
                };
            auto dKdx = [&c3, &c4, &c5](double x, double p0, double  d1p0, double  d2p0, double  p1, double  d1p1, double  d2p1) {
                return (d1p0)+x * ((d2p0)+x * (3.0 * c3(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * (4.0 * c4(p0, d1p0, d2p0, p1, d1p1, d2p1) + x * 5.0 * c5(p0, d1p0, d2p0, p1, d1p1, d2p1))));
                };

            const double dv_dxi = dKdx(x, f0, d1f0, d2f0, f1, d1f1, d2f1) / delta;

            return dv_dxi;

        }
    };

    inline
    double GetDifferential1(const double r, const double* values, int even_odd_zero) {
        return GetDifferential1(r, values, num_grids, xi_min, xi_delta, even_odd_zero);
    };

};

