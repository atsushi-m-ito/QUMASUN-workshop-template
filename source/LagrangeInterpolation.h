#pragma once


template <int N>
class LagrangeInterpolator {
private:

    double F_table[N];
    double points[N];
public:
    LagrangeInterpolator() = default;

    static constexpr int size = N;

    template <class F>
    void Initialize(const double* points_, F func) {
        for (int i = 0; i < N; ++i) {
            points[i] = points_[i];
            F_table[i] = func(points_[i]);
        }
    }

    /*
    * 区間[-1,1]の節点(point)に対して、funcが偶関数になっている場合
    * コストが半分で済む
    */
    template <class F>
    void InitializeEven(const double* points_, F func) {

        for (int i = 0; i < N; ++i) {
            points[i] = points_[i];
        }
        for (int i = N/2; i < N; ++i) {
            F_table[i] = func(points_[i]);
            F_table[N - i - 1] = F_table[i];
        }
    }

    
    inline double LagrangeInterpolation(double x) {
        double val = 0.0;
        
        for (int i = 0; i < N; ++i) {
            double v = F_table[i];
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                v *= (x - points[j]) / (points[i] - points[j]);
            }
            val += v;
        }
        return val;
    };

    
    inline double LagrangeInterpolationDerivative(double x) {
        double val = 0.0;        
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                if (i == k) continue;
                double v = F_table[i] / (points[i] - points[k]);
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    if (j == k) continue;
                    v *= (x - points[j]) / (points[i] - points[j]);
                }
                val += v;
            }
        }
        return val;
    };


};


namespace ChebyshevNodes{

inline void SetPoints(int N, double* points) {
    for (int i = 0; i < N; ++i) {
        double angle = (double)(2 * i + 1) / ((double)(2 * N)) * M_PI;
        points[i] = cos(angle);
    }
};

}
