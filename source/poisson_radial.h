#pragma once
#include <cmath>

//Vhatを積分によって求める方法//
//input is rho, output is Vhart
inline 
void PoissonRadial(const double* rho_v, int num_grid, const double* radius, double dx, double* vhart_v) {

    double sum_ex2rho = 0.0;
    //const double* radius = m_radius;
    for (int i = 0; i < num_grid; i++) {
        const double r = radius[i];
        const double r3 = r * r * r;

        sum_ex2rho += rho_v[i] * r3 * dx;
        vhart_v[i] = sum_ex2rho / r;

    }

    double sum_exrho = 0.0;
    for (int i = num_grid - 1; i >= 0; i--) {
        const double r = radius[i];
        const double r2 = r * r;


        vhart_v[i] += sum_exrho;
        vhart_v[i] *= 4.0 * M_PI;
        sum_exrho += rho_v[i] * r2 * dx;

    }

}
