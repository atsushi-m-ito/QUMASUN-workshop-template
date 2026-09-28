#pragma once
#include "vps_loader.h"
#include "print_matrix.h"



inline
void TestDiagonal(PseudoPot_MBK* pp){

    auto Inner = [](const double* p1, const double* p2, const double* radius, int num_radial_grid, double dxi) {
        double sum = 0.0;
        for (int i = 0; i < num_radial_grid; ++i) {
            sum += p1[i] * p2[i] * radius[i] * radius[i] * radius[i];
        }
        sum *= dxi;
        return sum;
    };

    auto Normalize = [&Inner](double* p1, const double* radius, int num_radial_grid, double dxi) {
        double norm = Inner(p1, p1, radius, num_radial_grid, dxi);
        norm = 1.0 / sqrt(norm);
        
        for (int i = 0; i < num_radial_grid; ++i) {
            p1[i] *= norm;
        }
        };


    auto Sub = [](double* p1, const double* p2, double coef, int num_radial_grid) {
        
        for (int i = 0; i < num_radial_grid; ++i) {
            p1[i] -= p2[i] * coef;
        }
        };

    const int max_l = pp->MaxProjectorL();
    int n_begin = 0;
    for (int l = 0; l <= max_l; ++l) {
        int n_end = pp->num_radial_projectors;
        for (int n = n_begin; n < pp->num_radial_projectors; ++n) {
            const int my_l = pp->projector_quantum_l[n];
            if (l < my_l) {
                n_end = n;
                break;
            }
        }

        //loaded projectors are P matrix 
        //orthodiagonalized projectors are Psi matrix
        const int num_radial_grids = pp->num_radial_grids;
        printf("projector: l = %d: n in [%d, %d)\n", l, n_begin, n_end);
        auto psi_set = std::make_unique<double[]>(num_radial_grids * (n_end - n_begin));
        for (int n = n_begin; n < n_end; ++n) {
            double* psi_n = &psi_set[num_radial_grids * (n - n_begin)];
            memcpy(psi_n, pp->projector[n*2], sizeof(double) * num_radial_grids);

            //diagonalize//
            for (int m = n_begin; m < n; ++m) {
                double* psi_m = &psi_set[num_radial_grids * (m - n_begin)];
                double inner = Inner(psi_n, psi_m, pp->radius, num_radial_grids, pp->xi_delta);
                Sub(psi_n, psi_m, inner, num_radial_grids);
            }

            Normalize(psi_n, pp->radius, num_radial_grids, pp->xi_delta);

        }

        //calculate Q matrix, where P = Psi * Q
        const int n_size = (n_end - n_begin);
        auto Q = std::make_unique<double[]>(n_size* n_size);
        for (int n = n_begin; n < n_end; ++n) {
            double* pp_n = pp->projector[n*2];
            
            //diagonalize//
            for (int m = n_begin; m < n_end; ++m) {
                double* psi_m = &psi_set[num_radial_grids * (m - n_begin)];
                double inner = Inner(pp_n, psi_m, pp->radius, num_radial_grids, pp->xi_delta);
                Q[(m - n_begin) + n_size * (n - n_begin)] = inner;
            }
        }

        std::string filename("pp-diag_l" + std::to_string(l) + ".txt");
        OutputMatrix(&Q[0], n_size, n_size, filename.c_str());

         n_begin = n_end;
    }
}
