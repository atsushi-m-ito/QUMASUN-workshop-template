#pragma once
#include <vector>
#include <memory>
#include <complex>
#include "GridRange.h"
#include "soacomplex.h"
#include "vecmath.h"
//#include "w_zhegvx.h"
#include "GramSchmidt_mpi.h"
#include "inverse_m.h"
#include "print_matrix.h"

//#define DIIS_EVERY_RESIDUAL_RECALCULATE

#define DIIS_PLUS_RESIDUAL

#if 1

class DIIS_Psi {

    int history_max = 0;
    int current_index = 0;

    //note: 最終的な目的が結合後の残差の絶対値二乗の最小化なので, 実部だけ考えればよい//
    double* mat_R = nullptr;
    double* mat_S = nullptr;

//#ifndef DIIS_PLUS_RESIDUAL
    const double lambda = -0.1;  //parameter of trial step//
//#endif

    int Index(int i) {
        return (i + history_max) % history_max;
    }

public:
    void Initialize(int history_count_, int num_solution)
    {
        history_max = history_count_;
#ifdef DIIS_PLUS_RESIDUAL
        mat_R = new double[(history_max + 2) * (history_max + 2) * num_solution];
#else
        mat_R = new double[(history_max +1) * (history_max + 1) *  num_solution];
#endif
    }

    ~DIIS_Psi() {
        delete[] mat_R;
    }

    template <class OperationA, class WATCH>
    void Eigen_z_multi_mpi(const GridRangeMPI& l_grid, const double dVol, const int num_solution,
        double* eigen_values, SoAComplex* eigen_vectors, double* keep,
        int SCF_step, int sk,
/*
#ifdef USE_SCALAPACK
        const BlacsGridInfo& blacs_grid,
#endif
*/
        OperationA OpeA, WATCH& watch)
    {

        using namespace vecmath;

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const bool is_root = (proc_id == 0);
        const size_t local_size = l_grid.Size3D();


        auto IntervalPointers = [](double* a, size_t N, size_t num_solution) {
            std::vector<SoAComplex> pointers(num_solution);
            for (size_t i = 0; i < num_solution; ++i) {
                pointers[i].re = a + (i * 2) * N;
                pointers[i].im = a + (i * 2 + 1) * N;
            }
            return pointers;
            };

        auto IntervalPointersList = [](double* a, size_t N, size_t num_solution, int history_count) {
            std::vector<std::vector<SoAComplex>> pointers_list(history_count);
            for (size_t h = 0; h < history_count; ++h) {
                auto& pointers = pointers_list[h];
                pointers.resize(num_solution);
                for (size_t n = 0; n < num_solution; ++n) {
                    pointers[n].re = a + (n * 2) * N + N * num_solution * 2 * h;
                    pointers[n].im = a + (n * 2 + 1) * N + N * num_solution * 2 * h;
                }
            }
            return pointers_list;
            };


        int history_count = std::min(history_max, current_index + 1);

        std::vector<std::vector<SoAComplex>> state = IntervalPointersList(keep, local_size, num_solution, history_max);
        std::vector<std::vector<SoAComplex>> residual = IntervalPointersList(keep + local_size * num_solution * history_max*2, local_size, num_solution, history_max);

        //set current state
        const int c_index = Index(current_index);
        {
            const double sqrtV = sqrt(dVol);
            auto& x = state[c_index];
            for (int n = 0; n < num_solution; ++n) {
                for (int i = 0; i < local_size; ++i) {
                    x[n].re[i] = eigen_vectors[n].re[i] * sqrtV;
                    x[n].im[i] = eigen_vectors[n].im[i] * sqrtV;
                }
            }
        }



#ifdef DIIS_EVERY_RESIDUAL_RECALCULATE
        auto tmp_buffer = std::make_unique<double[]>(num_solution * history_max * history_max*2);
        //calculate residual of current state
        {
            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution * history_count;
            for (int h = 0; h < history_count; ++h) {
                auto& x = state[h];
                auto& Hx = residual[h];

                OpeA(Hx.data(), x.data(), num_solution); // calculate Ax from x 	

                for (int n = 0; n < num_solution; ++n) {
                    auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                    l_ei[n + num_solution*h] = xHx.r;
                }
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution* history_count, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int h = 0; h < history_count; ++h) {
                auto& x = state[h];
                auto& Hx = residual[h];
                for (int n = 0; n < num_solution; ++n) {
                    for (size_t i = 0; i < local_size; ++i) {
                        //here, Hx is in memory of residual//
                        Hx[n].re[i] = (Hx[n].re[i] - ei[n + num_solution * h] * x[n].re[i]);
                        Hx[n].im[i] = (Hx[n].im[i] - ei[n + num_solution * h] * x[n].im[i]);
                    }
                }
            }
        }

        //update matrix R//
        {
            //auto& x = state[c_index];
            
            auto* l_xhx = &tmp_buffer[0];
            auto* sum_xhx = &tmp_buffer[0] + num_solution * history_count* history_count;

            for (int j = 0; j < history_count; ++j) {
                auto& r = residual[j];

                for (int h = 0; h <= j; ++h) {

                    auto& rh = residual[h];
                    for (int n = 0; n < num_solution; ++n) {
                        //auto xhx = SoAC::InnerProd(xh[n], x[n], local_size);
                        auto rhr = SoAC::InnerProd(rh[n], r[n], local_size);
                        l_xhx[n + num_solution * (h + history_count * j)] = rhr.r;

                        //note: 最終的な目的が結合後の残差の絶対値二乗の最小化なので, 実部だけ考えればよい//
                    }
                }
            }

            watch.Record(10);
            MPI_Allreduce(l_xhx, sum_xhx, num_solution * history_count* history_count, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * (history_max + 1) * (history_max + 1);

                for (int j = 0; j < history_count; ++j) {
                    for (int h = 0; h < history_count; ++h) {
                        mat_R[(h + (history_max + 1) * j) + offset] = sum_xhx[(n + num_solution * (h + history_count * j))];
                        mat_R[(j + (history_max + 1) * h) + offset] = sum_xhx[(n + num_solution * (h + history_count * j))];                        
                    }
                    mat_R[(history_max + (history_max + 1) * j) + offset] = 1.0;
                    mat_R[(j + (history_max + 1) * history_max) + offset] = 1.0;
                }
                mat_R[(history_max + (history_max + 1) * history_max) + offset] = 0.0;
            }

        }


#else
        auto tmp_buffer = std::make_unique<double[]>(num_solution * history_max * 4);
        //calculate residual of current state
        {
            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution;
            auto& x = state[c_index];
            auto& Hx = residual[c_index];

            OpeA(Hx.data(), x.data(), num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    Hx[n].re[i] = (Hx[n].re[i] - ei[n] * x[n].re[i]);
                    Hx[n].im[i] = (Hx[n].im[i] - ei[n] * x[n].im[i]);
                }
            }
        }

        //update matrix R//
        {
            auto& x = state[c_index];
            auto& r = residual[c_index];

            auto* l_xhx = &tmp_buffer[0];
            auto* sum_xhx = &tmp_buffer[0] + num_solution * history_count;


            for (int h = 0; h < history_count; ++h) {
                int h_index = Index(c_index - h);
                //auto& xh = state[h_index];
                auto& rh = residual[h_index];
                for (int n = 0; n < num_solution; ++n) {
                    //auto xhx = SoAC::InnerProd(xh[n], x[n], local_size);
                    auto rhr = SoAC::InnerProd(rh[n], r[n], local_size);
                    l_xhx[(n + num_solution * h)] = rhr.r;

                    //note: 最終的な目的が結合後の残差の絶対値二乗の最小化なので, 実部だけ考えればよい//
                }
            }
            watch.Record(10);
            MPI_Allreduce(l_xhx, sum_xhx, num_solution * history_count, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * (history_max + 1) * (history_max + 1);
                for (int h = 0; h < history_count; ++h) {
                    int h_index = Index(c_index - h);
                    mat_R[(h_index + (history_max + 1) * c_index) + offset] = sum_xhx[(n + num_solution * h)];
                    mat_R[(c_index + (history_max + 1) * h_index) + offset] = sum_xhx[(n + num_solution * h)];
                    mat_R[(history_max + (history_max + 1) * h_index) + offset] = 1.0;
                    mat_R[(h_index + (history_max + 1) * history_max) + offset] = 1.0;
                }
                mat_R[(history_max + (history_max + 1) * history_max) + offset] = 0.0;
            }

        }

#endif

        

        //auto update_flags = std::make_unique<bool[]>(num_solution);

#ifdef DIIS_PLUS_RESIDUAL
        
        auto optinal_buf = std::make_unique<double[]>(local_size*2*2* num_solution);
        auto optional_psi = IntervalPointers(&optinal_buf[0], local_size, num_solution);
        auto optional_R = IntervalPointers(&optinal_buf[0] + local_size * 2 * num_solution, local_size, num_solution);
        {

            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution;

            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * (history_max + 1) * (history_max + 1);
                auto& currentX = state[c_index][n]; 
                auto& currentR = residual[c_index][n];
                auto& o_psi = optional_psi[n];
                //double isqrtN = 1.0 / sqrt(mat_R[(c_index + (history_max + 1) * c_index) + offset]);
                for (int i = 0; i < local_size; ++i) {
                    o_psi.re[i] = currentX.re[i] + lambda * currentR.re[i];
                    o_psi.im[i] = currentX.im[i] + lambda * currentR.im[i];
                }

            }

            OpeA(optional_R.data(), optional_psi.data(), num_solution); // calculate Ax from x 

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(optional_psi[n], optional_R[n], local_size);
                l_ei[n] = xHx.r;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    optional_R[n].re[i] = (optional_R[n].re[i] - ei[n] * optional_psi[n].re[i]);
                    optional_R[n].im[i] = (optional_R[n].im[i] - ei[n] * optional_psi[n].im[i]);
                }
            }
        }

        //calculate coefficient//
        {
            auto RR = std::make_unique<double[]>((history_max + 2) * (history_max + 2));
            auto invRR = std::make_unique<double[]>((history_max + 2) * (history_max + 2));


            auto* l_xhx = &tmp_buffer[0];
            auto* sum_xhx = &tmp_buffer[0] + num_solution * (history_count+1);
            

            for (int h = 0; h < history_count; ++h) {
                auto& rh = residual[h];
                for (int n = 0; n < num_solution; ++n) {
                    //auto xhx = SoAC::InnerProd(xh[n], x[n], local_size);
                    auto rhr = SoAC::InnerProd(rh[n], optional_R[n], local_size);
                    l_xhx[(n + num_solution * h)] = rhr.r;
                    
                }
            }
            for (int n = 0; n < num_solution; ++n) {
                //auto xhx = SoAC::InnerProd(xh[n], x[n], local_size);
                auto rhr = SoAC::InnerProd(optional_R[n], optional_R[n], local_size);
                l_xhx[(n + num_solution * history_count)] = rhr.r;

            }
            watch.Record(10);
            MPI_Allreduce(l_xhx, sum_xhx, num_solution * (history_count+1), MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);


            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * (history_max + 1) * (history_max + 1);

                for (int j = 0; j < history_count; ++j) {
                    for (int h = 0; h < history_count; ++h) {
                        RR[h + (history_count + 2) * j] = mat_R[h + (history_max + 1) * j + offset];
                    }
                    RR[history_count+1 + (history_count + 2) * j] = 1.0;
                }
                for (int h = 0; h < history_count; ++h) {
                    RR[h + (history_count + 2) * (history_count+1)] = 1.0;
                }
                RR[(history_count + 1) + (history_count + 2) * (history_count+1)] = 0.0;
                RR[(history_count + 1) + (history_count + 2) * (history_count)] = 1.0;
                RR[(history_count) + (history_count + 2) * (history_count + 1)] = 1.0;

                for (int h = 0; h < history_count + 1; ++h) {
                    RR[h + (history_count + 2) * (history_count)] = sum_xhx[(n + num_solution * h)];
                    RR[(history_count) + (history_count + 2) * h] = sum_xhx[(n + num_solution * h)];
                }
                

#ifdef DEBUG_PRINT
                printf("Matrix RR(%d)\n", n);
                PrintMatrix(&RR[0], history_count + 2, history_count + 2);
#endif

                InverseM(&RR[0], (history_count + 2), &invRR[0]);//入力も破壊されるので複製//

#ifdef DEBUG_PRINT
                printf("invRR(%d)\n", n);
                PrintMatrix(&invRR[0], history_count + 2, history_count + 2);
#endif

                double sum_a = 0.0;
                double* alpha = &tmp_buffer[0] + (history_count+1) * n;
                for (int h = 0; h < history_count+1; ++h) {
                    alpha[h] = invRR[history_count+1 + (history_count + 2) * h];
                    sum_a += alpha[h];
                }

#ifdef DEBUG_PRINT
                printf("Alpha[%d] sum = %f,  check = %f%f\n", n, sum_a, invRR[history_count + (history_count + 1) * history_count]);
                PrintMatrix(alpha, 1, history_count + 1);
#endif


            }
            MPI_Bcast(&tmp_buffer[0], num_solution * (history_count + 1), MPI_DOUBLE, 0, l_grid.mpi_comm);
        }


        //prediction next state
        {
            for (int n = 0; n < num_solution; ++n) {
                //if (update_flags[n]) 
                {
                    memset(eigen_vectors[n].re, 0, sizeof(double) * local_size);
                    memset(eigen_vectors[n].im, 0, sizeof(double) * local_size);
                }
            }
            for (int h = 0; h < history_count; ++h) {
                //int h_index = Index(c_index - h);
                //auto& x = state[h_index];
                auto& x = state[h];
                for (int n = 0; n < num_solution; ++n) {
                    //if (update_flags[n]) 
                    {
                        double alpha = tmp_buffer[(h + (history_count + 1) * n)];

                        for (size_t i = 0; i < local_size; ++i) {
                            eigen_vectors[n].re[i] += alpha * x[n].re[i];
                            eigen_vectors[n].im[i] += alpha * x[n].im[i];
                        }

                    }
                }
            }

            for (int n = 0; n < num_solution; ++n) {
                //if (update_flags[n]) 
                {
                    double alpha = tmp_buffer[(history_count + (history_count + 1) * n)];

                    for (size_t i = 0; i < local_size; ++i) {
                        eigen_vectors[n].re[i] += alpha * optional_psi[n].re[i];
                        eigen_vectors[n].im[i] += alpha * optional_psi[n].im[i];
                    }

                }
            }

        }

#else
                
        //calculate coefficient//
        {

            auto RR = std::make_unique<double[]>((history_max+1) * (history_max + 1));
            auto invRR = std::make_unique<double[]>((history_max + 1) * (history_max + 1));

            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * (history_max + 1) * (history_max + 1);

#if 0
                double* alpha = &tmp_buffer[0] + history_count * n;
                for (int j = 0; j < history_count; ++j) {
                    for (int h = 0; h < history_count; ++h) {
                        RR[h + history_count  * j] = mat_R[h + (history_max + 1) * j + offset];
                    }
                }

                auto res = SolvePulayCoefficients(&RR[0], alpha, history_count);
                if (res) {
                    printf("ERROR: \n");
                    for (int h = 0; h < history_count; ++h) {
                        alpha[h] = 0.0;
                    }
                    alpha[c_index] = 1.0;
                }

                double sum_a = 0.0;
                for (int h = 0; h < history_count; ++h) {
                    sum_a += alpha[h];
                }

                printf("Alpha[%d]: %f\n",n, sum_a);

                PrintMatrix(alpha, 1, history_count);
                
#else
                if (history_count == history_max) {
                    memcpy(&RR[0], &mat_R[offset], sizeof(double) * (history_max + 1) * (history_max + 1));                    
                } else {
                    for (int j = 0; j < history_count; ++j) {
                        for (int h = 0; h < history_count; ++h) {
                            RR[h + (history_count + 1) * j] = mat_R[h + (history_max + 1) * j + offset];
                        }
                        RR[history_count + (history_count + 1) * j] = 1.0;
                    }
                    for (int h = 0; h < history_count; ++h) {
                        RR[h + (history_count + 1) * history_count] = 1.0;
                    }
                    RR[history_count + (history_count + 1) * history_count] = 0.0;
                }

#ifdef DEBUG_PRINT
                printf("Matrix RR(%d)\n", n);
                PrintMatrix(&RR[0], history_count + 1, history_count + 1);
#endif

                InverseM(&RR[0], (history_count + 1), &invRR[0]);//入力も破壊されるので複製//

#ifdef DEBUG_PRINT
                printf("invRR(%d)\n", n);
                PrintMatrix(&invRR[0], history_count + 1, history_count + 1);
#endif

                double sum_a = 0.0;
                double* alpha = &tmp_buffer[0] + history_count * n;
                for (int h = 0; h < history_count; ++h) {
                    alpha[h] = invRR[history_count + (history_count + 1) * h];
                    sum_a += alpha[h];
                }

#ifdef DEBUG_PRINT
                printf("Alpha[%d] sum = %f,  check = %f%f\n",n, sum_a, invRR[history_count + (history_count + 1) * history_count]);
                PrintMatrix(alpha, 1, history_count);
#endif

#endif
            }
            MPI_Bcast(&tmp_buffer[0], num_solution * history_count, MPI_DOUBLE, 0, l_grid.mpi_comm);
        }


        //prediction next state
        {
            for (int n = 0; n < num_solution; ++n) {
                //if (update_flags[n]) 
                {
                    memset(eigen_vectors[n].re, 0, sizeof(double) * local_size);
                    memset(eigen_vectors[n].im, 0, sizeof(double) * local_size);
                }
            }
            for (int h = 0; h < history_count; ++h) {
                //int h_index = Index(c_index - h);
                //auto& x = state[h_index];
                auto& x = state[h];
                for (int n = 0; n < num_solution; ++n) {
                    //if (update_flags[n]) 
                    {
                        double alpha = tmp_buffer[(h + history_count * n)];

                        for (size_t i = 0; i < local_size; ++i) {
                            eigen_vectors[n].re[i] += alpha * x[n].re[i];
                            eigen_vectors[n].im[i] += alpha * x[n].im[i];
                        }

                    }
                }
            }

        }
#endif



        //next psi, Gram-Schmidt, and eigen values
        {
            int n_index = Index(c_index + 1);
            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution;
            auto& x = eigen_vectors;
            auto& Hx = residual[n_index];

#ifdef DIIS_PLUS_RESIDUAL

            double isqrtV = 1.0 / sqrt(dVol);
            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    x[n].re[i] *= isqrtV;
                    x[n].im[i] *= isqrtV;
                }
            }

            GramSchmidt_ddm(l_grid, num_solution, x, dVol);

            //eigen value for new states
            OpeA(Hx.data(), x, num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r * dVol;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, eigen_values, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

#else
            OpeA(Hx.data(), x, num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    Hx[n].re[i] = (Hx[n].re[i] - ei[n] * x[n].re[i]);
                    Hx[n].im[i] = (Hx[n].im[i] - ei[n] * x[n].im[i]);
                }
            }

            double isqrtV = 1.0 / sqrt(dVol);

            for (int n = 0; n < num_solution; ++n) {
                //if (update_flags[n]) 
                {

                    for (size_t i = 0; i < local_size; ++i) {
                        //here, Hx is in memory of residual//
                        x[n].re[i] = (x[n].re[i] + lambda * Hx[n].re[i]) * isqrtV;
                        x[n].im[i] = (x[n].im[i] + lambda * Hx[n].im[i]) * isqrtV;
                    }
                } /*else {
                    for (size_t i = 0; i < local_size; ++i) {
                        //here, Hx is in memory of residual//
                        x[n].re[i] *= isqrtV;
                        x[n].im[i] *= isqrtV;
                    }
                }*/
            }


            GramSchmidt_ddm(l_grid, num_solution, x, dVol);


            //eigen value for new states
            OpeA(Hx.data(), x, num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r * dVol;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, eigen_values, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);
#endif
#ifdef DEBUG_PRINT

            printf("Eigen Value:\n");
            for (int n = 0; n < num_solution; ++n) {
                printf("  %d, %f\n", n, eigen_values[n]);
            }
#endif

        }

        ++current_index;

    }

};

#else

class DIIS_Psi {

    int history_max = 0;
    int current_index = 0;

    double* mat_R = nullptr;
    double* mat_S = nullptr;

    const double lambda = -0.1;  //parameter of trial step//

    int Index(int i) {
        return (i + history_max) % history_max;
    }

public:
    void Initialize(int history_count_, int num_solution)
    {
        history_max = history_count_;
        mat_R = new double[history_count_ * history_count_ * 2 * num_solution];
        mat_S = new double[history_count_ * history_count_ * 2 * num_solution];
        memset(mat_R, 0, sizeof(double) * history_count_ * history_count_ * 2 * num_solution);
        memset(mat_S, 0, sizeof(double) * history_count_ * history_count_ * 2 * num_solution);
        /*
        for (int n = 0; n < num_solution; ++n) {
            int64_t offset = n * history_max * history_max * 2;
            for (int h = 0; h < history_max; ++h) {
                mat_R[offset + 2 * (h + history_max * h)] = 1.0;
                mat_S[offset + 2 * (h + history_max * h)] = 1.0;
            }

        }*/
    }

    ~DIIS_Psi() {
        delete[] mat_R;
        delete[] mat_S;
    }

    template <class OperationA, class WATCH>
    void Eigen_z_multi_mpi(const GridRangeMPI& l_grid, const double dVol, const int num_solution, 
            double* eigen_values, SoAComplex* eigen_vectors, double* keep,
            int SCF_step, int sk,
#ifdef USE_SCALAPACK
            const BlacsGridInfo& blacs_grid,
#endif
            OperationA OpeA, WATCH& watch)
    {

        using namespace vecmath;

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const bool is_root = (proc_id == 0);
        const size_t local_size = l_grid.Size3D();


        auto IntervalPointers = [](double* a, size_t N, size_t num_solution) {
            std::vector<SoAComplex> pointers(num_solution);
            for (size_t i = 0; i < num_solution; ++i) {
                pointers[i].re = a + (i * 2) * N;
                pointers[i].im = a + (i * 2 + 1) * N;
            }
            return pointers;
            };

        auto IntervalPointersList = [](double* a, size_t N, size_t num_solution, int history_count) {
            std::vector<std::vector<SoAComplex>> pointers_list(history_count);
            for (size_t h = 0; h < history_count; ++h) {
                auto& pointers = pointers_list[h];
                pointers.resize(num_solution);
                for (size_t n = 0; n < num_solution; ++n) {
                    pointers[n].re = a + (n * 2) * N + N* num_solution * 2*h;
                    pointers[n].im = a + (n * 2 + 1) * N + N * num_solution * 2 * h;
                }
            }
            return pointers_list;
            };


        int history_count = std::min(history_max, SCF_step + 1);

        std::vector<std::vector<SoAComplex>> state = IntervalPointersList(keep, local_size, num_solution, history_max);
        std::vector<std::vector<SoAComplex>> residual = IntervalPointersList(keep + local_size* num_solution* history_max * 2, local_size, num_solution, history_max);
        
        //set current state
        const int c_index = Index(current_index);
        {
            const double sqrtV = sqrt(dVol);
            auto& x = state[c_index];
            for (int n = 0; n < num_solution; ++n) {
                for (int i = 0; i < local_size; ++i) {
                    x[n].re[i] = eigen_vectors[n].re[i] * sqrtV;
                    x[n].im[i] = eigen_vectors[n].im[i] * sqrtV;
                }
            }
        }

        //calculate residual of current state

        auto tmp_buffer = std::make_unique<double[]>(num_solution * history_max *8);
        
        {
            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution;
            auto& x = state[c_index];
            auto& Hx = residual[c_index];

            OpeA(Hx.data(), x.data(), num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    Hx[n].re[i] = (Hx[n].re[i] - ei[n] * x[n].re[i]);
                    Hx[n].im[i] = (Hx[n].im[i] - ei[n] * x[n].im[i]);
                }
            }
        }
        

        //update matrix R//
        {
            auto& x = state[c_index];
            auto& r = residual[c_index];

            auto* l_xhx = &tmp_buffer[0];
            auto* sum_xhx = &tmp_buffer[0] + 4 * num_solution * history_count;


            for (int h = 0; h < history_count; ++h) {
                int h_index = Index(c_index - h);
                auto& xh = state[h_index];
                auto& rh = residual[h_index];
                for (int n = 0; n < num_solution; ++n) {
                    auto xhx = SoAC::InnerProd(xh[n], x[n], local_size);
                    auto rhr = SoAC::InnerProd(rh[n], r[n], local_size);
                    l_xhx[0 + 4 * (n + num_solution * h)] = xhx.r;
                    l_xhx[1 + 4 * (n + num_solution * h)] = xhx.i;
                    l_xhx[2 + 4 * (n + num_solution * h)] = rhr.r;
                    l_xhx[3 + 4 * (n + num_solution * h)] = rhr.i;
                }
            }
            watch.Record(10);
            MPI_Allreduce(l_xhx, sum_xhx, 4*num_solution* history_count, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int h = 0; h < history_count; ++h) {
                int h_index = Index(c_index - h);
                for (int n = 0; n < num_solution; ++n) {
                    int64_t offset = n * history_max * history_max * 2;
                    mat_S[0 + 2 * (h_index + history_max * c_index) + offset] = sum_xhx[0 + 4 * (n + num_solution * h)];
                    mat_S[1 + 2 * (h_index + history_max * c_index) + offset] = sum_xhx[1 + 4 * (n + num_solution * h)];
                    mat_S[0 + 2 * (c_index + history_max * h_index) + offset] = sum_xhx[0 + 4 * (n + num_solution * h)];
                    mat_S[1 + 2 * (c_index + history_max * h_index) + offset] = -sum_xhx[1 + 4 * (n + num_solution * h)];
                    mat_R[0 + 2 * (h_index + history_max * c_index) + offset] = sum_xhx[2 + 4 * (n + num_solution * h)];
                    mat_R[1 + 2 * (h_index + history_max * c_index) + offset] = sum_xhx[3 + 4 * (n + num_solution * h)];
                    mat_R[0 + 2 * (c_index + history_max * h_index) + offset] = sum_xhx[2 + 4 * (n + num_solution * h)];
                    mat_R[1 + 2 * (c_index + history_max * h_index) + offset] = -sum_xhx[3 + 4 * (n + num_solution * h)];
                }
            }
        }


        auto update_flags = std::make_unique<bool[]>(num_solution);

        //calculate coefficient//
        {
            auto eigen_value = std::make_unique<double[]>(history_max );
            auto eigen_vec = std::make_unique<double[]>(history_max * history_max * 2);
            //lapackで書き換えられてしまうので一次バッファに入れる//
            auto tmpR = std::make_unique<double[]>(history_max * history_max * 2);
            auto tmpS = std::make_unique<double[]>(history_max * history_max * 2);

            for (int n = 0; n < num_solution; ++n) {
                int64_t offset = n * history_max * history_max * 2;
                double* alpha = &tmp_buffer[0] + 2 * history_max * n;
                PrintMatrix(&mat_R[offset], history_max*2, history_max);
                PrintMatrix(&mat_S[offset], history_max * 2, history_max);
                memcpy(&tmpR[0], &mat_R[offset], sizeof(double) * history_max * history_max * 2);
                memcpy(&tmpS[0], &mat_S[offset], sizeof(double) * history_max * history_max * 2);
                lapack_ZHEGVX(&tmpR[0], &tmpS[0], &eigen_value[0], &eigen_vec[0], history_count, 1, history_count, history_max);

                double norm_alpha = 0.0;
                for (int h = 0; h < history_count *2; ++h) {
                    alpha[h] = eigen_vec[h];
                    norm_alpha += alpha[h] * alpha[h];
                }
                printf("Alpha (ei = %f, |alpha|^2 = %f\n", eigen_value[0], norm_alpha);
                if ((norm_alpha < 1.0e-8 ) || (eigen_value[0] < 1.0e-8)) {
                    update_flags[n] = false;
                    for (int h = 0; h < history_count * 2; ++h) {
                        alpha[h] = 0.0;
                    }
                    alpha[c_index * 2] = 1.0;

                    printf("fallback \n");
                } else {
                    update_flags[n] = true;
                }
                PrintMatrix(alpha, 2, history_count);
            }
            MPI_Bcast(&tmp_buffer[0], num_solution* history_count * 2, MPI_DOUBLE, 0, l_grid.mpi_comm);
        }

        //prediction next state
        {
            for (int n = 0; n < num_solution; ++n) {
                //if (update_flags[n]) 
                {
                    memset(eigen_vectors[n].re, 0, sizeof(double) * local_size);
                    memset(eigen_vectors[n].im, 0, sizeof(double) * local_size);
                }
            }
            for (int h = 0; h < history_count; ++h) {
                int h_index = Index(c_index - h);
                auto& x = state[h_index];
                for (int n = 0; n < num_solution; ++n) {
                    //if (update_flags[n]) 
                    {
                        double alpha_re = tmp_buffer[2 * (h + history_max * n)];
                        double alpha_im = tmp_buffer[2 * (h + history_max * n) + 1];

                        for (size_t i = 0; i < local_size; ++i) {
                            eigen_vectors[n].re[i] += alpha_re * x[n].re[i] - alpha_im * x[n].im[i];
                            eigen_vectors[n].im[i] += alpha_re * x[n].im[i] + alpha_im * x[n].re[i];
                        }

                    }
                }
            }

        }


        //next psi, Gram-Schmidt, and eigen values
        {
            int n_index = Index(c_index + 1);
            auto* l_ei = &tmp_buffer[0];
            auto* ei = &tmp_buffer[0] + num_solution;
            auto& x = eigen_vectors;
            auto& Hx = residual[n_index];

            OpeA(Hx.data(), x, num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, ei, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            for (int n = 0; n < num_solution; ++n) {
                for (size_t i = 0; i < local_size; ++i) {
                    //here, Hx is in memory of residual//
                    Hx[n].re[i] = (Hx[n].re[i] - ei[n] * x[n].re[i]);
                    Hx[n].im[i] = (Hx[n].im[i] - ei[n] * x[n].im[i]);
                }
            }

            double isqrtV = 1.0/sqrt(dVol);
            
            for (int n = 0; n < num_solution; ++n) {
                if (update_flags[n]) {

                    for (size_t i = 0; i < local_size; ++i) {
                        //here, Hx is in memory of residual//
                        x[n].re[i] = (x[n].re[i] + lambda * Hx[n].re[i]) * isqrtV;
                        x[n].im[i] = (x[n].im[i] + lambda * Hx[n].im[i]) * isqrtV;
                    }
                } else {
                    for (size_t i = 0; i < local_size; ++i) {
                        //here, Hx is in memory of residual//
                        x[n].re[i] *= isqrtV;
                        x[n].im[i] *= isqrtV;
                    }
                }
            }
        

            GramSchmidt_ddm(l_grid, num_solution, x, dVol);
        

            //eigen value for new states
            OpeA(Hx.data(), x, num_solution); // calculate Ax from x 	

            for (int n = 0; n < num_solution; ++n) {
                auto xHx = SoAC::InnerProd(x[n], Hx[n], local_size);
                l_ei[n] = xHx.r * dVol;
            }
            watch.Record(10);
            MPI_Allreduce(l_ei, eigen_values, num_solution, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
            watch.Record(14);

            printf("Eigen Value:\n");
            for (int n = 0; n < num_solution; ++n) {
                printf("  %d, %f\n", n, eigen_values[n]);
            }
        }

        ++current_index;

    }

};
#endif

