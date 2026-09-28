#pragma once
#include "fftw_executor.h"
#include "GridRange.h"
#include "GridScatterGather.h"
#include "soacomplex.h"
#include "gyield/gy_memory.h"

class TransKerker {
private:
    GridRange& m_global_grid;
    GridRangeMPI& ml_grid;
    double dx;
    double dy;
    double dz;
    FFTW_Executor* m_fftw = nullptr;
    int m_step = 0;
public:
    TransKerker(GridRange& global_grid, GridRangeMPI& l_grid, double dx_, double dy_, double dz_) :
        m_global_grid(global_grid), ml_grid(l_grid), 
        dx(dx_), dy(dy_), dz(dz_)
    {

        m_fftw = new FFTW_Executor;
        m_fftw->Initialize(global_grid.SizeX(), global_grid.SizeY(), global_grid.SizeZ(), FFTW_ESTIMATE);
    }
    ~TransKerker() {
        delete m_fftw;
    }

    void Convert(double* l_diff_rho, double paramA, double paramG0) {
        
        const int64_t size_3d = m_global_grid.Size3D();

        double* global_rho = nullptr;
        if (IsRoot(ml_grid.mpi_comm)) {
            global_rho = gy::AlignedAlloc<double>(size_3d);
        }
        printf("[%d]TEST:Kerker:0\n", GetProcessID(ml_grid.mpi_comm)); fflush(stdout);
        MPI_Barrier(ml_grid.mpi_comm);

        GatherGrid(m_global_grid, global_rho, ml_grid, l_diff_rho, 0);

        printf("[%d]TEST:Kerker:1\n", GetProcessID(ml_grid.mpi_comm)); fflush(stdout);
        MPI_Barrier(ml_grid.mpi_comm);
        if (IsRoot(ml_grid.mpi_comm)) {
#ifdef _DEBUG
            {//test
                const int64_t size_x = m_global_grid.SizeX();
                const int64_t size_y = m_global_grid.SizeY();
                const int64_t size_z = m_global_grid.SizeZ();
                std::string filepath("test_diff_" + std::to_string(m_step) + ".txt");
                FILE* fp = fopen(filepath.c_str(), "w");
                fprintf(fp, "#index, diff_rho\n");

                for (int64_t iz = 0; iz < size_z; ++iz) {
                    int64_t i = iz * size_x * size_y;
                    fprintf(fp, "%d\t%.15f\n", iz, global_rho[i]);
                }

                fprintf(fp, "\n");
                fclose(fp);
            }
#endif
            
            auto* rho_k = m_fftw->GetBuffer();
            {
                
                double sum = 0.0;
                for (int i = 0; i < size_3d; ++i) {
                    sum += global_rho[i];
                    rho_k[i].r = global_rho[i];
                    rho_k[i].i = 0.0;
                }
                sum *= dx * dy * dz;
            
                printf("[%d]TEST:Kerker:2, %.15f\n", GetProcessID(ml_grid.mpi_comm), sum); fflush(stdout);
            }
            
            m_fftw->ForwardDirect(rho_k);
            //auto* rho_k = m_fftw_r2c->GetBuffer();

            printf("[%d]TEST:Kerker:3\n", GetProcessID(ml_grid.mpi_comm)); fflush(stdout);

            
            ConvertKerker_GlobalK(rho_k, paramA / (double)(m_global_grid.Size3D()), paramG0, m_global_grid.SizeX(), m_global_grid.SizeY(), m_global_grid.SizeZ(), dx, dy, dz, m_global_grid.SizeX() / 2 + 1);

            printf("[%d]TEST:Kerker:4\n", GetProcessID(ml_grid.mpi_comm)); fflush(stdout);
        
            m_fftw->BackwardDirect(rho_k);
            
            {
                const int size_3d = m_global_grid.Size3D();
                double sum = 0.0;
                for (int i = 0; i < size_3d; ++i) {
                    global_rho[i] = rho_k[i].r;
                    sum += global_rho[i];
                }
                
                sum *= dx * dy * dz;

                printf("[%d]TEST:Kerker:5, %.15f\n", GetProcessID(ml_grid.mpi_comm), sum); fflush(stdout);
            }
            
        }
        
        MPI_Barrier(ml_grid.mpi_comm);
        ScatterGrid(ml_grid, l_diff_rho, m_global_grid, global_rho, 0);
        printf("[%d]TEST:Kerker:6\n", GetProcessID(ml_grid.mpi_comm)); fflush(stdout);
        MPI_Barrier(ml_grid.mpi_comm);

        if(global_rho) gy::AlignedFree(global_rho);
        ++m_step;
    }

    static inline
    void ConvertKerker_GlobalK(OneComplex* rhok, double paramA, double paramG0, int size_x, int size_y, int size_z, double dx, double dy, double dz,
            int kx_end = 0) {

        //kx_end is used for FFTW_R2C mode//
        if (kx_end == 0) {
            kx_end = size_x;
        }
        const int kz_end = size_z;

               
        
        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));
        
        auto SQ = [](double x) { return x * x; };

        const double k2_0 = SQ(paramG0 * std::min(std::min(coef1_x, coef1_y), coef1_z));

        //#define NO_FOLD

        for (int kz = 0; kz < kz_end; ++kz) {

            const double kz2 = SQ(coef1_z * (double)(kz < size_z / 2 ? kz : kz - size_z));
            //const double kz2 = SQ((double)(kz < size_z / 2 ? kz : kz - size_z));

            for (int ky = 0; ky < size_y; ++ky) {
                const double ky2 = SQ(coef1_y * (double)(ky < size_y / 2 ? ky : ky - size_y));
                //const double ky2 = SQ((double)(ky < size_y / 2 ? ky : ky - size_y));

                for (int kx = 0; kx < kx_end; ++kx) {

                    const double kx2 = SQ(coef1_x * (double)(kx < size_x / 2 ? kx : kx - size_x));
                    //const double kx2 = SQ((double)(kx < size_x / 2 ? kx : kx - size_x));

                    const double k2 = (kx2 + ky2 + kz2);

                    const size_t i = kx + kx_end * (ky + (size_y * kz));

                    if (kx == 0 && ky == 0 && kz == 0) {
                        //rhok[i][0] *= paramA;
                        //rhok[i][1] *= paramA;
                        rhok[i].r = 0.0;
                        rhok[i].i = 0.0;
                    } else {
                        const double factor = k2 / (k2 + k2_0);
                        rhok[i].r *= paramA * factor;
                        rhok[i].i *= paramA * factor;
                    }
                }

            }
        }
    }

};
