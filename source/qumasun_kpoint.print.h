#pragma once
#include <mpi.h>
#include "qumasun_kpoint.h"

inline
void QUMASUN_KPOINT::mPrintTime() {
    const bool is_root = IsRoot(m_mpi_comm);
    

    //for performance////////////////////////////////////////////////////////////
    auto GemmGFLOPS = [](int N, int M, int K, int64_t count, double time) {
        return (double)count * (double)N * (double)M * (double)K * 1.0e-9 / time;//factor 2 means fma//
        };

    double gemm_flops[3] = {
        GemmGFLOPS(2 * num_solution, 2 * num_solution, ml_grid.Size3D(), watch.GetCount(17), watch.GetTime(17)),
        GemmGFLOPS(ml_grid.Size3D(), num_solution, num_solution * 2, watch.GetCount(16), watch.GetTime(16)),
        GemmGFLOPS(num_solution, num_solution, num_solution, 4 * watch.GetCount(19), watch.GetTime(19))
    };
    double total_gemm_flops[3]{ 0.0 };
    MPI_Reduce(gemm_flops, total_gemm_flops, 3, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
#if defined( ZGEMM_3_MPI1D) || defined( ZGEMM_BLASMP)
    total_gemm_flops[2] /= (double)GetNumProcess(m_ddm_comm);
    gemm_flops[2] /= (double)GetNumProcess(m_ddm_comm);
#endif

    if (is_root) {

        printf("\nPeak performance in GEMM=====================\n");
        printf("  DGEMM1(x2HS)   : %f GFLOPS  (%f GFLOPS/root_proc)\n", total_gemm_flops[0], gemm_flops[0]);
        printf("  DGEMM2(x2x)    : %f GFLOPS  (%f GFLOPS/root_proc)\n", total_gemm_flops[1], gemm_flops[1]);
        if (watch.GetCount(19) > 0) {
            printf("  ZGEMM3(HS2HS)  : %f GFLOPS  (%f GFLOPS/root_proc)\n", total_gemm_flops[2], gemm_flops[2]);
        } else {
            printf("  ZGEMM3(HS2HS)  : no-called\n");
        }

        printf("\nCalculation time====================\n");
        const double total_tm = watch.Total();
        //const double calc_tm = watch.Total({ 2,3,4,6,10,11,12,13,14,15,16,17,18,19 });
        const double prepare_core_tm = watch.Total({ 1,21,22,23,24,25,26,27,28,29 });
        const double prepare_core_hr_tm = watch.Total({ 31,33,34,35,36,37,38 });
        const double potential_tm = watch.Total({ 4,50,51,52,53,54,55,56,57,58 });
        const double force_tm = watch.Total({ 8,70,71,72,73,74,75,76 });
        const double energy_tm = watch.Total({ 6,60,61,62,63,64,65,66});
        const double output_tm = watch.Total({ 42,43,44 });
        const double move_nucl_tm = watch.Total({ 45, 20,21 });
        const double density_tm = watch.Total({ 3, 30 });
        const double init_tm = watch.Total({ 0,7 }) + prepare_core_tm + prepare_core_hr_tm;
        const double calc_tm = total_tm - init_tm - output_tm - move_nucl_tm;

        printf("Total time      : %f [s]\n", total_tm);
        printf("  Output        : %f [s]\n", output_tm);
        watch.Print("mInitialState   :", 0);
        printf("mPrepareCore    : %f [s]\n", prepare_core_tm);
        //watch.Print("mPrepareCore    :", 1);
        //watch.Print("--ChargeVlocal  :", 21);
        watch.Print("--GatherRho     :", 23);
        watch.Print("--PoissonVlocal :", 24);
        watch.Print("--ScatterVlocal :", 25);
        watch.Print("--SetPccCharge  :", 26);
        watch.Print("--GatherPCC     :", 27);
        watch.Print("--GetValence    :", 28);
        watch.Print("--EnnCorrection :", 29);

        printf("mPrepareCore(HR): %f [s]\n", prepare_core_hr_tm);
        watch.Print("--ChargeVlocal  :", 31);
        watch.Print("--GatherRho     :", 33);
        watch.Print("--PoissonVlocal :", 34);
        watch.Print("--ScatterVlocal :", 35);
        watch.Print("--DownConvVlocal:", 36);
        watch.Print("--GradInKspase  :", 37);
        watch.Print("--IFFT          :", 38);
        watch.Print("--stanspose     :", 32);

        watch.Print("InitialDensity  :", 7);
        printf("MoveNuclei      : %f [s]\n", move_nucl_tm);
        watch.Print("--Barrier       :", 45);
        watch.Print("--UpdatePosLocal:", 20);
        watch.Print("--UpdatePosNL   :", 21);


        printf("\nCalculation time: %f [s]\n", calc_tm);
        watch.Print("mSetOccupancy   :", 2);
        printf("mSetDensity     : %f [s]\n", density_tm);
        watch.Print("--PsiToRho      :", 3);
        watch.Print("--mixing        :", 30);
        printf("mSetPotential   : %f [s]\n", potential_tm);
#ifdef DDM_FFT
        watch.Print("--Exchange-Fwd  :", 53);
        watch.Print("--PoissonVhart  :", 54);
        watch.Print("--Exchange-Bwd  :", 51);
        watch.Print("--Transpose3    :", 52);
        watch.Print("--UpConvVlocal  :", 56);
        watch.Print("--FFT/IFFT      :", 58);
        watch.Print("--etc           :", 57);

        //watch.Print("--PoissonVhart  :", 50);
        //watch.Print("--UpConvert(HR) :", 51);
        //watch.Print("--IFFT(HR)      :", 52);
        //watch.Print("--Normalize(HR) :", 53);
        //watch.Print("--Check(HR)     :", 54);
        watch.Print("--XC            :", 55);
        //watch.Print("--SumTotal      :", 56);
        //watch.Print("--Scatter       :", 57);
        watch.Print("--in-loop       :", 4);
#else
        watch.Print("--PoissonVhart  :", 50);
        watch.Print("--UpConvert(HR) :", 51);
        watch.Print("--IFFT(HR)      :", 52);
        watch.Print("--Normalize(HR) :", 53);
        watch.Print("--Check(HR)     :", 54);
        watch.Print("--XC            :", 55);
        watch.Print("--SumTotal      :", 56);
        watch.Print("--Scatter       :", 57);
        watch.Print("--in-loop       :", 4);
#endif
        printf("mGetTotalEnergy : %f [s]\n", energy_tm);
        watch.Print("--Kinetic       :", 60);
        watch.Print("--Nonlocal      :", 61);
        watch.Print("--InnerVRho     :", 62);
        watch.Print("--InnerVRho(HR) :", 63);
        watch.Print("--XC            :", 64);
        watch.Print("--VhartAtPoint  :", 65);
        watch.Print("--CoreCoreDirect:", 66);

        printf("mGetForce       : %f [s]\n", force_tm);
        watch.Print("--Nonlocal(Diff):", 75);
        watch.Print("--Nonlocal(Inner:", 76);
        watch.Print("--Nonlocal(other:", 70);
        watch.Print("--DiffV         :", 71);
        watch.Print("--InnerNuclRho  :", 72);
        watch.Print("--DiffV(HR)     :", 73);
        watch.Print("--InnerNucl(HR) :", 74);
        watch.Print("--in-loop       :", 8);

        printf("EigenSolver     :--\n");
        
        double k_ope_fft_tm;
        if (m_algorithm_Laplacian == 1) {
            k_ope_fft_tm = watch.Total({ 11,82,83,84,85,86,87,88 });
            printf("--K-operation(1): %f [s]\n", k_ope_fft_tm);
            printf("  --gather      :\n");
            watch.Print("   -calc.size   :", 82);
            watch.Print("   -AlltoAllv   :", 83);
            watch.Print("   -rearrange   :", 84);
            watch.Print("   -fft-forward :", 87);
            watch.Print("  --Laplacian   :", 11);
            printf("  --scatter     :\n");
            watch.Print("   -fft-backward:", 88);
            watch.Print("   -rearrange   :", 85);
            watch.Print("   -AlltoAllv   :", 86);
        } else if (m_algorithm_Laplacian == 2) {
            k_ope_fft_tm = watch.Total({ 11,90,91,92,93,94 });
            printf("--K-operation(2): %f [s]\n", k_ope_fft_tm);
            watch.Print("  --ExchangeFowd:", 91);
            watch.Print("  --FFT & Lap   :", 92);
            watch.Print("  --ExchangeBack:", 93);
            watch.Print("  --transpose   :", 94);
            watch.Print("  --SoA2AoS     :", 90);
        } else {
            k_ope_fft_tm = watch.Total({ 11 });
            watch.Print("--K-operation   :", 11);            
        }
        watch.Print("--V-operation   :", 12);
        watch.Print("--PPNonlocal    :", 13);
        watch.Print("--GEMM1(x2HS)   :", 17);
        watch.Print("--GEMM2(x2x)    :", 16);
        watch.Print("--GEMM3(HS2HS)  :", 19);
        watch.Print("--LAPACK        :", 15);
        watch.Print("--MPI           :", 14);
        watch.Print("--others        :", 10);
        watch.Print("--wait-imbalance:", 18);
        watch.Print("--call          :", 5);

        //for bench
             printf("bench-Laplacian : %f [s]\n", k_ope_fft_tm + watch.GetTime(60));
             printf("bench-GEMM      : %f [s]\n", watch.Total({ 16, 17, 19 }));
             printf("bench-Eigen     : %f [s]\n", watch.GetTime(15));
             printf("bench-Nonlocal  : %f [s]\n", watch.GetTime(13));
             printf("bench-Others    : %f [s]\n", calc_tm - k_ope_fft_tm - watch.Total({ 13,15,16, 17, 19,60 }));
    }


    {
        double d_flops[2];
        m_pp_SvF.BenchFlopsNonlocal(d_flops);
        double mean_flops[2]{ 0.0 };
        double max_flops[2]{ 0.0 };
        double min_flops[2]{ 0.0 };
        MPI_Reduce(d_flops, mean_flops, 2, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
        MPI_Reduce(d_flops, max_flops, 2, MPI_DOUBLE, MPI_MAX, 0, m_mpi_comm);
        MPI_Reduce(d_flops, min_flops, 2, MPI_DOUBLE, MPI_MIN, 0, m_mpi_comm);
        double d_num_procs = (double)GetNumProcess(m_mpi_comm);
        mean_flops[0] /= d_num_procs;        
        mean_flops[1] /= d_num_procs; 
        if (is_root) {
            printf("\n\nPerformance in Pseudo Pptential--\n");
            printf("  Nonlocal-Inner : %f GFLOPS/proc(mean)\n", mean_flops[0] * 1.e-9);
            printf("                 : %f GFLOPS/proc(max)\n", max_flops[0] * 1.e-9);
            printf("                 : %f GFLOPS/proc(min)\n", min_flops[0] * 1.e-9);
            /*
            printf("  Nonlocal-force : %f GFLOPS/proc(mean)\n", mean_flops[1] * 1.e-9);
            printf("                 : %f GFLOPS/proc(max)\n", max_flops[1] * 1.e-9);
            printf("                 : %f GFLOPS/proc(min)\n", min_flops[1] * 1.e-9);
            */
            printf("\n");
        }
    }

}



