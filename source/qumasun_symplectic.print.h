#pragma once
#include <mpi.h>
#include "qumasun_symplectic.h"

inline
void QUMASUN_SYMPLECTIC::mPrintTime() {
    const bool is_root = IsRoot(m_mpi_comm);
    if (is_root) {


        printf("\nCalculation time====================\n");
        const double total_tm = watch.Total();
        //const double calc_tm = watch.Total({ 2,3,4,6,10,11,12,13,14,15,16,17,18,19 });
        const double prepare_core_tm = watch.Total({ 1,22,23,24,25,26,27,28,29 });
        const double move_nucl_tm = watch.Total({45, 20,21 });
        const double prepare_core_hr_tm = watch.Total({ 31,32,33,34,35,36,37,38 });
        const double init_state_tm = watch.Total({ 0,40,41 });
        const double init_tm = watch.Total({ 0,40,41,7 });
        const double output_tm = watch.Total({ 42,43,44 });
        const double potential_tm = watch.Total({ 4,50,51,52,53,54,55,56,57,58 });
        const double force_tm = watch.Total({ 8,70,71,72,73,74,75,76 });
        const double energy_tm = watch.Total({ 6,60,61,62,63,64,65,66,67 });
        const double symplectic_tm = watch.Total({ 80, 81, 82,83,84,85,86,87,88 });
        const double calc_tm = total_tm - init_tm - output_tm;

        printf("Total time      : %f [s]\n", total_tm);
        printf("  Output          : %f [s]\n", output_tm);
        printf("mInitialState   : %f [s]\n", init_state_tm);
        watch.Print("--LoadState     :", 40);
        watch.Print("--ScatterState  :", 41);
        watch.Print("InitialDensity  :", 7);

        
        printf("\nCalculation time: %f [s]\n", calc_tm);
        printf("mPrepareCore    : %f [s]\n", prepare_core_tm);
        //watch.Print("mPrepareCore    :", 1);
        //watch.Print("--ChargeVlocal  :", 21);
        //watch.Print("--Scaling       :", 22);
        //watch.Print("--GatherRho     :", 23);
        //watch.Print("--PoissonVlocal :", 24);
        watch.Print("--ScatterVlocal :", 25);
        watch.Print("--SetPccCharge  :", 26);
        watch.Print("--GatherPCC     :", 27);
        watch.Print("--GetValence    :", 28);
        watch.Print("--EnnCorrection :", 29);

        printf("mPrepareCore(HR): %f [s]\n", prepare_core_hr_tm);
        watch.Print("--ChargeVlocal  :", 31);
#ifdef DDM_FFT
        watch.Print("--Exchange-Fwd  :", 33);
        watch.Print("--PoissonVlocal :", 34);
        watch.Print("--Exchange-Bwd  :", 35);
        watch.Print("--Transpose3    :", 32);
        watch.Print("--DownConvVlocal:", 36);
        watch.Print("--FFT/IFFT      :", 38);
        watch.Print("--etc           :", 37);

#else
        watch.Print("--GatherRho     :", 33);
        watch.Print("--PoissonVlocal :", 34);
        watch.Print("--ScatterVlocal :", 35);
        watch.Print("--DownConvVlocal:", 36);
        watch.Print("--GradInKspase  :", 37);
        watch.Print("--IFFT          :", 38);
#endif

        printf("MoveNuclei      : %f [s]\n", move_nucl_tm);
        watch.Print("--Barrier       :", 45);
        watch.Print("--UpdatePosLocal:", 20);
        watch.Print("--UpdatePosNL   :", 21);

        watch.Print("mSetOccupancy   :", 2);
        watch.Print("mSetDensity     :", 3);
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
        watch.Print("--in-loop       :", 6);
        watch.Print("--in-test       :", 67);

        printf("mGetForce       : %f [s]\n", force_tm);
        watch.Print("--Nonlocal(Diff):", 75);
        watch.Print("--Nonlocal(Inner:", 76);
        watch.Print("--Nonlocal(other:", 70);
        watch.Print("--DiffV         :", 71);
        watch.Print("--InnerNuclRho  :", 72);
        watch.Print("--DiffV(HR)     :", 73);
        watch.Print("--InnerNucl(HR) :", 74);
        watch.Print("--in-loop       :", 8);

        printf("Symplectic Integrator : %f [s]\n", symplectic_tm);
        watch.Print("--Kinetic       :", 80);
        watch.Print("--K-gather      :", 83);
        watch.Print("--K-Laplacian   :", 84);
        watch.Print("--K-scatter     :", 85);
        watch.Print("--K-SoA2AoS     :", 86);
        watch.Print("--K-fft-forward :", 87);
        watch.Print("--K-fft-backward:", 88);
        watch.Print("--Potential     :", 81);
        watch.Print("--PP-nonlocal   :", 82);

        printf("TimeEvolusionState     :--\n");
        //watch.Print("EigenSolver     :", 5);
        watch.Print("--K-operation   :", 11);
        watch.Print("--V-operation   :", 12);
        watch.Print("--PPNonlocal    :", 13);
        watch.Print("--others        :", 10);

        printf("Output                 :--\n");
        watch.Print("--Gather        :", 43);
        watch.Print("--WriteCube     :", 44);
        watch.Print("--others        :", 42);
#ifdef _DEBUG
        printf("debug All timer:--\n");
        for (int i = 0; i < watch.num_sector; ++i) {
            watch.Print(std::to_string(i).c_str(), i);
        }
#endif
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




inline
void QUMASUN_SYMPLECTIC::PrintTime() {
    mPrintTime();
}
