#pragma once
#include "mpi.h"
#include <cfloat>
#include <memory>
#include <vector>
#include "vec3.h"
#include "nucleus.h"


    class DynamicScaling1
    {
    private:
        //double m_lower_force_limit = 1.0e-2;
        std::vector<double> m_Energy_list;
        std::vector<double> m_scaling_factor_list;
        std::vector<double> m_box_x_list;

        MPI_Comm m_mpi_comm;
        DynamicScaling1() = delete;
    public:

        DynamicScaling1(MPI_Comm mpi_comm) :
            m_mpi_comm(mpi_comm)
        {

        };

        /*
        指定したステップだけ構造緩和を行う
        */
        template<class QUMASUN_T>
        void Evolve(QUMASUN_T& qumasun, double scale_begin, double scale_end, const int num_steps, int num_nuclei, Nucleus* nuclei, double* box_axis)
        {
            const bool is_root = IsRoot(m_mpi_comm);

            auto nuclei_scaled = std::make_unique<Nucleus[]>(num_nuclei);
            double box_scaled[3];
            
            for (int istep = 0; istep < num_steps; istep++) {
                const double ratio = (double)istep / (double)(num_steps - 1);
                double scaling_factor = (scale_end - scale_begin) * ratio + scale_begin;
                
                for (int i = 0; i < num_nuclei; i++) {
                    nuclei_scaled[i].Z = nuclei[i].Z;
                    nuclei_scaled[i].Rx = nuclei[i].Rx * scaling_factor;
                    nuclei_scaled[i].Ry = nuclei[i].Ry * scaling_factor;
                    nuclei_scaled[i].Rz = nuclei[i].Rz * scaling_factor;
                }

                box_scaled[0] = box_axis[0] * scaling_factor;
                box_scaled[1] = box_axis[4] * scaling_factor;
                box_scaled[2] = box_axis[8] * scaling_factor;

                if (is_root) {
                    printf("====================================\n");
                    printf("Scaling: %d\n", istep);
                    printf("  Scaling_factor = %f\n", scaling_factor);
                    printf("  box_x = %f\n", box_scaled[0]);
                    fflush(stdout);
                }



                MPI_Bcast(&box_scaled[0], sizeof(double) * 3, MPI_BYTE, 0, m_mpi_comm);
                MPI_Bcast(&nuclei_scaled[0], sizeof(Nucleus) * num_nuclei, MPI_BYTE, 0, m_mpi_comm);
                qumasun.ResetBox(box_scaled[0], box_scaled[1], box_scaled[2], num_nuclei, &nuclei_scaled[0]);


                qumasun.Execute(istep);
                double U = qumasun.GetTotalEnergy();

                if (is_root) {
                    printf("  Etot = %f\n", U);
                    fflush(stdout);
                }

                m_Energy_list.push_back(U);
                m_scaling_factor_list.push_back(scaling_factor);
                m_box_x_list.push_back(box_scaled[0]);


                std::string cube_path("test_eigenvalue_");
                cube_path += std::to_string(istep) + ".txt";
                qumasun.OutputEigenValue(cube_path.c_str());

            }//end of loop "istep"



        }

        void OutputLog(const char* filepath) {
            if (!IsRoot(m_mpi_comm)) return;

            const int i_end = (int)m_Energy_list.size();

            printf("  step = %d\n", i_end);
            fflush(stdout);


            FILE* fp = fopen(filepath, "w");
            fprintf(fp, "#step, scaling_factor, box_x, Etot\n");
            for (int i = 0; i< i_end; ++i) {
                fprintf(fp, "%d\t%.10f\t%.10f\t%.10f\n", i, m_scaling_factor_list[i], m_box_x_list[i], m_Energy_list[i]);
            }
            fclose(fp);
        };


    
    };

