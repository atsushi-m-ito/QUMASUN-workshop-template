#pragma once
#include <map>
#include <vector>
#include "qumasun_input.h"
#include "JobParam.h"
#include "folding.h"
#include "physical_param.h"
#include "atomic_number.h"
#include "symmetry_checker.h"
#include "grid_cutoff_energy.h"
#include "upf_oncv_loader.h"
#include "vps_loader.h"
#include "qps_loader.h"

namespace QUMASUN {

	void SetSupercell(JobParam& job_params, Input& input)
	{
		//super-cellの読み込み//	
		int super_cell[3];
		job_params.GetIntArray("Material.SuperCell", super_cell, 3);
		//generator.SetSuperCell(super_cell[0], super_cell[1], super_cell[2]);
		const int num_total_cells = super_cell[0] * super_cell[1] * super_cell[2];

		//unit-lattice-vectorの読み込み//
		constexpr int LINE_SIZE = 1024;
		const char DELIMITER[] = " \t\r\n";
		char line[LINE_SIZE];
		struct V3 {
			double x, y, z;
		};
		V3 unit_vector[3];
		for (int i = 0; i < 3; ++i) {
			strcpy(line, job_params.GetBlockString("Material.UnitLatticeVector", i));

			char* p = strtok(line, DELIMITER);
			unit_vector[i].x = strtod(p, nullptr);

			p = strtok(nullptr, DELIMITER);
			unit_vector[i].y = strtod(p, nullptr);

			p = strtok(nullptr, DELIMITER);
			unit_vector[i].z = strtod(p, nullptr);
		}

		
		if (job_params.IsEqualString("Material.UnitLatticeVector.Unit", "Ang")) {
			//default is angstrome//
			for (int i = 0; i < 3; ++i) {
				unit_vector[i].x *= Ang_as_Bohr;
				unit_vector[i].y *= Ang_as_Bohr;
				unit_vector[i].z *= Ang_as_Bohr;
			}
		}

        double vacuum_cell_left[3] = { 0.0 };
        double vacuum_cell_right[3] = { 0.0 };
        if (job_params.GetDoubleArray("Material.VacuumCell.Leftside", vacuum_cell_left, 3)) {
            //generator.SetVacuumCellLeft(vacuum_cell[0], vacuum_cell[1], vacuum_cell[2]);
        }
        if (job_params.GetDoubleArray("Material.VacuumCell.Rightside", vacuum_cell_right, 3)) {
            //generator.SetVacuumCellRight(vacuum_cell[0], vacuum_cell[1], vacuum_cell[2]);
        }

        
		{
            double sx = (double)super_cell[0] + vacuum_cell_left[0] + vacuum_cell_right[0];
			input.box_axis[0] = sx * unit_vector[0].x;
			input.box_axis[1] = sx * unit_vector[0].y;
			input.box_axis[2] = sx * unit_vector[0].z;

            double sy = (double)super_cell[1] + vacuum_cell_left[1] + vacuum_cell_right[1];
			input.box_axis[3] = sy * unit_vector[1].x;
			input.box_axis[4] = sy * unit_vector[1].y;
			input.box_axis[5] = sy * unit_vector[1].z;

            double sz = (double)super_cell[2] + vacuum_cell_left[2] + vacuum_cell_right[2];
			input.box_axis[6] = sz * unit_vector[2].x;
			input.box_axis[7] = sz * unit_vector[2].y;
			input.box_axis[8] = sz * unit_vector[2].z;
		}




		//粒子数の読み込み//
		const int num_unit_atoms = job_params.GetBlockNumLines("Material.UnitCell");
		std::vector<Nucleus> atom_u_pos;
		for (int i = 0; i < num_unit_atoms; i++) {
			strcpy(line, job_params.GetBlockString("Material.UnitCell", i));

			char* p = strtok(line, DELIMITER);
			const int Z = msz::GetAtomicNumber(p);

			char* sx = strtok(nullptr, DELIMITER);
			char* sy = strtok(nullptr, DELIMITER);
			char* sz = strtok(nullptr, DELIMITER);

			//double unit_mass = ATOMIC_MASS[unit_element];

			//generator.AddIntoUnitcell(unit_element, unit_mass,
			atom_u_pos.emplace_back(Nucleus{ Z, strtod(sx, nullptr), strtod(sy, nullptr), strtod(sz, nullptr) });
		}

		if (job_params.IsEqualString("Material.UnitCell.Unit", "Ang")) {
			for (auto&& nuc : atom_u_pos) {
				nuc.Rx *= Ang_as_Bohr;
				nuc.Ry *= Ang_as_Bohr;
				nuc.Rz *= Ang_as_Bohr;
			}
		} else if (job_params.IsEqualString("Material.UnitCell.Unit", "FRAC")) {
			for (auto&& nuc : atom_u_pos) {
				const double rx = nuc.Rx * unit_vector[0].x + nuc.Ry * unit_vector[1].x + nuc.Rz * unit_vector[2].x;
				const double ry = nuc.Rx * unit_vector[0].y + nuc.Ry * unit_vector[1].y + nuc.Rz * unit_vector[2].y;
				const double rz = nuc.Rx * unit_vector[0].z + nuc.Ry * unit_vector[1].z + nuc.Rz * unit_vector[2].z;
				nuc.Rx = rx;
				nuc.Ry = ry;
				nuc.Rz = rz;
			}
		}


		input.num_nuclei = num_unit_atoms * num_total_cells;
		for (int cz = 0; cz < super_cell[2]; ++cz) {
			for (int cy = 0; cy < super_cell[1]; ++cy) {
				for (int cx = 0; cx < super_cell[0]; ++cx) {
					const double rx = (vacuum_cell_left[0] + (double)cx) * unit_vector[0].x + (vacuum_cell_left[1] + (double)cy) * unit_vector[1].x + (vacuum_cell_left[2] + (double)cz) * unit_vector[2].x;
					const double ry = (vacuum_cell_left[0] + (double)cx) * unit_vector[0].y + (vacuum_cell_left[1] + (double)cy) * unit_vector[1].y + (vacuum_cell_left[2] + (double)cz) * unit_vector[2].y;
					const double rz = (vacuum_cell_left[0] + (double)cx) * unit_vector[0].z + (vacuum_cell_left[1] + (double)cy) * unit_vector[1].z + (vacuum_cell_left[2] + (double)cz) * unit_vector[2].z;

					for (int k = 0; k < num_unit_atoms; ++k) {
						
						input.nuclei.emplace_back(Nucleus{ atom_u_pos[k].Z, 
							atom_u_pos[k].Rx + rx, +atom_u_pos[k].Ry + ry, +atom_u_pos[k].Rz + rz});
					}
				}
			}
		}
		
        for (auto&& nuc : input.nuclei) {
            Folding(nuc.Rx, nuc.Ry, nuc.Rz, input.box_axis);
        }
	}

    std::string GetDirectoryPath(const char* filepath) {
        const char* p = std::strrchr(filepath, '/');
        if (p == nullptr) {
            p = std::strrchr(filepath, '\\');
            if (p == nullptr) {
                return std::string();
            }
        }

        return std::string(filepath).substr(0, p - filepath + 1);
    }

    std::string ConnectPath(const std::string& dirpath, const std::string& filepath) {
        if (dirpath.size() == 0)return filepath;
        std::string dir_file_path( dirpath + filepath);
        if (dirpath[dirpath.size() - 1] == '\\') {
            for (auto&& a : dir_file_path) {
                if (a == '/') {
                    a = '\\';
                }
            }
        }        
        return dir_file_path;
    }

    /*
    * System.Initial.StateにDirectoryが指定された場合にファイルリストに展開
    * num_solutionsも実際に読み込まれるファイル数に自動で変更される
    */
    int DecodeInitialStateDir(Input& input, const std::string& dir_path) {
        if (input.initial_state.empty()) return 0;//noting to do

        //directory指定が一つでもあるか確認//
        {
            bool is_dir_found = false;
            for (auto& dir : input.initial_state) {
                if (dir[dir.size() - 1] == '/') {
                    is_dir_found = true;
                    break;
                }
            }
            if (is_dir_found == false) {
                //directory指定では無い場合だが//
                //num_solutionとファイル数があっているかチェック//
                int factor = (input.spin_polarization == 0) ? 2 : 4;

                if (input.num_solutions != input.initial_state.size() / factor) {
                    printf("ERROR: num_solution does not equal to num of initial state file, %d != %d\n", input.num_solutions, (int)(input.initial_state.size() / factor));
                    input.num_solutions = (int)input.initial_state.size() / factor;
                }

                return 0;//ディレクトリのdecodeはしていないので0//
            }
        }

        const int is_spin_polarize = input.spin_polarization;
        std::vector<std::string>& dir_list_org = input.initial_state;
        std::vector<std::string> state_files_up;
        std::vector<std::string> state_files_down;
        
        const bool is_tune_occupancy = (input.dynamics_mode == DynamicsMode::SymplecticEhrenfestMD)
            || (input.dynamics_mode == DynamicsMode::SemiSymplecticTest1)
            || (input.dynamics_mode == DynamicsMode::SemiSymplecticTest2)
            || (input.dynamics_mode == DynamicsMode::SemiSymplecticTest3);

        for (auto& dir : dir_list_org) {
            if (dir[dir.size() - 1] != '/') {
                printf("ERROR: mix of state file and directory is not supported\n");
            } else {

                std::string actual_dir;
                std::string key;
                if (dir[0] == '[') {
                    auto pos = dir.find(']');
                    key = dir.substr(0, pos + 1);
                    actual_dir = dir.substr(pos + 1);
                } else {
                    actual_dir = dir;
                }

                std::vector<std::string> state_files_tmp;
                constexpr int LINE_SIZE = 1024;
                char line[LINE_SIZE];
                char DELIMITER[] = " \t\r\n,";
                {
                    std::string file_list_path = actual_dir + "state_file_list.txt";
                    FILE* fp = fopen(file_list_path.c_str(), "r");
                    if (fp == nullptr) {

                        //re-try by adding directory path//
                        std::string dir_file_path = ConnectPath(dir_path, file_list_path);
                        fp = fopen(dir_file_path.c_str(), "r");
                        if (fp == nullptr) {
                            //error//
                            printf("Cannot find initial state: %s\n", file_list_path.c_str());
                            continue;
                        }

                        actual_dir = ConnectPath(dir_path, actual_dir);
                    }


                    while (fgets(line, LINE_SIZE, fp) != nullptr) {
                        state_files_tmp.push_back(key + actual_dir + strtok(line,DELIMITER));
                    }
                    fclose(fp);
                }

                
                //occupancyの読み込み//////////////////////////////////////////////
                std::string eigen_path = actual_dir + "test_eigenvalue.txt";
                FILE* fp = fopen(eigen_path.c_str(), "r");
                if (fp == nullptr) {
                    //error//
                    printf("Cannot find initial state: %s\n", eigen_path.c_str());
                    continue;                    
                }


                struct EigenOrder {
                    int index;
                    int spin;
                    double eigen;
                    double occupancy;
                };
                std::vector<EigenOrder> eigen_list;

                double ve_tot=0.0;
                bool has_spin = false;
                double re_weight_by_k_sample = 1.0; //Bloch定理を利用してk点を111に展開して読み込む場合のoccupancyのre-weighting
                int k_sample_x = 0;
                int k_sample_y = 0;
                int k_sample_z = 0;
                {
                    int index = 0;
                    int up_or_dn = 0;  //0 or 1
                    while (fgets(line, LINE_SIZE, fp) != nullptr) {
                        if (strncmp(line, "E", 1) == 0) {
                            if (index == 0) {
                                //k-sampleing点サイズ読み込み
                                char* p = strchr(line, ':');
                                p = strchr(p + 1, '/'); 
                                k_sample_x = strtol(p+1, &p, 10);
                                p = strchr(p + 1, '/');
                                k_sample_y = strtol(p + 1, &p, 10);
                                p = strchr(p + 1, '/');
                                k_sample_z = strtol(p + 1, &p, 10);
                                if (input.initial_expand_from_kpoint) {
                                    re_weight_by_k_sample = (double)(k_sample_x * k_sample_y * k_sample_z);
                                }
                            }
                            char* p = strrchr(line, ',');
                            if (strncmp(p + 2, "up", 2) == 0) {
                                up_or_dn = 0;
                            } else {
                                up_or_dn = 1;
                                has_spin = true;
                            }
                        } else {
                            char* p = line;
                            int id = strtol(p, &p, 10);
                            double ei = strtod(p, &p);
                            double occ = strtod(p, &p);                            
                            occ *= re_weight_by_k_sample;                            
                            ve_tot += occ;
                            eigen_list.push_back(EigenOrder{ index,up_or_dn,ei,occ });
                            ++index;
                        }
                    }
                }
                fclose(fp);


                
                if(is_tune_occupancy){
                    //sortしてoccupiedを0 or 1(2)にする, TDDFT向け//
                    int num_ve = (int)floor(ve_tot + 0.5);
                    

                    std::sort(eigen_list.begin(), eigen_list.end(), [](const EigenOrder& a, const EigenOrder& b) {
                        return a.eigen < b.eigen;
                        });
                    

                    if (has_spin) {
                        for (int i = 0; i < num_ve; ++i) {
                            eigen_list[i].occupancy = 1.0;
                        }
                        for (int i = num_ve; i < eigen_list.size(); ++i) {
                            eigen_list[i].occupancy = 0.0;
                        }
                    } else {
                        for (int i = 0; i < num_ve / 2; ++i) {
                            eigen_list[i].occupancy = 2.0;
                        }
                        for (int i = num_ve / 2; i < eigen_list.size(); ++i) {
                            eigen_list[i].occupancy = 0.0;
                        }
                        if (num_ve & 0x1) {
                            eigen_list[num_ve / 2].occupancy = 1.0;
                        }
                    }
                

                    std::sort(eigen_list.begin(), eigen_list.end(), [](const EigenOrder& a, const EigenOrder& b) {
                        return a.index < b.index;
                        });
                    

                    //occupancy追加//
                    if (has_spin) {

                        int64_t index = input.initial_occupancy.size();
                        for (const auto& a : eigen_list) {
                            if (a.spin == 0) {
                                input.initial_occupancy.push_back(OccupancyInfo{ 1, a.occupancy, 0.0 });
                            }
                        }
                        for (const auto& a : eigen_list) {
                            if (a.spin == 1) {
                                input.initial_occupancy[index].occupancy_down = a.occupancy;
                                ++index;
                            }
                        }
                    } else {
                        if (is_spin_polarize) {
                            for (const auto& a : eigen_list) {
                                input.initial_occupancy.push_back(OccupancyInfo{ 1, a.occupancy / 2.0, a.occupancy / 2.0 });
                            }
                        } else {
                            for (const auto& a : eigen_list) {
                                input.initial_occupancy.push_back(OccupancyInfo{ 1, a.occupancy, 0.0});
                            }
                        }
                    }
                }

                //state_fileをup/downに振り分け//
                if (has_spin) {
                    if (is_spin_polarize) {
                        //計算条件も読み込みファイルもspin対応//
                        int index = 0;
                        for (const auto& a : eigen_list) {
                            if (a.spin == 0) {
                                state_files_up.push_back(state_files_tmp[index]);
                                state_files_up.push_back(state_files_tmp[index+1]);
                            } else {
                                state_files_down.push_back(state_files_tmp[index]);
                                state_files_down.push_back(state_files_tmp[index+1]);
                            }
                            index += 2;
                        }
                    } else {
                        //計算条件はspin非対応だが、読み込みファイルはspin対応//
                        //upだけを読み込む//
                        int index = 0;
                        for (const auto& a : eigen_list) {
                            if (a.spin == 0) {
                                state_files_up.push_back(state_files_tmp[index]);
                                state_files_up.push_back(state_files_tmp[index+1]);
                            }
                            index += 2;
                        }
                    }
                } else {
                    //計算条件はspin対応だが、読み込みファイルはspin非対応//
                    //upをdown用としても二重に読み込み//
                    if (is_spin_polarize) {                        
                        state_files_up.insert(state_files_up.end(), state_files_tmp.begin(), state_files_tmp.end());
                        state_files_down.insert(state_files_down.end(), state_files_tmp.begin(), state_files_tmp.end());

                    } else {
                        state_files_up.insert(state_files_up.end(), state_files_tmp.begin(), state_files_tmp.end());
                    }
                }

            }
        }

        input.initial_state.clear();
        input.initial_state.insert(input.initial_state.end(), state_files_up.begin(), state_files_up.end());
        if (!state_files_down.empty()) {
            input.initial_state.insert(input.initial_state.end(), state_files_down.begin(), state_files_down.end());
        }
        {
            //change num_solutions //
            int factor = (input.spin_polarization == 0) ? 2 : 4;

            if (input.num_solutions != input.initial_state.size() / factor) {
                printf("NOTE: num_solution is changed from %d to %d\n", input.num_solutions, (int)(input.initial_state.size() / factor));
                input.num_solutions = (int)input.initial_state.size() / factor;
            }
        }

        return input.num_solutions;
    }

    //擬ポテンシャルを読み込んでve数から必要な電子数を見積る
    int EstimateNumElectrons(Input& input, int system_charge, MPI_Comm& mpi_comm) {
        int ve_total = 0;
        if (IsRoot(mpi_comm)) {
            std::map<int, int>& ve_for_Z = input.ve_for_Z;
            int buf[2];
            for (const auto& pp : input.pseudo_pot_set) {
                auto filepath = pp.second.c_str();
                const int length = std::strlen(filepath);
                int ve = 0;
                if (std::strncmp(filepath + length - 4, ".vps", 4) == 0) {
                    ve = LoadValenceElectronVPS(filepath);
                } else if (std::strncmp(filepath + length - 4, ".qps", 4) == 0) {
                    ve = LoadValenceElectronQPS(filepath);
                } else if (std::strncmp(filepath + length - 4, ".upf", 4) == 0) {
                    ve = LoadValenceElectronUPF_ONCV(filepath);
                }
                ve_for_Z.insert({ pp.first, ve });

                buf[0] = pp.first;
                buf[1] = ve;            
                MPI_Bcast(buf, 2, MPI_INT, 0, mpi_comm);
            }
            //終了シグナル//
            buf[0] = 0;
            buf[1] = 0;
            MPI_Bcast(buf, 2, MPI_INT, 0, mpi_comm);

            
            for (int i = 0; i < input.num_nuclei; ++i) {
                ve_total += ve_for_Z[input.nuclei[i].Z];                
            }

            ve_total -= system_charge;
        } else {

            std::map<int, int>& ve_for_Z = input.ve_for_Z;
            int buf[2]{ 0 };
            MPI_Bcast(buf, 2, MPI_INT, 0, mpi_comm);
            while (buf[0] != 0) {                
                ve_for_Z.insert({ buf[0], buf[1]});
                MPI_Bcast(buf, 2, MPI_INT, 0, mpi_comm);
            }
        }

        MPI_Bcast(&ve_total, 1, MPI_INT,0, mpi_comm);
        input.num_electrons = ve_total;
        
        if (IsRoot(mpi_comm)) {
            printf("NumValenceElectrons: %d\n", ve_total);
            fflush(stdout);
        }
                
        return ve_total;
    }


    inline
    std::vector<int> IntList(const std::string& text) {
        std::vector<int> vals;

        char* endptr = nullptr;
        const char* p = text.c_str();
        while (p) {
            int d = std::strtol(p, &endptr, 10);
            if (p == endptr) {
                break;
            }
            p = endptr;

            vals.push_back(d);
        }
        return vals;
    }


    void GridSizeFromCutoffRy(double cutE, double box_x, double box_y, double box_z, int* divisor, int* grid_size){
        if (cutE > 0.0) {
            double dx = QUMASUN::GridWidthFromCutoffEnergy(cutE / 2.0);
            int Nx = (int)(box_x / dx + 0.5);
            int Ny = (int)(box_y / dx + 0.5);
            int Nz = (int)(box_z / dx + 0.5);
            for (int i = 0; i < 3; ++i) {
                divisor[i] = std::max(1, divisor[i]);
            }

            auto FitN = [](int Nx, int divisor, double box_width, double cutE) {
                Nx = ((Nx + divisor - 1) / divisor) * divisor;
                double E1 = 2.0 * QUMASUN::CutoffEnergyFromGridWidth(box_width / (double)Nx);
                if (E1 > cutE) {
                    double E2 = 2.0 * QUMASUN::CutoffEnergyFromGridWidth(box_width / (double)(Nx - divisor));
                    if (sqrt(E1 - cutE) < sqrt(cutE - E2)) {
                        return Nx;
                    } else {
                        return Nx - divisor;
                    }
                } else {
                    double E3 = 2.0 * QUMASUN::CutoffEnergyFromGridWidth(box_width / (double)(Nx + divisor));
                    if (sqrt(E3 - cutE) < sqrt(cutE - E1)) {
                        return Nx + divisor;
                    } else {
                        return Nx;
                    }
                }
                };

            Nx = FitN(Nx, divisor[0], box_x, cutE);
            Ny = FitN(Ny, divisor[1], box_y, cutE);
            Nz = FitN(Nz, divisor[2], box_z, cutE);
            grid_size[0] = Nx;
            grid_size[1] = Ny;
            grid_size[2] = Nz;
        }
    }

    int MakeInput(const char* input_file, Input& input, MPI_Comm& mpi_comm) {
        JobParam jobparam(input_file, "*.Begin", "*.End");
        {
            auto res = jobparam.GetLastError();
            if (res == JobParam::ErrorCode::CANNOT_OPEN_FILE) {
                return -1;
            } else if (res == JobParam::ErrorCode::INVALID_BEGIN_END_WORD) {
                return -2;
            }
        }

        auto dir_path = GetDirectoryPath(input_file);

        //仕様変更したkeywordをreplace-keyとして登録(内部でキーを置換する)//
        jobparam.ReplaceKey("SCF.Initial.Density", "Initial.Density");
        jobparam.ReplaceKey("SCF.Initial.State", "Initial.State");
        jobparam.ReplaceKey("SCF.NumSolusions", "System.NumSolutions");
        jobparam.ReplaceKey("System.NumSolusions", "System.NumSolutions");
        jobparam.ReplaceKey("Dynamics.Output.PartialCharge", "Dynamics.Output.OrbitalDensity");
        


        constexpr int LINE_SIZE = 1024;
        const char DELIMITER[] = " \t\r\n";
        char line[LINE_SIZE];

        //condition of SCF solver///////////////
        input.scf_step = jobparam.GetInt("SCF.Step", 1);
        input.eigen_step_per_scf = jobparam.GetInt("SCF.EigenSolver.Step", 3);
        input.eigen_step_initial = jobparam.GetInt("SCF.EigenSolver.InitialStep", input.eigen_step_per_scf);
        const char* hamiltonian_name = jobparam.GetString("System.Hamiltonian");
        input.hamiltonian_type = ToHamiltonianType(hamiltonian_name ? hamiltonian_name : "KohnSham_PP");
        if (jobparam.IsEqualString("System.XC", "LDA") || jobparam.IsEqualString("System.XC", "LSDA")) {
            input.xc_type = XC_TYPE::LDA;
            input.xc_model = XC_MODEL::LDA_CA;
        } else if (jobparam.IsEqualString("System.XC", "GGA_PBE")) {
            input.xc_type = XC_TYPE::GGA;
            input.xc_model = XC_MODEL::GGA_PBE;
        } else {
            input.xc_type = XC_TYPE::LDA;
            input.xc_model = XC_MODEL::LDA_CA;
        }



        //SetArray(input.box_axis, 20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0);


        const int num_elements = jobparam.GetBlockNumLines("System.AtomicElement");
        for (int i = 0; i < num_elements; ++i) {
            strcpy(line, jobparam.GetBlockString("System.AtomicElement", i));

            char* p = strtok(line, DELIMITER);
            const int Z = msz::GetAtomicNumber(p);

            p = strtok(nullptr, DELIMITER);
            input.pseudo_pot_set.emplace(std::make_pair(Z, p));

            p = strtok(nullptr, DELIMITER);
            if (p) {
                input.atomic_wave_set.emplace(std::make_pair(Z, p));
            } else {
                input.atomic_wave_set.emplace(std::make_pair(Z, ""));
            }
        }

        //condition of electrons and nuclei///////////////
        if (jobparam.IsEqualString("System.NumElectrons", "auto")) {
            //中性からの差分指定//
            input.num_electrons = -1;
        } else {
            input.num_electrons = jobparam.GetInt("System.NumElectrons", 1);
        }

        //マクロ(halfve+)が使われているときは0になるので後から自動判定//
        input.num_solutions = jobparam.GetInt("System.NumSolutions", 0);//alias


        //粒子とboxの読み込み//
        SetSupercell(jobparam, input);


        //condition of system size///////////////		
        input.HR_ratio = jobparam.GetInt("System.HighResolution.Ratio", 2);
        if (jobparam.IsEqualString("System.SpaceGrid", "auto")) {

            if (jobparam.Find("System.SpaceGrid.CutoffEnergyRy")) {
                const double cutE = jobparam.GetDouble("System.SpaceGrid.CutoffEnergyRy");
                int divisor[3]{ 1,1,1 };
                jobparam.GetIntArray("System.SpaceGrid.Divisor", divisor, 3);

                GridSizeFromCutoffRy(cutE, input.box_axis[0], input.box_axis[4], input.box_axis[8], divisor, input.grid_size);
            }
        } else {
            jobparam.GetIntArray("System.SpaceGrid", input.grid_size, 3);
        }

        //プレ計算の設定の読み込み//
        input.scf_pre_step = jobparam.GetInt("SCF.Pre.Step", 0);  //これを設定すると、小さいグリッドで事前計算をして初期状態を作る..
        input.scf_pre_eigen_step = jobparam.GetInt("SCF.Pre.EigenSolver.Step", input.eigen_step_per_scf);//事前計算でのLOBPCGステップ数:設定しない場合は本計算と同じ//
        input.scf_pre_eigen_step_initial = jobparam.GetInt("SCF.Pre.EigenSolver.InitialStep", input.eigen_step_initial);//事前計算でのLOBPCGステップ数:設定しない場合は本計算と同じ//
        if (jobparam.IsEqualString("SCF.Pre.SpaceGrid", "auto")) {
            if (jobparam.Find("SCF.Pre.SpaceGrid.CutoffEnergyRy")) {
                const double cutE = jobparam.GetDouble("SCF.Pre.SpaceGrid.CutoffEnergyRy");
                int divisor[3]{ 1,1,1 };
                jobparam.GetIntArray("System.SpaceGrid.Divisor", divisor, 3);

                GridSizeFromCutoffRy(cutE, input.box_axis[0], input.box_axis[4], input.box_axis[8], divisor, input.scf_pre_grid_size);
            }
        } else {
            jobparam.GetIntArray("SCF.Pre.SpaceGrid", input.scf_pre_grid_size, 3);
        }
        


        if (jobparam.Find("Initial.Density")) {
            input.initial_density = jobparam.GetString("Initial.Density");
        } else {
            input.initial_density = "none";
        }
        if (jobparam.Find("Initial.DensityDifference")) {
            input.initial_density_difference = jobparam.GetString("Initial.DensityDifference");
        } else {
            input.initial_density_difference = "none";
        }

        input.spin_polarization = jobparam.GetInt("System.SpinPolarization.Mode", 0);
        input.initial_spin_difference = jobparam.GetInt("Initial.SpinDifference", 0);
        {
            const int num = jobparam.GetBlockNumLines("Initial.AtomicSpinDifference");
            for (int i = 0; i < num; ++i) {
                input.initial_atomic_spin_difference.push_back(std::strtod(jobparam.GetBlockString("Initial.AtomicSpinDifference", i), nullptr));
            }
        }

        {
            auto res = jobparam.GetIntArray("System.KpointSample", input.kpoint_sample, 3);
            if (!res) {
                input.kpoint_sample[0] = 1; input.kpoint_sample[1] = 1; input.kpoint_sample[2] = 1;
            };

            if (jobparam.IsEqualString("System.KpointSymmetry", "FULL_XYZ")) {
                input.kpoint_symmetry = QUMASUN::KPOINT_SYMMETRY::FULL_XYZ;
            } else if (jobparam.IsEqualString("System.KpointSymmetry", "auto")) {
                input.kpoint_symmetry = QUMASUN::KPOINT_SYMMETRY::AUTO;
            } else {
                input.kpoint_symmetry = QUMASUN::KPOINT_SYMMETRY::NONE;
            }
        }


        const int num_init_state = jobparam.GetBlockNumLines("Initial.State");
        for (int i = 0; i < num_init_state; ++i) {
            input.initial_state.push_back(jobparam.GetBlockString("Initial.State", i));
        }
        
        if (jobparam.IsEqualString("Initial.State.Mode", "file")) {
            input.initial_state_mode = 1;
        }else if (jobparam.IsEqualString("Initial.State.Mode", "random")) {
            input.initial_state_mode = 0;
        }else if (jobparam.IsEqualString("Initial.State.Mode", "wave")) {
            input.initial_state_mode = 2;
        }


        const int num_init_occupancy = jobparam.GetBlockNumLines("Initial.Occupancy");
        for (int i = 0; i < num_init_occupancy; ++i) {
            strcpy(line, jobparam.GetBlockString("Initial.Occupancy", i));
            char* p = strtok(line, DELIMITER);
            OccupancyInfo occ;
            occ.num_state = strtol(p, nullptr, 10);
            p = strtok(nullptr, DELIMITER);
            occ.occupancy_up = strtod(p, nullptr);
            p = strtok(nullptr, DELIMITER);
            if (p) {
                occ.occupancy_down = strtod(p, nullptr);
            } else {
                occ.occupancy_down = occ.occupancy_up;
            }
            input.initial_occupancy.push_back(occ);

        }

        input.initial_expand_from_kpoint = jobparam.GetInt("Initial.ExpandFromKpoint", 0);

        input.temperature_K = jobparam.GetDouble("SCF.Temperature", 300.0);
        input.scf_mixing_ratio = jobparam.GetDouble("SCF.MixingRatio", 0.25);
        input.scf_mixing_kerker_factor = jobparam.GetDouble("SCF.MixingKerkerFactor", 2.0);
        input.scf_mixing_num_history = jobparam.GetInt("SCF.MixingNumHistory", 9);
        if (jobparam.IsEqualString("SCF.MixingMode", "Simple")) {
            input.scf_mixing_mode = MixingMode::Simple;
        } else if (jobparam.IsEqualString("SCF.MixingMode", "LBFGS")) {
            input.scf_mixing_mode = MixingMode::LBFGS;
        } else if (jobparam.IsEqualString("SCF.MixingMode", "SimpleKerker")) {
            input.scf_mixing_mode = MixingMode::SimpleKerker;
        } else if (jobparam.IsEqualString("SCF.MixingMode", "LBFGSKerker")) {
            input.scf_mixing_mode = MixingMode::LBFGSKerker;
        } else if (jobparam.IsEqualString("SCF.MixingMode", "DIIS")) {
            input.scf_mixing_mode = MixingMode::DIIS;
        } else {//default//
            input.scf_mixing_mode = MixingMode::DIIS;
        }
        input.scf_residual_convergence_threshold = jobparam.GetDouble("SCF.ResidualConvergence", 0.0);
        input.scf_threshold_density = jobparam.GetDouble("SCF.Threshold.Density", 0.0);
        input.scf_threshold_energy = jobparam.GetDouble("SCF.Threshold.Energy", 0.0);

        //Dynamics////////////////////////////////////////////////////////
        if (jobparam.IsEqualString("Dynamics.Mode", "Test")) {
            input.dynamics_mode = DynamicsMode::Test;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Relaxation")) {
            input.dynamics_mode = DynamicsMode::Relaxation2;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Relaxation1")) {
            input.dynamics_mode = DynamicsMode::Relaxation1;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Relaxation2")) {
            input.dynamics_mode = DynamicsMode::Relaxation2;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Relaxation3")) {
            input.dynamics_mode = DynamicsMode::Relaxation3;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Relaxation4")) {
            input.dynamics_mode = DynamicsMode::Relaxation4;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "Scaling1")) {
            input.dynamics_mode = DynamicsMode::Scaling1;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "TDDFT")) {
            input.dynamics_mode = DynamicsMode::TDDFT;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "EhrenfestMD")) {
            input.dynamics_mode = DynamicsMode::SymplecticEhrenfestMD;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "SymplecticEhrenfestMD")) {
            input.dynamics_mode = DynamicsMode::SymplecticEhrenfestMD;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "SemiSymplecticTest1")) {
            input.dynamics_mode = DynamicsMode::SemiSymplecticTest1;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "SemiSymplecticTest2")) {
            input.dynamics_mode = DynamicsMode::SemiSymplecticTest2;
        } else if (jobparam.IsEqualString("Dynamics.Mode", "SemiSymplecticTest3")) {
            input.dynamics_mode = DynamicsMode::SemiSymplecticTest3;
        }

        input.dynamics_step = jobparam.GetInt("Dynamics.Step", 1);
        input.dynamics_start_step = jobparam.GetInt("Dynamics.Start.Step", 0);
        input.dynamics_output_step = jobparam.GetInt("Dynamics.Output.Step", 100);
        if (input.dynamics_output_step < 1) {
            input.dynamics_output_step = 0;
        }

        {
            const int num_line = jobparam.GetBlockNumLines("Dynamics.Output.OrbitalDensity");
            for (int i = 0; i < num_line; i++) {
                strcpy(line, jobparam.GetBlockString("Dynamics.Output.OrbitalDensity", i));
                char* endptr = nullptr;
                int d = std::strtol(line, &endptr, 10);
                if (line != endptr) {
                    input.dynamics_output_orbital_density.insert(d);
                }
            }
            //note: dynamics_output_orbital_density is std::set, then inputted indexes are sorted.            
        }
        {
            const int num_line = jobparam.GetBlockNumLines("Dynamics.SavePoint");
            for (int i = 0; i < num_line; i++) {
                strcpy(line, jobparam.GetBlockString("Dynamics.SavePoint", i));
                char* endptr = nullptr;
                int d = std::strtol(line, &endptr, 10);
                if (line != endptr) {
                    input.save_point.push_back(d);
                }
            }
            //note: dynamics_output_orbital_density is std::set, then inputted indexes are sorted.            
            std::sort(std::begin(input.save_point), std::end(input.save_point));
        }
        input.dynamics_force_threshold = jobparam.GetDouble("Dynamics.Force.Threshold", 0.01);
        input.dynamics_timestep_dt = jobparam.GetDouble("Dynamics.TimeStep", 0.01);

        //mass and velocity///////////////////////////////////////////////////
        double mass_as_au = 1.0;
        if (jobparam.IsEqualString("Dynamics.Mass.Unit", "me") || jobparam.IsEqualString("Dynamics.Mass.Unit", "a.u.")) {
            mass_as_au = 1.0;
        } else if (jobparam.IsEqualString("Dynamics.Mass.Unit", "u") || jobparam.IsEqualString("Dynamics.Mass.Unit", "Da")) {
            mass_as_au = mass_u_as_au;
        }

        double velocity_as_au = 1.0;
        if (jobparam.IsEqualString("Dynamics.Velocity.Unit", "a.u.")) {
            velocity_as_au = 1.0;
        } else if (jobparam.IsEqualString("Dynamics.Velocity.Unit", "m/s")) {
            velocity_as_au = velocity_ms_as_au;
        }


        //粒子数の読み込み//
        const int num_line_velocity = jobparam.GetBlockNumLines("Dynamics.Velocity");
        for (int i = 0; i < num_line_velocity; i++) {
            strcpy(line, jobparam.GetBlockString("Dynamics.Velocity", i));

            char* p = strtok(line, DELIMITER);
            input.mass.emplace_back(strtod(p, nullptr) * mass_as_au);

            char* sx = strtok(nullptr, DELIMITER);
            char* sy = strtok(nullptr, DELIMITER);
            char* sz = strtok(nullptr, DELIMITER);

            //double unit_mass = ATOMIC_MASS[unit_element];

            //generator.AddIntoUnitcell(unit_element, unit_mass,
            input.velocity.emplace_back(strtod(sx, nullptr) * velocity_as_au);
            input.velocity.emplace_back(strtod(sy, nullptr) * velocity_as_au);
            input.velocity.emplace_back(strtod(sz, nullptr) * velocity_as_au);

        }

        if (const char* str = jobparam.GetString("Dynamics.Parameter.Scaling1")) {
            input.optional_parameters.insert(std::make_pair<std::string, std::string>("Scaling1", str));
        }

        double position_as_au = 1.0;
        if (jobparam.IsEqualString("Material.UnitLatticeVector.Unit", "Ang")) {
            position_as_au = Ang_as_Bohr;
        }

        //波動関数に初速を与えるルール//
        const int num_line_added_velocity = jobparam.GetBlockNumLines("Initial.State.Option");
        for (int i = 0; i < num_line_added_velocity; i++) {
            strcpy(line, jobparam.GetBlockString("Initial.State.Option", i));

            char* key = strtok(line, DELIMITER);
            //input.mass.emplace_back(strtod(p, nullptr) * mass_as_au);
            char* opecode = strtok(nullptr, DELIMITER);
            if (std::strncmp(opecode, "AddV", 4) != 0) continue;

            char* sx = strtok(nullptr, DELIMITER);
            char* sy = strtok(nullptr, DELIMITER);
            char* sz = strtok(nullptr, DELIMITER);

            //generator.AddIntoUnitcell(unit_element, unit_mass,
            AddedVelocityForWave add_to_wave;
            add_to_wave.velocity_x = (strtod(sx, nullptr) * velocity_as_au);
            add_to_wave.velocity_y = (strtod(sy, nullptr) * velocity_as_au);
            add_to_wave.velocity_z = (strtod(sz, nullptr) * velocity_as_au);

            sx = strtok(nullptr, DELIMITER);
            sy = strtok(nullptr, DELIMITER);
            sz = strtok(nullptr, DELIMITER);

            //generator.AddIntoUnitcell(unit_element, unit_mass,
            add_to_wave.center_x = (strtod(sx, nullptr) * position_as_au);
            add_to_wave.center_y = (strtod(sy, nullptr) * position_as_au);
            add_to_wave.center_z = (strtod(sz, nullptr) * position_as_au);
            input.velocity_for_wave.emplace(key, add_to_wave);
        }


        if (jobparam.Find("Output.StateVector")) {
            input.output_state_vector = jobparam.GetString("Output.StateVector");
        } else {
            input.output_state_vector = "none";
        }
        if (jobparam.Find("Output.OrbitalDensity")) {
            input.output_orbital_density = jobparam.GetString("Output.OrbitalDensity");
        } else {
            input.output_orbital_density = "none";
        }


        jobparam.GetDoubleArray("Test.ShiftGrid", input.test_shift_grid, 3);
        if(jobparam.IsEqualString("Test.Algorithm.Laplacian", "FFT2") ){
            input.test_algorithm_Laplacian = 2;
        }


        ////////////////////////////////////////////////////////
        //以下は簡略化のための自動設定機能
        ///////////////////////////////////////////////////////

        //電子数の自動判定//
        if (input.num_electrons == -1) {
            int system_total_charge = jobparam.GetInt("System.TotalCharge", 0);
            EstimateNumElectrons(input, system_total_charge, mpi_comm);
        }

        //num_solutionsの差分指定//
        {
            std::string str( jobparam.GetString("System.NumSolutions"));
            auto p = str.find_first_not_of( DELIMITER);
            if (str.compare(p, 9, "(halfve+)") == 0) {
                input.num_solutions = std::stol(str.substr(p + 9), nullptr, 10);
                input.num_solutions += (input.num_electrons + 1) / 2;
            }
        }

        //初期stateのディレクトリ指定時にファイル名へdecode//
        //ディレクトリ指定だった場合はnum_solutionも書き換わる//
        DecodeInitialStateDir(input, dir_path);

        //k点サンプルの対称性のauto判定//
        if (input.kpoint_symmetry == QUMASUN::KPOINT_SYMMETRY::AUTO) {
            SymmetryChecker checker;
            uint32_t res = checker.Check(input.nuclei.data(), input.num_nuclei, input.box_axis);
            input.kpoint_symmetry = res;
            //結果表示//
            if (IsRoot(mpi_comm)) {
                printf("Automatic Symmetry Check:\n");
                if (input.kpoint_symmetry == QUMASUN::KPOINT_SYMMETRY::NONE) {
                    printf("  no-symmetric\n");
                } else {
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::MIRROR_XY) {
                        printf("  mirror_x-y\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::MIRROR_YZ) {
                        printf("  mirror_y-z\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::MIRROR_ZX) {
                        printf("  mirror_z-x\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_X) {
                        printf("  inversion_x\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_Y) {
                        printf("  inversion_y\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_Z) {
                        printf("  inversion_z\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_BOTH_XY) {
                        printf("  inversion_both_x_y\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_BOTH_YZ) {
                        printf("  inversion_both_y_z\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_BOTH_ZX) {
                        printf("  inversion_both_z_x\n");
                    }
                    if (input.kpoint_symmetry & QUMASUN::KPOINT_SYMMETRY::NEGAPOSI_ALL_XYZ) {
                        printf("  inversion_all_x_y_z\n");
                    }

                }
            }
        }

		return 0;
	}
}

