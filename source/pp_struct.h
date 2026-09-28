#pragma once
#include <cstdint>
#include <vector>

class PseudoPot_MBK {
public:
    int valence_electron = 0;
    int num_radial_projectors = 0;
    int num_radial_grids = 0;
    double cutoff_vlocal = 0.0;
    //cutoff length in radial coordinate for l=0,1,2,3(angler q number)//
    double cutoff_r[4]{ 0.0 ,0.0 ,0.0 ,0.0 };

    //std::vector<int> projector_quantum_l;
    int* projector_quantum_l = nullptr;
    double* projector_energy_up = nullptr;
    double* projector_energy_down = nullptr;

    int64_t buffer_size = 0;
    double* buffer_all = nullptr;
    double* radius = nullptr;
    double* V_local = nullptr;
    std::vector<double*> projector;

    double* pcc_charge = nullptr;
    int has_pcc_charge = 0;
    double cutoff_pcc = 0.0;

    double* nucl_charge = nullptr;
    int has_nucl_charge = 0;

    double* ve_density = nullptr;
    int has_ve_density = 0;

    double xi_min;   //required to interpolate values on xi grid//
    double xi_delta; //required to interpolate values on xi grid//

public:
    /*
    * Release all memory, and this is not called in destructor
    */
    void Release() {
        delete[] buffer_all;
        delete[] projector_quantum_l;
        delete[] projector_energy_up;
    }

    int TotalProjectorLM() const {
        int num = 0;
        for (int k = 0; k < num_radial_projectors; ++k) {
            const int l = projector_quantum_l[k];
            num += 2 * l + 1;
        }
        return num;
    }

    int MaxProjectorL() const {
        int max_l = 0;
        for (int k = 0; k < num_radial_projectors; ++k) {
            const int l = projector_quantum_l[k];
            if (max_l < l)max_l = l;
        }
        return max_l;
    }

    double MaxCutoffLength() const {
        double max_cutoff = cutoff_vlocal;
        for (int l = 0; l < 4; ++l) {
            if (max_cutoff < cutoff_r[l]) max_cutoff = cutoff_r[l];
        }
        return max_cutoff;
    }

    template<class T>
    void mSwap(T& a, T& b) {
        T c = a;
        b = a;
        a = c;
    }

    void SortByL() {
        for (int k = 1; k < num_radial_projectors; ++k) {
            for (int j = 2; j < num_radial_projectors - j; ++j) {
                const int l0 = projector_quantum_l[j - 1];
                const int l = projector_quantum_l[j];
                if (l < l0) {
                    mSwap(projector_quantum_l[j - 1], projector_quantum_l[j]);
                    mSwap(projector_energy_up[j - 1], projector_energy_up[j]);
                    mSwap(projector_energy_down[j - 1], projector_energy_down[j]);
                    mSwap(projector[(j - 1) * 2], projector[j * 2]);
                    mSwap(projector[(j - 1) * 2 + 1], projector[j * 2 + 1]);
                }
            }
        }
    }
};
