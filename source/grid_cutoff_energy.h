#pragma once
/********************************
* 
* 空間グリッド幅といわゆるカットオフエネルギー(Ry)の変換に関する関数群
* 
* カットオフエネルギーとは、シミュレーションボックスの一片に対して
* 取り得る最も高い波数の波の持つエネルギーに相当する.
* 本来自然では無限の波数まで存在するが、
* シミュレーションでは数元のグリッド数Nになるため、
* 取り得る最も高い波数はN/2となる.
* 割る2となる理由は、空間グリッドNに対して、
* フーリエ変換した時の逆空間のグリッド範囲は(-N/2, N/2]となることに由来する.
* 
* すると, boxの幅をWとしたとき、逆空間でN/2とな波の波数は
* k = 2\pi *(N/2)/W = \pi / dx
* となる. ただしdxはグリッド幅である.
* このときのエネルギーは
* k^2 / 2 [Hartree]
* となる.
* 伝統的にDFTではRyを使って
* k^2 [Ry]
* とすることも多い。これがカットオフエネルギーである.
*********************************/
#include <cmath>

namespace QUMASUN {
    //Estimate cutoff energy [Hartree] from grid width [Bohr]//
    inline double CutoffEnergyFromGridWidth(double dx) {
        const double k = M_PI / dx;
        return k * k / 2.0;
    }
    
    //Estimate grid width [Bohr] from cutoff energy [Hartree]//
    inline double GridWidthFromCutoffEnergy(double cutoff_energy) {
        return M_PI / sqrt(2.0*cutoff_energy);
    }
}
