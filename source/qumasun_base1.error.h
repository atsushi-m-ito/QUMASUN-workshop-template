#pragma once
#include "qumasun_base1.h"

namespace QUMASUN {
    namespace ErrorCode {
        constexpr uint32_t NO_ERROR = 0;
        constexpr uint32_t OCCUPANCY_NOT_AGREEMENT = 1;
    }
}

uint32_t QUMASUN_BASE1::CheckError() {
    uint32_t res;
    MPI_Allreduce(&m_error_flag, &res, 1, MPI_UINT32_T, MPI_MAX, m_mpi_comm);
    return res;
}
