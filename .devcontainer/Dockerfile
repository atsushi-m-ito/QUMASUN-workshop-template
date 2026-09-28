FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04

RUN apt-get update && apt-get install -y \
    wget \
    gpg \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
       | gpg --dearmor \
       > /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
       > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y \
       intel-oneapi-compiler-dpcpp-cpp \
       intel-oneapi-mkl-devel \
       intel-oneapi-mpi-devel \
    && echo 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1' >> /etc/bash.bashrc