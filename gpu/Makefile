# Falcons CUDA decoder. nvcc 12.8 is required for native Blackwell (sm_120) code.
NVCC ?= /usr/local/cuda-12.8/bin/nvcc
CXX ?= g++
FALCONS_BUILD ?= build
INC = -I../include -I$(FALCONS_BUILD)/include -I.
NVFLAGS = -O3 -std=c++17 -arch=sm_120 --fmad=false -Xcompiler -fopenmp $(INC)
CXXFLAGS = -O3 -std=c++17 -mavx2 -mfma -fopenmp $(INC)
LIBS = -L/usr/local/cuda-12.8/lib64 -Wl,-rpath,/usr/local/cuda-12.8/lib64 -lcudart $(FALCONS_BUILD)/libFalcons_fse.a -lzstd -fopenmp

all: fgpu_decode fgpu_compress fgpu_difftest fgpu_bench fgpu_difftest_scalar

fgpu_recon.o: fgpu_recon.cu fgpu_compress.inl fgpu_encode.inl fgpu_fse_enc.inl fgpu_pure.inl fgpu_entropy.inl fgpu_recon.h fgpu_plan.hpp fgpu_shared.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

fgpu_difftest.o: fgpu_difftest.cpp fgpu_recon.h fgpu_plan.hpp fgpu_shared.h fgpu_stream.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

fgpu_difftest: fgpu_difftest.o fgpu_recon.o
	$(CXX) $^ -o $@ $(LIBS)

# Same test against a scalar Falcons build, whose -P streams are unfused throughout.
fgpu_difftest_scalar.o: fgpu_difftest.cpp fgpu_recon.h fgpu_plan.hpp fgpu_shared.h fgpu_stream.hpp
	$(CXX) -O3 -std=c++17 -fopenmp $(INC) -c $< -o $@

fgpu_difftest_scalar: fgpu_difftest_scalar.o fgpu_recon.o
	$(CXX) $^ -o $@ $(LIBS)

fgpu_compress.o: fgpu_compress.cpp fgpu_recon.h fgpu_plan.hpp fgpu_shared.h fgpu_stream.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

fgpu_compress: fgpu_compress.o fgpu_recon.o
	$(CXX) $^ -o $@ $(LIBS)

fgpu_decode.o: fgpu_decode.cpp fgpu_recon.h fgpu_plan.hpp fgpu_shared.h fgpu_stream.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

fgpu_decode: fgpu_decode.o fgpu_recon.o
	$(CXX) $^ -o $@ $(LIBS)

fgpu_bench.o: fgpu_bench.cpp fgpu_recon.h fgpu_plan.hpp fgpu_shared.h fgpu_stream.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

fgpu_bench: fgpu_bench.o fgpu_recon.o
	$(CXX) $^ -o $@ $(LIBS)

clean:
	rm -f *.o fgpu_decode fgpu_compress fgpu_difftest fgpu_difftest_scalar fgpu_bench
