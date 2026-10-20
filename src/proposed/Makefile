CXX      = mpic++
CXXFLAGS = -std=c++20 -O2
TARGET   = experiment_mpi
SRCS     = experiment_mpi.cpp lib_output.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) result_np*.txt break.txt
