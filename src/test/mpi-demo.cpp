#include <mpi.h>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // rankごとに異なるシードで独立に探索(ここではダミープレイアウト)
    std::mt19937 rng(12345 + rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const int playouts = 100000;
    double local_sum = 0.0;
    for (int i = 0; i < playouts; ++i) local_sum += dist(rng);

    // 全プロセスの合計を集約
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        double value = global_sum / (static_cast<double>(playouts) * size);
        const double threshold = 0.5;
        std::cout << "value = " << value
                  << ", >= threshold: " << (value >= threshold ? "yes" : "no") << "\n";
    }
    MPI_Finalize();
}