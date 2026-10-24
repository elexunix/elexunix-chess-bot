#pragma once
#define gpuErrchk(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "error file " << file << " line " << line << ": " << cudaGetErrorString(code) << std::endl;
    exit(code);
  }
}

//static const int cost_pawn = 10, cost_knight = 30, cost_bishop = 30, cost_rook = 50, cost_queen = 90;
constexpr int cost_pawn = 100, cost_knight = 300, cost_bishop = 300, cost_rook = 500, cost_queen = 900;
constexpr int material0_of_player = 8 * cost_pawn + 2 * cost_knight + 2 * cost_bishop + 2 * cost_rook + cost_queen;

constexpr uint64_t not_left_column = 0xfefefefefefefefe, not_right_column = 0x7f7f7f7f7f7f7f7f, not_top_row = 0x00ffffffffffffff, not_bottom_row = 0xffffffffffffff00, center_mask = 0x00003c3c3c3c0000,
                   white_cells = 0xaaaaaaaaaaaaaaaa, black_cells = 0x5555555555555555, row7 = 0x00ff000000000000, row2 = 0x000000000000ff00,
                   files[8] = {0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808, 0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080};

#ifdef __CUDA_ARCH__
#define __builtin_popcountll(x) __popcll(x)
#define __builtin_ctzll(x) (__ffsll(x) - 1)
#endif
