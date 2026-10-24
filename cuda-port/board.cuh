#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>
//#include <inplace_vector>
#include <cuda/std/tuple>
#include "curand.h"
#include "curand_kernel.h"
#include "constants.cuh"
#ifndef INCLUDE_BOARD
#define INCLUDE_BOARD


template<typename T, int capacity>
struct const_sized_vec {
  T data[capacity];
  int size_ = 0;
  __host__ __device__ void push_back(const T& value) {
    assert(size_ < capacity);
    data[size_++] = value;
  }
  /*template<typename... Args>
  T& emplace_back(Args&&... args) {
    if (size < capacity) {
      ::new(&data[size]) T(std::forward<Args>(args)...);  // std::forward to preserve lvalue/rvalueness
      return data[size++];
    }
  }*/
  template<typename... Args>
  __host__ __device__ T& emplace_back(Args&&... args) {
    assert(size_ < capacity);
    return data[size_++] = T(args...);
  }
  __host__ __device__ void pop_back() {
    --size_;
  }
  __host__ __device__ const T& operator[](int index) const {
    return data[index];
  }
  __host__ __device__ T& operator[](int index) {
    return data[index];
  }
  __host__ __device__ constexpr int max_size() const noexcept {
    return capacity;
  }
  __host__ __device__ int size() const {
    return size_;
  }
  __host__ __device__ bool empty() const {
    return size_ == 0;
  }
  __host__ __device__ void clear() {
    size_ = 0;
  }
  __host__ __device__ const T* begin() const {
    return &data[0];
  }
  __host__ __device__ T* begin() {
    return &data[0];
  }
  __host__ __device__ const T* end() const {
    return &data[size_];
  }
  __host__ __device__ T* end() {
    return &data[size_];
  }
};

template<bool use_new>
struct nnue_state_t;

constexpr auto max(auto x, auto y) { return x > y ? x : y; }
constexpr auto min(auto x, auto y) { return x < y ? x : y; }
constexpr auto abs(auto x) { return x >= 0 ? x : -x; }
constexpr auto clamp(auto x, auto a, auto b) { return x < a ? a : x > b ? b : x; }


struct move {
  int8_t i0 = -1, j0 = -1, i1 = -1, j1 = -1;
  //int16_t i0 : 4, j0 : 4, i1 : 4, j1 : 4;  // despite being more compact, works ~30% slower
  __host__ __device__ move() {}
  __host__ __device__ move(int8_t i0, int8_t j0, int8_t i1, int8_t j1) : i0(i0), j0(j0), i1(i1), j1(j1) {}
  __host__ __device__ move(int from, int to) : i0(from >> 3), j0(from & 7), i1(to >> 3), j1(to & 7) {}
  friend std::ostream& operator<<(std::ostream& out, const move m) { return out << char('a' + m.j0) << m.i0 + 1 << char('a' + m.j1) << m.i1 + 1;; }
  __host__ __device__ bool operator==(const move another) const { return i0 == another.i0 && j0 == another.j0 && i1 == another.i1 && j1 == another.j1;; }
  __host__ __device__ bool operator!=(const move another) const { return i0 != another.i0 || j0 != another.j0 || i1 != another.i1 || j1 != another.j1;; }
  __host__ __device__ bool operator<(const move& another) const { return cuda::std::tie(i0, j0, i1, j1) < cuda::std::tie(another.i0, another.j0, another.i1, another.j1);; }
  __host__ __device__ bool operator>(const move& another) const { return cuda::std::tie(i0, j0, i1, j1) > cuda::std::tie(another.i0, another.j0, another.i1, another.j1);; }
  __host__ __device__ bool operator<=(const move& another) const { return cuda::std::tie(i0, j0, i1, j1) <= cuda::std::tie(another.i0, another.j0, another.i1, another.j1);; }
  __host__ __device__ bool operator>=(const move& another) const { return cuda::std::tie(i0, j0, i1, j1) >= cuda::std::tie(another.i0, another.j0, another.i1, another.j1);; }
  __host__ __device__ move mb_inverted(bool inverted) const { return inverted ? move(i0 ^ 7, j0, i1 ^ 7, j1) : *this;; }
};
static_assert(sizeof(move) == 4);
//static_assert(std::is_invocable_v<std::greater<move>, move, move>);
//static_assert(std::is_invocable_v<std::ranges::greater, move, move>);


constexpr int enough = 100;  // ub for the number of possible moves, mb 218
using moves_vec_t = const_sized_vec<move, enough>;
constexpr int max_game_length = 256;
using hashes_vec_t = const_sized_vec<uint64_t,max_game_length>;
using string = const_sized_vec<char,100>;

enum color { WHITE, BLACK };

uint64_t host_zobrist_bases[14][64], host_s2m_flip_hash[1];
uint64_t host_knight_masks[64], host_king_masks[64], host_pawn_attacks[2][64];
int8_t host_rook_rays[64][4][8], host_bishop_rays[64][4][8];  // for each cell and direction, up to 8 cells on this ray, -1 marks the end
uint64_t host_king_nhoods[64], host_king_nhoods2[64], host_rook_perforation[64], host_bishop_perforation[64], host_passed_pawn_check[2][64];  // for heuristic_eval
bool precalced = false;
__constant__ uint64_t zobrist_bases[14][64], s2m_flip_hash[1];
__constant__ uint64_t knight_masks[64], king_masks[64], pawn_attacks[2][64];
__constant__ int8_t rook_rays[64][4][8], bishop_rays[64][4][8];  // for each cell and direction, up to 8 cells on this ray, -1 marks the end
__constant__ uint64_t king_nhoods[64], king_nhoods2[64], rook_perforation[64], bishop_perforation[64], passed_pawn_check[2][64];  // for heuristic_eval

class board {
  static std::string white_piece_begin(int i, int j) { return (i + j) % 2 ? "\e[1;43;97m" : "\e[1;44;97m";; }
  static std::string black_piece_begin(int i, int j) { return (i + j) % 2 ? "\033[1;43;30m" : "\033[1;44;30m";; }
  static std::string piece_end(int i, int j) { return "\033[0m";; }
public:
  uint64_t white_pawns, white_knights, white_bishops, white_rooks, white_queens;
  uint64_t black_pawns, black_knights, black_bishops, black_rooks, black_queens;
  unsigned int white_king_i : 3, white_king_j : 3, black_king_i : 3, black_king_j : 3;
  unsigned int rook00_moved : 1, rook07_moved : 1, king04_moved : 1, rook70_moved : 1, rook77_moved : 1, king74_moved : 1;
  int8_t en_passant_sq = 0;  // 0 means nothing!
  color side2move = {WHITE};
  uint64_t zobrist_hash = 0;

  static void precalc(int seed) {
    std::clog << "in precalc" << std::endl;
    std::mt19937_64 gen(seed);
    host_s2m_flip_hash[0] = gen();
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        uint64_t mask = 0;
        if (i - 2 >= 0 && j - 1 >= 0)
          mask |= 1ull << 8 * (i - 2) + (j - 1);
        if (i - 2 >= 0 && j + 1 < 8)
          mask |= 1ull << 8 * (i - 2) + (j + 1);
        if (i - 1 >= 0 && j - 2 >= 0)
          mask |= 1ull << 8 * (i - 1) + (j - 2);
        if (i - 1 >= 0 && j + 2 < 8)
          mask |= 1ull << 8 * (i - 1) + (j + 2);
        if (i + 1 < 8 && j - 2 >= 0)
          mask |= 1ull << 8 * (i + 1) + (j - 2);
        if (i + 1 < 8 && j + 2 < 8)
          mask |= 1ull << 8 * (i + 1) + (j + 2);
        if (i + 2 < 8 && j - 1 >= 0)
          mask |= 1ull << 8 * (i + 2) + (j - 1);
        if (i + 2 < 8 && j + 1 < 8)
          mask |= 1ull << 8 * (i + 2) + (j + 1);
        host_knight_masks[8 * i + j] = mask;
        mask = 0;
        if (i > 0 && j > 0)
          mask |= 1ull << 8 * (i - 1) + (j - 1);
        if (i > 0)
          mask |= 1ull << 8 * (i - 1) + j;
        if (i > 0 && j < 7)
          mask |= 1ull << 8 * (i - 1) + (j + 1);
        if (j > 0)
          mask |= 1ull << 8 * i + (j - 1);
        if (j < 7)
          mask |= 1ull << 8 * i + (j + 1);
        if (i < 7 && j > 0)
          mask |= 1ull << 8 * (i + 1) + (j - 1);
        if (i < 7)
          mask |= 1ull << 8 * (i + 1) + j;
        if (i < 7 && j < 7)
          mask |= 1ull << 8 * (i + 1) + (j + 1);
        host_king_masks[8 * i + j] = mask;
        std::fill(&host_rook_rays[8 * i + j][0][0], &host_rook_rays[8 * i + j][0][0] + 4 * 8, -1);
        for (int j1 = j + 1; j1 < 8; ++j1)
          host_rook_rays[8 * i + j][0][j1 - j - 1] = 8 * i + j1;
        for (int j1 = j - 1; j1 >= 0; --j1)
          host_rook_rays[8 * i + j][1][j - j1 - 1] = 8 * i + j1;
        for (int i1 = i + 1; i1 < 8; ++i1)
          host_rook_rays[8 * i + j][2][i1 - i - 1] = 8 * i1 + j;
        for (int i1 = i - 1; i1 >= 0; --i1)
          host_rook_rays[8 * i + j][3][i - i1 - 1] = 8 * i1 + j;
        std::fill(&host_bishop_rays[8 * i + j][0][0], &host_bishop_rays[8 * i + j][0][0] + 4 * 8, -1);
        for (int i1 = i + 1, j1 = j + 1; i1 < 8 && j1 < 8; ++i1, ++j1)
          host_bishop_rays[8 * i + j][0][i1 - i - 1] = 8 * i1 + j1;
        for (int i1 = i + 1, j1 = j - 1; i1 < 8 && j1 >= 0; ++i1, --j1)
          host_bishop_rays[8 * i + j][1][i1 - i - 1] = 8 * i1 + j1;
        for (int i1 = i - 1, j1 = j + 1; i1 >= 0 && j1 < 8; --i1, ++j1)
          host_bishop_rays[8 * i + j][2][i - i1 - 1] = 8 * i1 + j1;
        for (int i1 = i - 1, j1 = j - 1; i1 >= 0 && j1 >= 0; --i1, --j1)
          host_bishop_rays[8 * i + j][3][i - i1 - 1] = 8 * i1 + j1;
      }
    }
    for (int i = 0; i < 14; ++i)
      for (int j = 0; j < 64; ++j)
        host_zobrist_bases[i][j] = gen();
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        int cell = 8 * i + j;
        host_king_nhoods[cell] = 0;
        for (int i1 = std::max(0, i - 1); i1 <= i + 1 && i1 < 8; ++i1)
          for (int j1 = std::max(0, j - 1); j1 <= j + 1 && j1 < 8; ++j1)
            host_king_nhoods[cell] |= 1ull << 8 * i1 + j1;
        host_king_nhoods2[cell] = 0;
        for (int i1 = std::max(0, i - 2); i1 <= i + 2 && i1 < 8; ++i1)
          for (int j1 = std::max(0, j - 2); j1 <= j + 2 && j1 < 8; ++j1)
            host_king_nhoods2[cell] |= 1ull << 8 * i1 + j1;
        host_rook_perforation[cell] = 0;
        host_bishop_perforation[cell] = 0;
        for (int i1 = 0; i1 < 8; ++i1) {
          for (int j1 = 0; j1 < 8; ++j1) {
            host_rook_perforation[cell] |= uint64_t(i1 == i ^ j1 == j) << 8 * i1 + j1;
            host_bishop_perforation[cell] |= uint64_t(i1 != i && (i1 - j1 == i - j || i1 + j1 == i + j)) << 8 * i1 + j1;
          }
        }
        // a passed pawn is one for which there's no enemy pawn in front on this/adj files
        host_passed_pawn_check[0][cell] = 0;
        for (int i1 = i + 1; i1 < 8; ++i1)
          for (int j1 = std::max(0, j - 1); j1 < 8 && j1 <= j + 1; ++j1)
            host_passed_pawn_check[0][cell] |= 1ull << 8 * i1 + j1;
        host_passed_pawn_check[1][cell] = 0;
        for (int i1 = i - 1; i1 >= 0; --i1)
          for (int j1 = std::max(0, j - 1); j1 < 8 && j1 <= j + 1; ++j1)
            host_passed_pawn_check[1][cell] |= 1ull << 8 * i1 + j1;
      }
    }
    for (int sq = 0; sq < 64; ++sq) {
      uint64_t pos = 1ull << sq;
      host_pawn_attacks[WHITE][sq] |= pos<<7 & not_right_column | pos<<9 & not_left_column;
      host_pawn_attacks[BLACK][sq] |= pos>>7 & not_left_column | pos>>9 & not_right_column;
    }
    std::clog << "board::precalc copying to symbol" << std::endl;
    gpuErrchk(cudaMemcpyToSymbol(zobrist_bases, host_zobrist_bases, sizeof(zobrist_bases)));
    gpuErrchk(cudaMemcpyToSymbol(s2m_flip_hash, host_s2m_flip_hash, sizeof(s2m_flip_hash)));
    gpuErrchk(cudaMemcpyToSymbol(knight_masks, host_knight_masks, sizeof(knight_masks)));
    gpuErrchk(cudaMemcpyToSymbol(king_masks, host_king_masks, sizeof(king_masks)));
    gpuErrchk(cudaMemcpyToSymbol(pawn_attacks, host_pawn_attacks, sizeof(pawn_attacks)));
    gpuErrchk(cudaMemcpyToSymbol(rook_rays, host_rook_rays, sizeof(rook_rays)));
    gpuErrchk(cudaMemcpyToSymbol(bishop_rays, host_bishop_rays, sizeof(bishop_rays)));
    gpuErrchk(cudaMemcpyToSymbol(king_nhoods, host_king_nhoods, sizeof(king_nhoods)));
    gpuErrchk(cudaMemcpyToSymbol(king_nhoods2, host_king_nhoods2, sizeof(king_nhoods2)));
    gpuErrchk(cudaMemcpyToSymbol(rook_perforation, host_rook_perforation, sizeof(rook_perforation)));
    gpuErrchk(cudaMemcpyToSymbol(bishop_perforation, host_bishop_perforation, sizeof(bishop_perforation)));
    gpuErrchk(cudaMemcpyToSymbol(passed_pawn_check, host_passed_pawn_check, sizeof(passed_pawn_check)));
    precalced = true;
    std::clog << "board::precalc done\n";
  }

  __device__ void recalc_hash() {
    zobrist_hash = 0;
    uint64_t boards[] = {white_pawns, white_knights, white_bishops, white_rooks, white_queens, 0, black_pawns, black_knights, black_bishops, black_rooks, black_queens};
    for (int type = 0; type <= 10; ++type) {
      if (type == 5) continue;
      for (uint64_t pieces = boards[type]; pieces; pieces &= pieces - 1) {
        zobrist_hash ^= zobrist_bases[type][__builtin_ctzll(pieces)];
      }
    }
    zobrist_hash ^= zobrist_bases[5][8 * white_king_i + white_king_j];
    zobrist_hash ^= zobrist_bases[11][8 * black_king_i + black_king_j];
    zobrist_hash ^= zobrist_bases[12][rook00_moved + 2 * rook07_moved + 4 * king04_moved + 8 * rook70_moved + 16 * rook77_moved + 32 * king74_moved];
    zobrist_hash ^= zobrist_bases[13][en_passant_sq];
  }

private:
  __host__ __device__ uint64_t get_all_pieces() const {
    return white_pawns | white_knights | white_bishops | white_rooks | white_queens | 1ull << 8 * white_king_i + white_king_j |
           black_pawns | black_knights | black_bishops | black_rooks | black_queens | 1ull << 8 * black_king_i + black_king_j;
  }

  __host__ __device__ uint64_t get_rook_attacks(int sq, uint64_t occupancy) const {
    assert(0 <= sq && sq < 64);
    int r = sq >> 3, f = sq & 7;
    uint64_t attacks = 0;
    // N, S, E, W
    for (int dir : {8, -8, 1, -1}) {
      for (int i = 1; i < 8; ++i) {
        int target_sq = sq + dir * i;
        if (target_sq < 0 || target_sq >= 64)
          break;

        int target_r = target_sq >> 3, target_f = target_sq & 7;
        if (dir == 1 || dir == -1) { if (target_r != r) break; }
        if (dir == 8 || dir == -8) { if (target_f != f) break; }

        attacks |= 1ull << target_sq;
        if (occupancy & 1ull << target_sq)
          break;
      }
    }
    return attacks;
  }

  __device__ uint64_t get_bishop_attacks(int sq, uint64_t occupancy) const {
    assert(0 <= sq && sq < 64);
    int r = sq >> 3, f = sq & 7;
    uint64_t attacks = 0;
    // NE, NW, SE, SW
    for (int dir : {9, 7, -7, -9}) {
      for (int i = 1; i < 8; ++i) {
        int target_sq = sq + dir * i;
        if (target_sq < 0 || target_sq >= 64)
          break;

        int dist = max(abs(r - (target_sq>>3)), abs(f - (target_sq&7)));
        if (dist != i)
          break;

        attacks |= 1ull << target_sq;
        if (occupancy & 1ull << target_sq)
          break;
      }
    }
    return attacks;
  }

  __device__ bool is_attacked(int sq, color attacker_color) const {
    const uint64_t his_pawns   = attacker_color == WHITE ? white_pawns   : black_pawns;
    const uint64_t his_knights = attacker_color == WHITE ? white_knights : black_knights;
    const uint64_t his_bishops = attacker_color == WHITE ? white_bishops : black_bishops;
    const uint64_t his_rooks   = attacker_color == WHITE ? white_rooks   : black_rooks;
    const uint64_t his_queens  = attacker_color == WHITE ? white_queens  : black_queens;
    const int his_king_sq      = attacker_color == WHITE ? white_king_i<<3|white_king_j : black_king_i<<3|black_king_j;

    if (pawn_attacks[attacker_color == WHITE ? BLACK : WHITE][sq] & his_pawns)
      return true;
    if (knight_masks[sq] & his_knights)
      return true;
    if (king_masks[sq] & 1ull<<his_king_sq)
      return true;

    uint64_t occupancy = get_all_pieces();
    if (get_bishop_attacks(sq, occupancy) & (his_bishops | his_queens))
      return true;
    if (get_rook_attacks(sq, occupancy) & (his_rooks | his_queens))
      return true;

    return false;
  }
public:
  __host__ __device__ board() {}
  explicit __device__ board(int) : white_pawns(0xff << 8), white_knights(0x42), white_bishops(0x24), white_rooks(0x81), white_queens(0x08), white_king_i(0), white_king_j(4),
                        black_pawns(0xffull << 48), black_knights(0x42ull << 56), black_bishops(0x24ull << 56), black_rooks(0x81ull << 56), black_queens(0x08ull << 56), black_king_i(7), black_king_j(4),
                        rook00_moved(false), rook07_moved(false), king04_moved(false), rook70_moved(false), rook77_moved(false), king74_moved(false) {
    //assert(precalced);
    recalc_hash();
  }

  __host__ __device__ board(const board&) {}

  /*board(std::string fen) {
    assert(precalced);
    std::replace(fen.begin(), fen.end(), '_', ' ');
    white_pawns = white_knights = white_bishops = white_rooks = white_queens = black_pawns = black_knights = black_bishops = black_rooks = black_queens = 0;
    int i = 7, j = 0;
    for (int pos = 0; pos < fen.length(); ++pos) {
      char c = fen[pos];
      if (c == '/') {
        assert(i > 0);
        assert(j == 8);
        --i, j = 0;
      } else {
        uint64_t cell = 1ull << 8 * i + j;
        if ('1' <= c && c <= '9') {
          while (c > '0') {
            ++j, --c;
          }
        } else if (c == 'P') {
          white_pawns |= cell, ++j;
        } else if (c == 'N') {
          white_knights |= cell, ++j;
        } else if (c == 'B') {
          white_bishops |= cell, ++j;
        } else if (c == 'R') {
          white_rooks |= cell, ++j;
        } else if (c == 'Q') {
          white_queens |= cell, ++j;
        } else if (c == 'K') {
          white_king_i = i, white_king_j = j++;
        } else if (c == 'p') {
          black_pawns |= cell, ++j;
        } else if (c == 'n') {
          black_knights |= cell, ++j;
        } else if (c == 'b') {
          black_bishops |= cell, ++j;
        } else if (c == 'r') {
          black_rooks |= cell, ++j;
        } else if (c == 'q') {
          black_queens |= cell, ++j;
        } else if (c == 'k') {
          black_king_i = i, black_king_j = j++;
        } else if (c == ' ') {
          fen = fen.substr(pos + 1);
          break;
        }
      }
    }
    side2move = fen[0] == 'w' ? WHITE : BLACK;
    fen = fen.substr(2);
    rook00_moved = rook07_moved = king04_moved = rook70_moved = rook77_moved = king74_moved = true;
    for (char c : fen) {
      if (c == ' ')
        break;
      switch(c) {
        case 'K': king04_moved = rook07_moved = false; break;
        case 'Q': king04_moved = rook00_moved = false; break;
        case 'k': king74_moved = rook77_moved = false; break;
        case 'q': king74_moved = rook70_moved = false; break;
        case '-': break;
        default:  // hack per 960: Se riceve le lettere delle colonne, non crashare; azzera semplicemente i flag standard sperando in bene.
          if (c >= 'A' && c <= 'H')
            king04_moved = rook00_moved = rook07_moved = false;
          else if (c >= 'a' && c <= 'h')
            king74_moved = rook70_moved = rook77_moved = false;
          else
            throw std::invalid_argument("unable to parse castling rights from fen: " + fen);
          break;
        //default: throw std::invalid_argument("unable to parse castling rights from fen: " + fen);
      }
    }
    recalc_hash();
    if (side2move == BLACK)
      zobrist_hash ^= host_s2m_flip_hash;
  }*/

  __device__ void toggle_identity() {
    side2move = (color)!side2move;
    zobrist_hash ^= s2m_flip_hash[0];
  }

  template<bool>
  __host__ __device__ int pst_summands(int game_phase) const;
  template<bool>
  __device__ int one_sided_score(int game_phase) const;
  template<bool use_new>
  __device__ int hc_heuristic_score(const nnue_state_t<use_new>&) const;

  enum class desire_type_t { fill_array_of_moves, vector_of_moves, bool_has_any, generator_of_moves, attack_mask, perforation_mask, cnt_possible_moves };

  template<desire_type_t desire_type>
  struct list_possible_moves_ret_helper {
    using enum desire_type_t;
    using ret_type = cuda::std::conditional_t<desire_type == vector_of_moves, moves_vec_t, cuda::std::conditional_t<desire_type == bool_has_any, bool,
                          cuda::std::conditional_t<desire_type == generator_of_moves, bool, cuda::std::conditional_t<desire_type == cnt_possible_moves, int, uint64_t>>>>;
  };

  template<desire_type_t desire_type, bool only_captures = false, typename ret_t = list_possible_moves_ret_helper<desire_type>::ret_type>
  __device__ ret_t list_possible_moves() const {  // device bc of popcount
    moves_vec_t captures, noncaptures;
    uint64_t attack_mask = 0, perforation_mask = 0;
    int cnt_possible_moves = 0;
    constexpr bool need_mask = cuda::std::is_same_v<ret_t, uint64_t>;
    auto push_move = [&captures, &noncaptures, &attack_mask, &cnt_possible_moves](int from, int to, bool capturing) {
      if constexpr (desire_type == desire_type_t::cnt_possible_moves) {
        ++cnt_possible_moves;
        return;
      }
      if constexpr (desire_type == desire_type_t::attack_mask) {
        attack_mask |= 1ull << to;
      } else if constexpr (!need_mask) {
        if (capturing) {
          captures.emplace_back(from, to);
        } else if constexpr (!only_captures) {
          noncaptures.emplace_back(from, to);
        }
      }
    };

    const uint64_t my_pawns = side2move == WHITE ? white_pawns : black_pawns;
    const uint64_t my_knights = side2move == WHITE ? white_knights : black_knights;
    const uint64_t my_bishops = side2move == WHITE ? white_bishops : black_bishops;
    const uint64_t my_rooks = side2move == WHITE ? white_rooks : black_rooks;
    const uint64_t my_queens = side2move == WHITE ? white_queens : black_queens;
    const int my_king_i = side2move == WHITE ? white_king_i : black_king_i, my_king_j = side2move == WHITE ? white_king_j : black_king_j;

    const uint64_t my_something = my_pawns | my_knights | my_bishops | my_rooks | my_queens | 1ull << (my_king_i<<3|my_king_j);
    const uint64_t his_something = side2move == WHITE ? black_pawns | black_knights | black_bishops | black_rooks | black_queens | 1ull << (black_king_i<<3|black_king_j) :
                                                          white_pawns | white_knights | white_bishops | white_rooks | white_queens | 1ull << (white_king_i<<3|white_king_j);

    uint64_t occupied_squares = my_something | his_something, empty_squares = ~occupied_squares;

    if (side2move == WHITE) {
      uint64_t single_push = my_pawns<<8 & empty_squares;
      uint64_t double_push = (single_push&0xff0000ull)<<8 & empty_squares;

      for (; single_push; single_push &= single_push - 1) {
        int to = __builtin_ctzll(single_push);
        push_move(to - 8, to, false);
      }
      for (; double_push; double_push &= double_push - 1) {
        int to = __builtin_ctzll(double_push);
        push_move(to - 16, to, false);
      }
      for (uint64_t right_captures = (my_pawns & not_right_column) << 9 & his_something; right_captures; right_captures &= right_captures - 1) {
        int to = __builtin_ctzll(right_captures);
        push_move(to - 9, to, true);
      }
      for (uint64_t left_captures = (my_pawns & not_left_column) << 7 & his_something; left_captures; left_captures &= left_captures - 1) {
        int to = __builtin_ctzll(left_captures);
        push_move(to - 7, to, true);
      }

      if (en_passant_sq && my_pawns & (1ull<<en_passant_sq-7 | 1ull<<en_passant_sq-9)) {
        for (uint64_t from_mask = (1ull<<en_passant_sq>>7 & not_left_column | 1ull<<en_passant_sq>>9 & not_right_column) & my_pawns; from_mask; from_mask &= from_mask - 1) {
          push_move(__builtin_ctzll(from_mask), en_passant_sq, true);
        }
      }
    } else { // BLACK
      uint64_t single_push = my_pawns>>8 & empty_squares;
      uint64_t double_push = (single_push&0xff0000000000ull)>>8 & empty_squares;

      for (; single_push; single_push &= single_push - 1) {
        int to = __builtin_ctzll(single_push);
        push_move(to + 8, to, false);
      }
      for (; double_push; double_push &= double_push - 1) {
        int to = __builtin_ctzll(double_push);
        push_move(to + 16, to, false);
      }
      for (uint64_t right_captures = (my_pawns & not_right_column) >> 7 & his_something; right_captures; right_captures &= right_captures - 1) {
        int to = __builtin_ctzll(right_captures);
        push_move(to + 7, to, true);
      }
      for (uint64_t left_captures = (my_pawns & not_left_column) >> 9 & his_something; left_captures; left_captures &= left_captures - 1) {
        int to = __builtin_ctzll(left_captures);
        push_move(to + 9, to, true);
      }

      if (en_passant_sq && (my_pawns & (1ull << en_passant_sq + 7 | 1ull << en_passant_sq + 9))) {
        for (uint64_t from_mask = (1ull << en_passant_sq << 7 & not_right_column | 1ull << en_passant_sq << 9 & not_left_column) & my_pawns; from_mask; from_mask &= from_mask - 1) {
          push_move(__builtin_ctzll(from_mask), en_passant_sq, true);
        }
      }
    }

    for (uint64_t knights = my_knights; knights; knights &= knights - 1) {
      int from = __builtin_ctzll(knights);
      for (uint64_t targets = knight_masks[from] & ~my_something; targets; targets &= targets - 1) {
        int to = __builtin_ctzll(targets);
        push_move(from, to, 1ull<<to & his_something);
      }
    }

    for (uint64_t rooklike = my_rooks | my_queens; rooklike; rooklike &= rooklike - 1) {
      int from = __builtin_ctzll(rooklike);
      #pragma unroll(4)
      for (int ray_i = 0; ray_i < 4; ++ray_i) {
        for (int8_t *it = rook_rays[from][ray_i];; ++it) {
          int to = *it;
          if (to == -1)
            break;
          uint64_t to_mask = 1ull << to;
          if (to_mask & my_something)
            break;
          push_move(from, to, to_mask & his_something);
          if (to_mask & his_something)
            break;
        }
      }
      if constexpr (desire_type == desire_type_t::perforation_mask) {
        perforation_mask |= rook_perforation[from];
      }
    }

    for (uint64_t bishoplike = my_bishops | my_queens; bishoplike; bishoplike &= bishoplike - 1) {
      int from = __builtin_ctzll(bishoplike);
      #pragma unroll(4)
      for (int ray_i = 0; ray_i < 4; ++ray_i) {
        for (int8_t *it = bishop_rays[from][ray_i];; ++it) {
          int to = *it;
          if (to == -1)
            break;
          uint64_t to_mask = 1ull << to;
          if (to_mask & my_something)
            break;
          push_move(from, to, to_mask & his_something);
          if (to_mask & his_something)
            break;
        }
      }
      if constexpr (desire_type == desire_type_t::perforation_mask) {
        perforation_mask |= bishop_perforation[from];
      }
    }

    for (uint64_t king_mask = king_masks[my_king_i<<3|my_king_j] & ~my_something; king_mask; king_mask &= king_mask - 1) {
      unsigned int to = __builtin_ctzll(king_mask);
      push_move(my_king_i<<3|my_king_j, to, 1ull<<to & his_something);
    }

    if (!need_mask) {
      uint64_t under_enemy_attack = 0;
      bool need_attack_mask = (side2move == WHITE && !king04_moved && my_king_i == 0 && my_king_j == 4 && (!rook00_moved || !rook07_moved)) ||
                              (side2move == BLACK && !king74_moved && my_king_i == 7 && my_king_j == 4 && (!rook70_moved || !rook77_moved));
      if (need_attack_mask) {
        board b2(*this);
        b2.toggle_identity();
        under_enemy_attack = b2.list_possible_moves<desire_type_t::attack_mask>();
      }

      if (side2move == WHITE && my_king_i == 0 && my_king_j == 4 && !king04_moved) {
        if (!rook00_moved && !(occupied_squares & 0x0e) && !(under_enemy_attack & 0x1c)) {
          push_move(4, 2, false);
        }
        if (!rook07_moved && !(occupied_squares & 0x60) && !(under_enemy_attack & 0x70)) {
          push_move(4, 6, false);
        }
      } else if (side2move == BLACK && my_king_i == 7 && my_king_j == 4 && !king74_moved) {
        if (!rook70_moved && !(occupied_squares & 0x0eull<<56) && !(under_enemy_attack & 0x1cull<<56)) {
          push_move(60, 58, false);
        }
        if (!rook77_moved && !(occupied_squares & 0x60ull<<56) && !(under_enemy_attack & 0x70ull<<56)) {
          push_move(60, 62, false);
        }
      }
    }

    if constexpr (need_mask) {
      return (desire_type == desire_type_t::perforation_mask) ? perforation_mask : attack_mask;
    } else if constexpr (desire_type == desire_type_t::cnt_possible_moves) {
      return cnt_possible_moves;
    } else {
      for (move m : noncaptures) {
        assert(m.i1 >= 0 && m.i1 < 8 && m.j1 >= 0 && m.j1 < 8);
        assert(m.i0 >= 0 && m.i0 < 8 && m.j0 >= 0 && m.j0 < 8);
      }
      for (move m : captures) {
        assert(m.i1 >= 0 && m.i1 < 8 && m.j1 >= 0 && m.j1 < 8);
        assert(m.i0 >= 0 && m.i0 < 8 && m.j0 >= 0 && m.j0 < 8);
      }
      if constexpr (!only_captures) {
        for (move m : noncaptures)
          captures.push_back(m);
      }
      if constexpr(desire_type == desire_type_t::bool_has_any) {
        return !captures.empty();
      } else {
        return captures;
      }
    }
  }

  __device__ bool has_legal_moves() const {
    for (move m : list_possible_moves<desire_type_t::vector_of_moves>()) {
      board next = *this;
      (bool)next.make_move(m);
      if (!next.can_eat_king()) {
        return true;
      }
    }
    return false;
  }
  __device__ move get_unique_legal_move() const {  // or null if none
    move um;  // null
    for (move m : list_possible_moves<desire_type_t::vector_of_moves>()) {
      board next = *this;
      (bool)next.make_move(m);
      if (!next.can_eat_king()) {
        if (um == move()) {
          um = m;
        } else {
          return move();
        }
      }
    }
    return um;
  }
  __device__ bool is_stalemate() const {
    return !is_check() && !has_legal_moves();
  }

  __device__ bool can_eat_king() const {  // we can eat his king
    const int his_king_sq = side2move == WHITE ? black_king_i<<3|black_king_j : white_king_i<<3|white_king_j;
    return his_king_sq >= 0 ? is_attacked(his_king_sq, side2move) : true;  // if the king doesn't exist, assume yes
  }

  __device__ bool is_check() const {  // we are under check
    board tmp(*this);
    tmp.toggle_identity();
    return tmp.can_eat_king();
  }

  __host__ __device__ bool has_sufficient_material_for_nmp() const {
    // checks for whether there's anything except for king&pawns on my side
    return side2move == WHITE ? white_bishops | white_rooks | white_queens : black_bishops | black_rooks | black_queens;
  }

  template<bool use_new>
  __device__ bool can_give_check() const {
    for (move m : list_possible_moves<desire_type_t::vector_of_moves>()) {
      board tmp(*this);
      if (tmp.make_move<false>(m)) {
        return true;
      }
      if (tmp.can_eat_king()) {
        return true;
      }
    }
    return false;
  }

  template<int vigilance = 0>
  [[nodiscard]] __host__ __device__ bool is_capture(move m) const {
    assert(m != move(-1, -1, -1, -1));
    int to = 8 * m.i1 + m.j1;
    uint64_t mask_from = 1ull << (m.i0<<3|m.j0);

    const uint64_t his_something = side2move == WHITE ? black_pawns | black_knights | black_bishops | black_rooks | black_queens | 1ull << (black_king_i<<3|black_king_j) :
                                                          white_pawns | white_knights | white_bishops | white_rooks | white_queens | 1ull << (white_king_i<<3|white_king_j);

    if (1ull<<to & his_something)
      return true;

    const auto& my_pawns = (side2move == WHITE) ? white_pawns : black_pawns;
    if (mask_from & my_pawns) {
      if (vigilance >= 1 && (side2move == WHITE && m.i1 == 7 || side2move == BLACK && m.i1 == 0))
        return true;
      if (m.j0 != m.j1) // en passant
        return true;
      if (vigilance >= 2)
        return true;
    }
    return false;
  }

  __device__ void make_null_move() {
    toggle_identity();
  }

  template<bool toggle_colors = true>
  [[nodiscard]] __device__ bool make_move(move m) {
    assert(m != move(-1, -1, -1, -1));
    int from = 8 * m.i0 + m.j0, to = 8 * m.i1 + m.j1;
    uint64_t mask_from = 1ull << from, mask_to = 1ull << to;

    uint64_t &my_pawns    = side2move == WHITE ? white_pawns   : black_pawns,
             &my_knights  = side2move == WHITE ? white_knights : black_knights,
             &my_bishops  = side2move == WHITE ? white_bishops : black_bishops,
             &my_rooks    = side2move == WHITE ? white_rooks   : black_rooks,
             &my_queens   = side2move == WHITE ? white_queens  : black_queens,
             &his_pawns   = side2move == WHITE ? black_pawns   : white_pawns,
             &his_knights = side2move == WHITE ? black_knights : white_knights,
             &his_bishops = side2move == WHITE ? black_bishops : white_bishops,
             &his_rooks   = side2move == WHITE ? black_rooks   : white_rooks,
             &his_queens  = side2move == WHITE ? black_queens  : white_queens;

    const int color_offset = side2move == WHITE ? 0 : 6;
    if (mask_to & his_pawns)
      his_pawns &= ~mask_to, zobrist_hash ^= zobrist_bases[6 - color_offset + 0][to];
    else if (mask_to & his_knights)
      his_knights &= ~mask_to, zobrist_hash ^= zobrist_bases[6 - color_offset + 1][to];
    else if (mask_to & his_bishops)
      his_bishops &= ~mask_to, zobrist_hash ^= zobrist_bases[6 - color_offset + 2][to];
    else if (mask_to & his_rooks)
      his_rooks &= ~mask_to, zobrist_hash ^= zobrist_bases[6 - color_offset + 3][to];
    else if (mask_to & his_queens)
      his_queens &= ~mask_to, zobrist_hash ^= zobrist_bases[6 - color_offset + 4][to];

    if (side2move == WHITE ? m.i0 == white_king_i && m.j0 == white_king_j : m.i0 == black_king_i && m.j0 == black_king_j) {  // is king's move
      if (abs(m.j0 - m.j1) == 2)
        if (m.j1 == 2)
          my_rooks ^= 1ull << (m.i0<<3|0) | 1ull << (m.i0<<3|3),
          zobrist_hash ^= zobrist_bases[color_offset + 3][8 * m.i0 + 0] ^ zobrist_bases[color_offset + 3][8 * m.i0 + 3];
        else
          my_rooks ^= 1ull << (m.i0<<3|7) | 1ull << (m.i0<<3|5),
          zobrist_hash ^= zobrist_bases[color_offset + 3][8 * m.i0 + 7] ^ zobrist_bases[color_offset + 3][8 * m.i0 + 5];

      zobrist_hash ^= zobrist_bases[color_offset + 5][from] ^ zobrist_bases[color_offset + 5][to];

      if (side2move == WHITE)
        white_king_i = m.i1, white_king_j = m.j1;
      else
        black_king_i = m.i1, black_king_j = m.j1;
    } else if (mask_from & my_pawns) {
      int8_t promotion_rank = side2move == WHITE ? 7 : 0;
      if (m.i1 == promotion_rank)
        my_pawns ^= mask_from, my_queens ^= mask_to,
        zobrist_hash ^= zobrist_bases[color_offset + 0][from] ^ zobrist_bases[color_offset + 4][to];
      else if (to == en_passant_sq && m.j0 != m.j1)
        my_pawns ^= mask_from ^ mask_to,
        his_pawns ^= 1ull << (side2move == WHITE ? to - 8 : to + 8),
        zobrist_hash ^= zobrist_bases[color_offset + 0][from] ^ zobrist_bases[color_offset + 0][to] ^ zobrist_bases[6 - color_offset + 0][side2move == WHITE ? to - 8 : to + 8];
      else
        my_pawns ^= mask_from ^ mask_to,
        zobrist_hash ^= zobrist_bases[color_offset + 0][from] ^ zobrist_bases[color_offset + 0][to];
    } else if (mask_from & my_knights)
      my_knights ^= mask_from ^ mask_to, zobrist_hash ^= zobrist_bases[color_offset + 1][from] ^ zobrist_bases[color_offset + 1][to];
    else if (mask_from & my_bishops)
      my_bishops ^= mask_from ^ mask_to, zobrist_hash ^= zobrist_bases[color_offset + 2][from] ^ zobrist_bases[color_offset + 2][to];
    else if (mask_from & my_rooks)
      my_rooks ^= mask_from ^ mask_to, zobrist_hash ^= zobrist_bases[color_offset + 3][from] ^ zobrist_bases[color_offset + 3][to];
    else if (mask_from & my_queens)
      my_queens ^= mask_from ^ mask_to, zobrist_hash ^= zobrist_bases[color_offset + 4][from] ^ zobrist_bases[color_offset + 4][to];

    zobrist_hash ^= zobrist_bases[12][rook00_moved + 2 * rook07_moved + 4 * king04_moved + 8 * rook70_moved + 16 * rook77_moved + 32 * king74_moved];
    if (side2move == WHITE)
      king04_moved |= from == 4;
    else
      king74_moved |= from == 60;
    if (from == 0 || to == 0)
      rook00_moved = true;
    if (from == 7 || to == 7)
      rook07_moved = true;
    if (from == 56 || to == 56)
      rook70_moved = true;
    if (from == 63 || to == 63)
      rook77_moved = true;
    zobrist_hash ^= zobrist_bases[12][rook00_moved + 2 * rook07_moved + 4 * king04_moved + 8 * rook70_moved + 16 * rook77_moved + 32 * king74_moved];

    zobrist_hash ^= zobrist_bases[13][en_passant_sq];
    en_passant_sq = 0;
    if (mask_to & my_pawns && abs(m.i0 - m.i1) == 2)
      en_passant_sq = (from + to) / 2;
    zobrist_hash ^= zobrist_bases[13][en_passant_sq];

    if (to == (side2move == WHITE ? black_king_i<<3|black_king_j : white_king_i<<3|white_king_j))
      return true;
    if constexpr (toggle_colors)
      toggle_identity();
    return false;
  }

  template<bool toggle_colors>
  __device__ auto make_move(int i0, int j0, int i1, int j1) { return make_move<toggle_colors>({(int8_t)i0, (int8_t)j0, (int8_t)i1, (int8_t)j1}); }

  /*[[deprecated]] __device__ bool make_random_move(board& b, curandState_t &curand_state) {
    const moves_vec_t possible_moves(b.list_possible_moves<desire_type_t::vector_of_moves>());
    return b.make_move(possible_moves[curand(&curand_state) % possible_moves.size()]);
  }*/

  __host__ __device__ uint64_t get_hash() const { return zobrist_hash; }
  __host__ __device__ bool operator==(const board& another) const {
    if (zobrist_hash != another.zobrist_hash)
      return false;
    return cuda::std::tie(white_pawns, white_knights, white_bishops, white_rooks, white_queens,
                    black_pawns, black_knights, black_bishops, black_rooks, black_queens,
                    white_king_i, white_king_j, black_king_i, black_king_j,
                    rook00_moved, rook07_moved, king04_moved, rook70_moved, rook77_moved, king74_moved,
                    en_passant_sq, side2move) ==
           cuda::std::tie(another.white_pawns, another.white_knights, another.white_bishops, another.white_rooks, another.white_queens,
                    another.black_pawns, another.black_knights, another.black_bishops, another.black_rooks, another.black_queens,
                    another.white_king_i, another.white_king_j, another.black_king_i, another.black_king_j,
                    another.rook00_moved, another.rook07_moved, another.king04_moved, another.rook70_moved, another.rook77_moved, another.king74_moved,
                    another.en_passant_sq, another.side2move);
  }
  __host__ __device__ bool operator!=(const board& another) const { return !(*this == another); }

  /*void read_move(move& m, bool toggle_identity = false) const {
    while (true) {
      std::cout << "Your move: ";
      std::string s;
      std::cin >> s;
      if (s == "undo")
        return void(m = {-2, -2, -2, -2});
      m.i0 = s[1] - '1', m.j0 = 'A' <= s[0] && s[0] <= 'Z' ? s[0] - 'A' : s[0] - 'a';
      m.i1 = s[3] - '1', m.j1 = 'A' <= s[2] && s[2] <= 'Z' ? s[2] - 'A' : s[2] - 'a';
      if (toggle_identity)
        m.i0 ^= 7, m.i1 ^= 7;
      for (move pm : list_possible_moves<desire_type_t::vector_of_moves>())
        if (m == pm)
          return;
      std::cout << "Invalid move" << std::endl;
    }
  }*/

  __host__ __device__ string get_fen() const {
    uint64_t white_kings = 1ull << (white_king_i<<3|white_king_j), black_kings = 1ull << (black_king_i<<3|black_king_j);
    string fen;
    for (int i = 7; i >= 0; --i) {
      int last_blanks = 0;
      for (int j = 0; j < 8; ++j) {
        uint64_t cell = 1ull << 8 * i + j;
        char c = cell & white_pawns ? 'P' : cell & white_knights ? 'N' : cell & white_bishops ? 'B' : cell & white_rooks ? 'R' : cell & white_queens ? 'Q' : cell & white_kings ? 'K'
               : cell & black_pawns ? 'p' : cell & black_knights ? 'n' : cell & black_bishops ? 'b' : cell & black_rooks ? 'r' : cell & black_queens ? 'q' : cell & black_kings ? 'k' : ' ';
        if (c == ' ') {
          ++last_blanks;
        } else {
          if (last_blanks)
            fen.push_back('0' + last_blanks), last_blanks = 0;
          fen.push_back(c);
        }
      }
      if (last_blanks > 0)
        fen.push_back('0' + last_blanks);
      if (i)
        fen.push_back('/');
    }
    fen.push_back(side2move == WHITE ? 'w' : 'b');
    string castlings;
    if (!king04_moved && !rook07_moved)
      castlings.push_back('K');
    if (!king04_moved && !rook00_moved)
      castlings.push_back('Q');
    if (!king74_moved && !rook77_moved)
      castlings.push_back('k');
    if (!king74_moved && !rook70_moved)
      castlings.push_back('q');
    if (castlings.empty())
      castlings.clear(), castlings.push_back('-');
    for (char c : castlings)
      fen.push_back(c);
    fen.push_back(' ');
    if (!en_passant_sq)
      fen.push_back('-');
    else
      fen.push_back('a' + en_passant_sq%8), fen.push_back('1' + en_passant_sq/8);
    return fen;
  }

  std::string get_pgn(const move m, bool inverted = false) const {
    uint64_t from = 1ull << 8 * m.i0 + m.j0;
    std::string fig = (from & (white_pawns | black_pawns)) ? "" : (from & (white_knights | black_knights)) ? "N" : (from & (white_bishops | black_bishops)) ? "B" : (from & (white_rooks | black_rooks)) ? "R" : (from & (white_queens | black_queens)) ? "Q" : "K";
    if (fig == "K" && std::abs(m.j0 - m.j1) == 2)
      return m.j1 == 6 ? "O-O" : "O-O-O";
    int promotion_rank = (from & white_pawns) ? 7 : (from & black_pawns) ? 0 : -1;
    return fig + char('a' + m.j0) + char('1' + (inverted ? m.i0 ^ 7 : m.i0)) + "-" + char('a' + m.j1) + char('1' + (inverted ? m.i1 ^ 7 : m.i1)) + (promotion_rank != -1 && m.i1 == promotion_rank ? "=Q" : "");
  }

  void show() const {
    uint64_t white_kings = 1ull << (white_king_i<<3|white_king_j), black_kings = 1ull << (black_king_i<<3|black_king_j),
             white_something = white_pawns | white_knights | white_bishops | white_rooks | white_queens | white_kings,
             black_something = black_pawns | black_knights | black_bishops | black_rooks | black_queens | black_kings;
    std::cout << "en_passant_sq " << (int)en_passant_sq << std::endl;
    std::cout << "\t+";
    for (int j = 0; j < 8; ++j)
      std::cout << "-------+";
    std::cout << std::endl;
    for (int i = 7; i >= 0; --i) {
      std::cout << "\t|";
      for (int j = 0; j < 8; ++j)
        std::cout << white_piece_begin(i, j) << "       " << piece_end(i, j) << "|";
      std::cout << "\n";
      std::cout << "\t|";
      for (int j = 0; j < 8; ++j) {
        uint64_t this_cell = 1ull << 8 * i + j;
        std::cout << (white_something & this_cell ? white_piece_begin(i, j) : black_piece_begin(i, j)) << "   ";
        if ((white_pawns | black_pawns) & this_cell)
          std::cout << "P";
        else if ((white_knights | black_knights) & this_cell)
          std::cout << "N";
        else if ((white_bishops | black_bishops) & this_cell)
          std::cout << "B";
        else if ((white_rooks | black_rooks) & this_cell)
          std::cout << "R";
        else if ((white_queens | black_queens) & this_cell)
          std::cout << "Q";
        else if (i == white_king_i && j == white_king_j || i == black_king_i && j == black_king_j)
          std::cout << "K";
        else
          std::cout << " ";
        std::cout << "   " << piece_end(i, j) << "|";
      }
      std::cout << "\n";
      std::cout << "\t|";
      for (int j = 0; j < 8; ++j)
        std::cout << white_piece_begin(i, j) << "       " << piece_end(i, j) << "|";
      std::cout << "\n";
      std::cout << "\t+";
      for (int j = 0; j < 8; ++j)
        std::cout << "-------+";
      std::cout << "\n";
    }
    std::cout << std::endl;
    std::cout << "Zobrist hash: " << zobrist_hash << std::endl;
    std::string fen;
    for (char c : get_fen())
      fen += c;
    std::cout << "FEN: " << fen << std::endl;
    std::replace(fen.begin(), fen.end(), ' ', '_');
    std::cout << "Link: https://lichess.org/analysis/" + fen << std::endl;
  }
};
static_assert(sizeof(board) == 96);

/*namespace std {
  template<>
  struct hash<board> {
    static uint64_t operator()(const board& b) { return b.get_hash(); }
  };
}*/


#endif
