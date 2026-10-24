#include <cassert>
#include <fstream>
#include <immintrin.h>
#ifndef INCLUDE_NN_EVAL
#define INCLUDE_NN_EVAL
#include "board.cuh"

const std::string filename_old = "model_v8_128_8_8.bin",
                  filename_new = "model_v8_128_8_8.bin";
                  //filename_new = "model9d9.bin";
float q_step_01, q_step_12, q_step_23, q_step_34;

constexpr int s0 = 768 + 5;

constexpr int s1_old = 128, s2_old = 8, s3_old = 8, s4_old = 1;
__managed__ float nn_w01_old[s0][s1_old], nn_w10_old[s1_old][s0], nn_b1_old[s1_old],
             nn_w12_old[s1_old][s2_old], nn_w21_old[s2_old][s1_old], nn_b2_old[s2_old],
             nn_w23_old[s2_old][s3_old], nn_w32_old[s3_old][s2_old], nn_b3_old[s3_old],
             nn_w34_old[s3_old][s4_old], nn_w43_old[s4_old][s3_old], nn_b4_old[s4_old];

constexpr int s1_new = 128, s2_new = 8, s3_new = 8, s4_new = 1;
__managed__ float nn_w01_new[s0][s1_new], nn_w10_new[s1_new][s0], nn_b1_new[s1_new],
             nn_w12_new[s1_new][s2_new], nn_w21_new[s2_new][s1_new], nn_b2_new[s2_new],
             nn_w23_new[s2_new][s3_new], nn_w32_new[s3_new][s2_new], nn_b3_new[s3_new],
             nn_w34_new[s3_new][s4_new], nn_w43_new[s4_new][s3_new], nn_b4_new[s4_new];
//float
// ^T @.1s 77079:75865 +3 Elo

template<typename T>
T relu(T x) { return max(T(0), x); }

using weight_t = int16_t;
using late_weight_t = int8_t;
using accumulator_t = int32_t;
alignas(32) static weight_t nn_w01_q[s0][s1_new];
alignas(32) static accumulator_t nn_b1_q[s1_new];
static late_weight_t nn_w21_new_q[s2_new][s1_new], nn_b2_new_q[s2_new], nn_w32_new_q[s3_new][s2_new], nn_b3_new_q[s3_new], nn_w43_new_q[s4_new][s3_new], nn_b4_new_q[s4_new];

template<bool use_new>
__host__ __device__ float nn_eval(board b);

constexpr bool dual_nnue = true;  // @.3s 184:125 +67 Elo

template<bool use_new>
struct nnue_state_t {  //   @3s 304.5:232.5 +47 Elo
  static constexpr int s1 = use_new ? s1_new : s1_old;
  const float (*nn_w01)[s1], *nn_b1;
  static constexpr bool quantize = false;//IMPORTANT: at most one should be on, since weight arrays are shared
  alignas(32) float a1_pre_relu[2][s1];
  alignas(32) accumulator_t a1_pre_relu_q[2][s1];

  __device__ nnue_state_t(const board& b) /*: nn_w01(use_new ? nn_w01_new : nn_w01_old), nn_b1(use_new ? nn_b1_new : nn_b1_old)*/ {  //, const float nn_w01[s0][s1], const float nn_b1[s1]) : nn_w01(nn_w01), nn_b1(nn_b1) {
    if constexpr (use_new)
      nn_w01 = nn_w01_new, nn_b1 = nn_b1_new;
    else
      nn_w01 = nn_w01_old, nn_b1 = nn_b1_old;
    uint64_t masks[12] = {b.white_pawns, b.white_knights, b.white_bishops, b.white_rooks, b.white_queens, 1ull<<(b.white_king_i<<3|b.white_king_j),
                          b.black_pawns, b.black_knights, b.black_bishops, b.black_rooks, b.black_queens, 1ull<<(b.black_king_i<<3|b.black_king_j)};
    for (int j = 0; j < s1; ++j)
      if constexpr (quantize)
        a1_pre_relu_q[0][j] = nn_b1_q[j];
      else
        a1_pre_relu[0][j] = nn_b1[j];
    if (dual_nnue)
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[1][j] = nn_b1_q[j];
        else
          a1_pre_relu[1][j] = nn_b1[j];
    for (int i = 0; i < 12; ++i)
      for (int j = 0; j < s1; ++j)
        for (uint64_t pieces = masks[i]; pieces; pieces &= pieces - 1) {
          int idx = (i<<6) + __builtin_ctzll(pieces);
          if constexpr (quantize)
            a1_pre_relu_q[0][j] += nn_w01_q[idx][j];
          else
            a1_pre_relu[0][j] += nn_w01[idx][j];
        }
    if (dual_nnue) {
      for (int i = 0; i < 12; ++i) {
        int p = (i < 6 ? i + 6 : i - 6) << 6;
        for (int j = 0; j < s1; ++j)
          for (uint64_t pieces = masks[i]; pieces; pieces &= pieces - 1) {
            int idx = p + (__builtin_ctzll(pieces)^56);
            if constexpr (quantize)
              a1_pre_relu_q[1][j] += nn_w01_q[idx][j];
            else
              a1_pre_relu[1][j] += nn_w01[idx][j];
          }
      }
    }

    if (!b.king04_moved && !b.rook07_moved) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[768][j];
        else
          a1_pre_relu[0][j] += nn_w01[768][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[770][j];
          else
            a1_pre_relu[1][j] += nn_w01[770][j];
    }
    if (!b.king04_moved && !b.rook00_moved) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[769][j];
        else
          a1_pre_relu[0][j] += nn_w01[769][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[771][j];
          else
            a1_pre_relu[1][j] += nn_w01[771][j];
    }
    if (!b.king74_moved && !b.rook77_moved) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[770][j];
        else
          a1_pre_relu[0][j] += nn_w01[770][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[768][j];
          else
            a1_pre_relu[1][j] += nn_w01[768][j];
    }
    if (!b.king74_moved && !b.rook70_moved) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[771][j];
        else
          a1_pre_relu[0][j] += nn_w01[771][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[769][j];
          else
            a1_pre_relu[1][j] += nn_w01[769][j];
    }

    float ep_val = 1;
    for (int j = 0; j < s1; ++j)
      if constexpr (quantize)
        a1_pre_relu_q[0][j] += nn_w01_q[772][j];
      else
        a1_pre_relu[0][j] += nn_w01[772][j] * ep_val;
    if constexpr (dual_nnue) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[1][j] += nn_w01_q[772][j];
        else
          a1_pre_relu[1][j] += nn_w01[772][j] * ep_val;
    }
  }

  __device__ void update(const board &b, const board &next_b) {
    //assert(0 <= m.i0 && m.i0 < 8 && 0 <= m.j0 && m.j0 < 8 && 0 <= m.i1 && m.i1 < 8 && 0 <= m.j1 && m.j1 < 8);
    uint64_t b_white_kings = 1ull << (b.white_king_i<<3|b.white_king_j), b_black_kings = 1ull << (b.black_king_i<<3|b.black_king_j),
             next_b_white_kings = 1ull << (next_b.white_king_i<<3|next_b.white_king_j), next_b_black_kings = 1ull << (next_b.black_king_i<<3|next_b.black_king_j);
    uint64_t masks_b[12] = {b.white_pawns, b.white_knights, b.white_bishops, b.white_rooks, b.white_queens, b_white_kings, b.black_pawns, b.black_knights, b.black_bishops, b.black_rooks, b.black_queens, b_black_kings},
             masks_next_b[12] = {next_b.white_pawns, next_b.white_knights, next_b.white_bishops, next_b.white_rooks, next_b.white_queens, next_b_white_kings,
                                 next_b.black_pawns, next_b.black_knights, next_b.black_bishops, next_b.black_rooks, next_b.black_queens, next_b_black_kings};
    for (int i = 0; i < 12; ++i) {
      int offset_w = i << 6, offset_b = (i < 6 ? i + 6 : i - 6) << 6;
      for (uint64_t mask = masks_b[i] & ~masks_next_b[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask);
        if constexpr (quantize) {
//          add_feats<s1,sub_mode>(a1_pre_relu_q[0], nn_w01_q[offset_w | sq]);
//          if (dual_nnue)
//            add_feats<s1,sub_mode>(a1_pre_relu_q[1], nn_w01_q[offset_b | sq ^ 56]);
        } else {
          for (int j = 0; j < s1; ++j) {
            a1_pre_relu[0][j] -= nn_w01[offset_w | sq][j];
            if (dual_nnue)
              a1_pre_relu[1][j] -= nn_w01[offset_b | sq ^ 56][j];
          }
        }
      }
      for (uint64_t mask = masks_next_b[i] & ~masks_b[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask);
        if constexpr (quantize) {
//          add_feats<s1,add_mode>(a1_pre_relu_q[0], nn_w01_q[offset_w | sq]);
//          if (dual_nnue)
//            add_feats<s1,add_mode>(a1_pre_relu_q[1], nn_w01_q[offset_b | sq ^ 56]);
        } else {
          for (int j = 0; j < s1; ++j) {
            a1_pre_relu[0][j] += nn_w01[offset_w | sq][j];
            if (dual_nnue)
              a1_pre_relu[1][j] += nn_w01[offset_b | sq ^ 56][j];
          }
        }
      }
    }
    int dc1 = (!next_b.king04_moved && !next_b.rook07_moved) - (!b.king04_moved && !b.rook07_moved), dc2 = (!next_b.king04_moved && !next_b.rook00_moved) - (!b.king04_moved && !b.rook00_moved),
        dc3 = (!next_b.king74_moved && !next_b.rook77_moved) - (!b.king74_moved && !b.rook77_moved), dc4 = (!next_b.king74_moved && !next_b.rook70_moved) - (!b.king74_moved && !b.rook70_moved),
        dep = (next_b.en_passant_sq != 0) - (b.en_passant_sq != 0);
    // this stuff is way less expensive, ignore for now
    if (dc1) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += dc1 * nn_w01_q[768][j];
        else
          a1_pre_relu[0][j] += dc1 * nn_w01[768][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += dc1 * nn_w01_q[770][j];
          else
            a1_pre_relu[1][j] += dc1 * nn_w01[770][j];
    }
    if (dc2) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += dc2 * nn_w01_q[769][j];
        else
          a1_pre_relu[0][j] += dc2 * nn_w01[769][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += dc2 * nn_w01_q[771][j];
          else
            a1_pre_relu[1][j] += dc2 * nn_w01[771][j];
    }
    if (dc3) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += dc3 * nn_w01_q[770][j];
        else
          a1_pre_relu[0][j] += dc3 * nn_w01[770][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += dc3 * nn_w01_q[768][j];
          else
            a1_pre_relu[1][j] += dc3 * nn_w01[768][j];
    }
    if (dc4) {
      for (int j = 0; j < s1; ++j)
        if constexpr (quantize)
          a1_pre_relu_q[0][j] += dc4 * nn_w01_q[771][j];
        else
          a1_pre_relu[0][j] += dc4 * nn_w01[771][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if constexpr (quantize)
            a1_pre_relu_q[1][j] += dc4 * nn_w01_q[769][j];
          else
            a1_pre_relu[1][j] += dc4 * nn_w01[769][j];
    }
  }
};

template<bool use_new, int s1=use_new?s1_new:s1_old, int s2=use_new?s2_new:s2_old, int s3=use_new?s3_new:s3_old, int s4=use_new?s4_new:s4_old>
__host__ __device__ float forward_pass(board b, const nnue_state_t<use_new>& nnue_state, const float (&w01)[s0][s1], const float (&b1)[s1], const float (&w12)[s1][s2], const float (&b2)[s2],
                                                                     const float (&w23)[s2][s3], const float (&b3)[s3], const float (&w34)[s3][s4], const float (&b4)[s4],
                                                                     const float (&w10)[s1][s0],                        const float (&w21)[s2][s1],
                                                                     const float (&w32)[s3][s2],                        const float (&w43)[s4][s3]) {
  bool flip = b.side2move == BLACK;
  float a1[s1], a2[s2], a3[s3];
  static constexpr bool quantize = nnue_state_t<use_new>::quantize;
  for (int j = 0; j < s1; ++j)
    a1[j] = max(0.f, nnue_state.a1_pre_relu[dual_nnue?flip:0][j]);

  for (int j = 0; j < s2; ++j)
    a2[j] = b2[j];
  for (int i = 0; i < s1; ++i)
    for (int j = 0; j < s2; ++j)
      a2[j] += w21[j][i] * a1[i];
  for (int j = 0; j < s2; ++j)
    a2[j] = max(0.f, a2[j]);
  for (int j = 0; j < s3; ++j)
    a3[j] = b3[j];
  for (int i = 0; i < s2; ++i)
    for (int j = 0; j < s3; ++j)
      a3[j] += w32[j][i] * a2[i];
  for (int j = 0; j < s3; ++j)
    a3[j] = max(0.f, a3[j]);
  float output = b4[0];
  for (int k = 0; k < s3; ++k)
    output += a3[k] * w34[k][0];

  output = (!dual_nnue && flip ? -1.f : +1.f) * clamp(output, -.999f, .999f);  // bugfix clamp here @.3s 2505:2291 mb +16 Elo, @3s 2547:2331 +15 Elo
  return 400.f * -log(2.f / (output + 1.f) - 1.f);
}

template<int s1, int s2, int s3, int s4>
void read_net(std::string filename, float (&nn_w01)[s0][s1], float (&nn_b1)[s1], float (&nn_w12)[s1][s2], float (&nn_b2)[s2], float (&nn_w23)[s2][s3], float (&nn_b3)[s3], float (&nn_w34)[s3][s4], float (&nn_b4)[s4],
                                    float (&nn_w10)[s1][s0], float (&nn_w21)[s2][s1], float (&nn_w32)[s3][s2], float (&nn_w43)[s4][s3]) {
  std::ifstream fin(filename, std::ios::binary);
  if (fin) {
    fin.read(reinterpret_cast<char*>(&nn_w01), s0 * s1 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_b1),  s1 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_w12), s1 * s2 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_b2),  s2 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_w23), s2 * s3 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_b3),  s3 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_w34), s3 * s4 * sizeof(float));
    fin.read(reinterpret_cast<char*>(&nn_b4),  s4 * sizeof(float));
    if (!fin) {
      std::cerr << "Error: file " << filename << " is smaller than expected" << std::endl;
      exit(1);
    }
    fin.peek();
    if (!fin.eof()) {
      std::cerr << "Error: file " << filename << " is larger than expected" << std::endl;
      exit(1);
    }
  } else {
    std::cerr << "Error: No se pudo abrir " << filename << std::endl;
    exit(1);
  }
  for (int i = 0; i < s0; ++i)
    for (int j = 0; j < s1; ++j)
      nn_w10[j][i] = nn_w01[i][j];
  for (int i = 0; i < s1; ++i)
    for (int j = 0; j < s2; ++j)
      nn_w21[j][i] = nn_w12[i][j];
  for (int i = 0; i < s2; ++i)
    for (int j = 0; j < s3; ++j)
      nn_w32[j][i] = nn_w23[i][j];
  for (int i = 0; i < s3; ++i)
    for (int j = 0; j < s4; ++j)
      nn_w43[j][i] = nn_w34[i][j];
}

void precalc_nn_eval() {
  read_net<s1_old,s2_old,s3_old,s4_old>(filename_old, nn_w01_old, nn_b1_old, nn_w12_old, nn_b2_old, nn_w23_old, nn_b3_old, nn_w34_old, nn_b4_old, nn_w10_old, nn_w21_old, nn_w32_old, nn_w43_old);
  read_net<s1_new,s2_new,s3_new,s4_new>(filename_new, nn_w01_new, nn_b1_new, nn_w12_new, nn_b2_new, nn_w23_new, nn_b3_new, nn_w34_new, nn_b4_new, nn_w10_new, nn_w21_new, nn_w32_new, nn_w43_new);
  // we don't quantize first layer, the one that is nnue'ed
  // first we quantize all weights in layers 2+ to int8 range
//  float max_w = 0;
//  for (int i = 0; i < s1_new; ++i)
//    for (int j = 0; j < s2_new; ++j)
//      max_w = max(max_w, abs(nn_w12_new[i][j]));
//  float scale_w12 = std::numeric_limits<late_weight_t>::max() / max_w;
//  for (int j = 0; j < s2_new; ++j) {
//    nn_b2_new_q[j] = nn_b2_new[j] * scale_w12;
//    for (int i = 0; i < s1_new; ++i)
//      nn_w21_new_q[j][i] = nn_w21_new[j][i] * scale_w12;
//  }
//  max_w = 0;
//  for (int i = 0; i < s2_new; ++i)
//    for (int j = 0; j < s3_new; ++j)
//      max_w = max(max_w, abs(nn_w23_new[i][j]));
//  float scale_w23 = std::numeric_limits<late_weight_t>::max() / max_w;
//  for (int j = 0; j < s3_new; ++j) {
//    nn_b3_new_q[j] = nn_b3_new[j] * scale_w23;
//    for (int i = 0; i < s2_new; ++i)
//      nn_w32_new_q[j][i] = nn_w32_new[j][i] * scale_w23;
//  }
//  max_w = 0;
//  for (int i = 0; i < s3_new; ++i)
//    for (int j = 0; j < s4_new; ++j)
//      max_w = max(max_w, abs(nn_w34_new[i][j]));
//  float scale_w34 = std::numeric_limits<late_weight_t>::max() / max_w;
//  for (int j = 0; j < s4_new; ++j) {
//    nn_b4_new_q[j] = nn_b4_new[j] * scale_w34;
//    for (int i = 0; i < s3_new; ++i)
//      nn_w43_new_q[j][i] = nn_w43_new[j][i] * scale_w34;
//  }
}

template<bool use_new>
__host__ __device__ float nn_eval(board b, const nnue_state_t<use_new>& nnue_state) {
  if constexpr (use_new) {
    return forward_pass<use_new>(b, nnue_state, nn_w01_new, nn_b1_new, nn_w12_new, nn_b2_new, nn_w23_new, nn_b3_new, nn_w34_new, nn_b4_new, nn_w10_new, nn_w21_new, nn_w32_new, nn_w43_new);
  } else {
    return forward_pass<use_new>(b, nnue_state, nn_w01_old, nn_b1_old, nn_w12_old, nn_b2_old, nn_w23_old, nn_b3_old, nn_w34_old, nn_b4_old, nn_w10_old, nn_w21_old, nn_w32_old, nn_w43_old);
  }
}

#endif
