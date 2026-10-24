#include <cassert>
#include <fstream>
#include <immintrin.h>
#ifndef INCLUDE_NN_EVAL
#define INCLUDE_NN_EVAL
#include "board.h"

const std::string filename_old = "model9f9.bin",//"model9h12.bin",
                  filename_new = "model9j09.bin";
                  //filename_new = "model9d9.bin";
float q_step_01, q_step_12, q_step_23, q_step_34;

constexpr int s0 = 768 + 5;

constexpr int s1_old = 128, s2_old = 8, s3_old = 8, s4_old = 1;
alignas(32) static float nn_w01_old[s0][s1_old], nn_w10_old[s1_old][s0], nn_b1_old[s1_old],
             nn_w12_old[s1_old][s2_old], nn_w21_old[s2_old][s1_old], nn_b2_old[s2_old],
             nn_w23_old[s2_old][s3_old], nn_w32_old[s3_old][s2_old], nn_b3_old[s3_old],
             nn_w34_old[s3_old][s4_old], nn_w43_old[s4_old][s3_old], nn_b4_old[s4_old];

constexpr int s1_new = 128, s2_new = 8, s3_new = 8, s4_new = 1;
alignas(32) static float nn_w01_new[s0][s1_new], nn_w10_new[s1_new][s0], nn_b1_new[s1_new],
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
float nn_eval(board b);

constexpr bool dual_nnue = true;  // @.3s 184:125 +67 Elo

enum addition_mode_t {add_mode, sub_mode};
template<int s1, addition_mode_t addition_mode>  // false to subtract
inline void add_feats(accumulator_t *acc, const weight_t *w) {
  static_assert(s1 % 8 == 0);
  static_assert(std::is_same_v<weight_t,int16_t>);
  static_assert(std::is_same_v<accumulator_t,int32_t>);
  // ❤
  for (int j = 0; j < s1; j += 8) {
    __m128i w_pack_of_int16_t = _mm_load_si128((const __m128i*)&w[j]);  // load unaligned 128 bits into special registers
    __m256i w_pack_of_int32_t = _mm256_cvtepi16_epi32(w_pack_of_int16_t),  // sign extension (cvt=convert, epi=extended packed int)
            a_pack_of_int32_t = _mm256_load_si256((const __m256i*)&acc[j]);
    a_pack_of_int32_t = (addition_mode==add_mode?_mm256_add_epi32:_mm256_sub_epi32)(a_pack_of_int32_t, w_pack_of_int32_t);
    /*if constexpr (addition_mode == add_mode)
      a_pack_of_int32_t = _mm256_add_epi32(a_pack_of_int32_t, w_pack_of_int32_t);
    else
      a_pack_of_int32_t = _mm256_sub_epi32(a_pack_of_int32_t, w_pack_of_int32_t);*/
    _mm256_store_si256((__m256i*)&acc[j], a_pack_of_int32_t);
  }
}

template<bool use_new>
struct nnue_state_t {  //   @3s 304.5:232.5 +47 Elo
  static constexpr int s1 = use_new ? s1_new : s1_old;
  const float (*nn_w01)[s1], *nn_b1;
  static constexpr bool quantize = false;//IMPORTANT: at most one should be on, since weight arrays are shared
  alignas(32) float a1_pre_relu[2][s1];
  alignas(32) accumulator_t a1_pre_relu_q[2][s1];

  nnue_state_t(const board& b) /*: nn_w01(use_new ? nn_w01_new : nn_w01_old), nn_b1(use_new ? nn_b1_new : nn_b1_old)*/ {  //, const float nn_w01[s0][s1], const float nn_b1[s1]) : nn_w01(nn_w01), nn_b1(nn_b1) {
    if constexpr (use_new)
      nn_w01 = nn_w01_new, nn_b1 = nn_b1_new;
    else
      nn_w01 = nn_w01_old, nn_b1 = nn_b1_old;
    uint64_t masks[12] = {b.white_pawns, b.white_knights, b.white_bishops, b.white_rooks, b.white_queens, 1ull<<(b.white_king_i<<3|b.white_king_j),
                          b.black_pawns, b.black_knights, b.black_bishops, b.black_rooks, b.black_queens, 1ull<<(b.black_king_i<<3|b.black_king_j)};
    for (int j = 0; j < s1; ++j)
      if (quantize)
        a1_pre_relu_q[0][j] = nn_b1_q[j];
      else
        a1_pre_relu[0][j] = nn_b1[j];
    if (dual_nnue)
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[1][j] = nn_b1_q[j];
        else
          a1_pre_relu[1][j] = nn_b1[j];
    for (int i = 0; i < 12; ++i)
      for (int j = 0; j < s1; ++j)
        for (uint64_t pieces = masks[i]; pieces; pieces &= pieces - 1) {
          int idx = (i<<6) + __builtin_ctzll(pieces);
          if (quantize)
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
            if (quantize)
              a1_pre_relu_q[1][j] += nn_w01_q[idx][j];
            else
              a1_pre_relu[1][j] += nn_w01[idx][j];
          }
      }
    }

    if (!b.king04_moved && !b.rook07_moved) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[768][j];
        else
          a1_pre_relu[0][j] += nn_w01[768][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[770][j];
          else
            a1_pre_relu[1][j] += nn_w01[770][j];
    }
    if (!b.king04_moved && !b.rook00_moved) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[769][j];
        else
          a1_pre_relu[0][j] += nn_w01[769][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[771][j];
          else
            a1_pre_relu[1][j] += nn_w01[771][j];
    }
    if (!b.king74_moved && !b.rook77_moved) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[770][j];
        else
          a1_pre_relu[0][j] += nn_w01[770][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[768][j];
          else
            a1_pre_relu[1][j] += nn_w01[768][j];
    }
    if (!b.king74_moved && !b.rook70_moved) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += nn_w01_q[771][j];
        else
          a1_pre_relu[0][j] += nn_w01[771][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += nn_w01_q[769][j];
          else
            a1_pre_relu[1][j] += nn_w01[769][j];
    }

    float ep_val = 1;
    for (int j = 0; j < s1; ++j)
      if (quantize)
        a1_pre_relu_q[0][j] += nn_w01_q[772][j];
      else
        a1_pre_relu[0][j] += nn_w01[772][j] * ep_val;
    if constexpr (dual_nnue) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[1][j] += nn_w01_q[772][j];
        else
          a1_pre_relu[1][j] += nn_w01[772][j] * ep_val;
    }
  }

  void update(const board &b, const board &next_b) {
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
        if (quantize) {
          add_feats<s1,sub_mode>(a1_pre_relu_q[0], nn_w01_q[offset_w | sq]);
          if (dual_nnue)
            add_feats<s1,sub_mode>(a1_pre_relu_q[1], nn_w01_q[offset_b | sq ^ 56]);
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
        if (quantize) {
          add_feats<s1,add_mode>(a1_pre_relu_q[0], nn_w01_q[offset_w | sq]);
          if (dual_nnue)
            add_feats<s1,add_mode>(a1_pre_relu_q[1], nn_w01_q[offset_b | sq ^ 56]);
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
        if (quantize)
          a1_pre_relu_q[0][j] += dc1 * nn_w01_q[768][j];
        else
          a1_pre_relu[0][j] += dc1 * nn_w01[768][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += dc1 * nn_w01_q[770][j];
          else
            a1_pre_relu[1][j] += dc1 * nn_w01[770][j];
    }
    if (dc2) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += dc2 * nn_w01_q[769][j];
        else
          a1_pre_relu[0][j] += dc2 * nn_w01[769][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += dc2 * nn_w01_q[771][j];
          else
            a1_pre_relu[1][j] += dc2 * nn_w01[771][j];
    }
    if (dc3) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += dc3 * nn_w01_q[770][j];
        else
          a1_pre_relu[0][j] += dc3 * nn_w01[770][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += dc3 * nn_w01_q[768][j];
          else
            a1_pre_relu[1][j] += dc3 * nn_w01[768][j];
    }
    if (dc4) {
      for (int j = 0; j < s1; ++j)
        if (quantize)
          a1_pre_relu_q[0][j] += dc4 * nn_w01_q[771][j];
        else
          a1_pre_relu[0][j] += dc4 * nn_w01[771][j];
      if (dual_nnue)
        for (int j = 0; j < s1; ++j)
          if (quantize)
            a1_pre_relu_q[1][j] += dc4 * nn_w01_q[769][j];
          else
            a1_pre_relu[1][j] += dc4 * nn_w01[769][j];
    }
  }
};

template<bool use_new, int s1=use_new?s1_new:s1_old, int s2=use_new?s2_new:s2_old, int s3=use_new?s3_new:s3_old, int s4=use_new?s4_new:s4_old>
float forward_pass(board b, const nnue_state_t<use_new>& nnue_state, const float (&w01)[s0][s1], const float (&b1)[s1], const float (&w12)[s1][s2], const float (&b2)[s2],
                                                                     const float (&w23)[s2][s3], const float (&b3)[s3], const float (&w34)[s3][s4], const float (&b4)[s4],
                                                                     const float (&w10)[s1][s0],                        const float (&w21)[s2][s1],
                                                                     const float (&w32)[s3][s2],                        const float (&w43)[s4][s3]) {
  bool flip = b.side2move == BLACK;
  alignas(32) float a1[s1], a2[s2], a3[s3];
  static constexpr bool quantize = nnue_state_t<use_new>::quantize;
  for (int j = 0; j < s1; ++j)
    a1[j] = max(0.f, quantize ? q_step_01 * nnue_state.a1_pre_relu_q[dual_nnue?flip:0][j] : nnue_state.a1_pre_relu[dual_nnue?flip:0][j]);

  if (false) {  // ...under g++ -O1 both shows +220 Elo, under g++ -O2 both shows +90 Elo, under g++ -O3 both shows +82 Elo, under g++ -Ofast both shows -52 Elo
                  // later with alignas: @.2s 30020:30324 mb -2 Elo p=.094
                  // later @.2s 4923:5235 -11 Elo
    __m256 v_a_old, v_a_new = _mm256_load_ps(b2), v_wk;
    for (int k = 0; k < s1; ++k) {
      v_a_old = _mm256_set1_ps(a1[k]);  // broadcasts float32_t to 8xfloat32_t
      v_wk = _mm256_load_ps(w12[k]);
      v_a_new = _mm256_fmadd_ps(v_a_old, v_wk, v_a_new);  // fma: v_a2 += v_a1 * v_w12;
    }
    v_a_new = _mm256_max_ps(v_a_new, _mm256_setzero_ps());  // relu
    _mm256_store_ps(a2, v_a_new);
    v_a_new = _mm256_load_ps(b3);
    for (int k = 0; k < s2; ++k) {
      v_a_old = _mm256_set1_ps(a2[k]);
      v_wk = _mm256_load_ps(w23[k]);
      v_a_new = _mm256_fmadd_ps(v_a_old, v_wk, v_a_new);
    }
    v_a_new = _mm256_max_ps(v_a_new, _mm256_setzero_ps());
    _mm256_store_ps(a3, v_a_new);
    _mm256_zeroupper();  // n'aide pas, maintenant 2014.5:2217.5 -17 Elo
  } else if (false) {  // not yet
    const int shift = 6;
    int a1_[s1], a2_[s2], a3_[s3];
    for (int i = 0; i < s1; ++i)
      a1_[i] = int(a1[i]) << 8;// / q_step_12;
    for (int j = 0; j < s2; ++j)
      a2_[j] = nn_b2_new_q[j];
    for (int i = 0; i < s1; ++i)
      for (int j = 0; j < s2; ++j)
        a2_[j] += nn_w21_new_q[j][i] * a1_[i];
    for (int j = 0; j < s2; ++j)
      a2_[j] = max(0, a2_[j]) >> shift;
    for (int j = 0; j < s3; ++j)
      a3_[j] = b3[j];
    for (int i = 0; i < s2; ++i)
      for (int j = 0; j < s3; ++j)
        a3_[j] += w32[j][i] * a2_[i];
    for (int j = 0; j < s3; ++j)
      a3_[j] = max(0, a3_[j]) >> shift, a3[j] = a3_[j] * q_step_23;
    //std::cout << a1_[0] << " " << a2_[0] << " " << a3_[0] << std::endl;
  } else if (false) {  // @.1s 48355:49324 -3 Elo
    for (int j = 0; j < s2; ++j)
      a2[j] = b2[j];
      for (int j = 0; j < s2; ++j)
    for (int i = 0; i < s1; ++i)
        a2[j] += w21[j][i] * a1[i];
    for (int j = 0; j < s2; ++j)
      a2[j] = max(0.f, a2[j]);
    for (int j = 0; j < s3; ++j)
      a3[j] = b3[j];
      for (int j = 0; j < s3; ++j)
    for (int i = 0; i < s2; ++i)
        a3[j] += w32[j][i] * a2[i];
    for (int j = 0; j < s3; ++j)
      a3[j] = max(0.f, a3[j]);
  } else if (false) {  // @.1s 11324:11991 -10 Elo
    for (int j = 0; j < s2; ++j)
      a2[j] = b2[j];
    for (int i = 0; i < s1; ++i)
      for (int j = 0; j < s2; ++j)
        a2[j] += w12[i][j] * a1[i];
    for (int j = 0; j < s2; ++j)
      a2[j] = max(0.f, a2[j]);
    for (int j = 0; j < s3; ++j)
      a3[j] = b3[j];
    for (int i = 0; i < s2; ++i)
      for (int j = 0; j < s3; ++j)
        a3[j] += w23[i][j] * a2[i];
    for (int j = 0; j < s3; ++j)
      a3[j] = max(0.f, a3[j]);
  } else {
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
  }
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
  float max_w = 0;
  for (int i = 0; i < s1_new; ++i)
    for (int j = 0; j < s2_new; ++j)
      max_w = max(max_w, abs(nn_w12_new[i][j]));
  float scale_w12 = std::numeric_limits<late_weight_t>::max() / max_w;
  for (int j = 0; j < s2_new; ++j) {
    nn_b2_new_q[j] = nn_b2_new[j] * scale_w12;
    for (int i = 0; i < s1_new; ++i)
      nn_w21_new_q[j][i] = nn_w21_new[j][i] * scale_w12;
  }
  max_w = 0;
  for (int i = 0; i < s2_new; ++i)
    for (int j = 0; j < s3_new; ++j)
      max_w = max(max_w, abs(nn_w23_new[i][j]));
  float scale_w23 = std::numeric_limits<late_weight_t>::max() / max_w;
  for (int j = 0; j < s3_new; ++j) {
    nn_b3_new_q[j] = nn_b3_new[j] * scale_w23;
    for (int i = 0; i < s2_new; ++i)
      nn_w32_new_q[j][i] = nn_w32_new[j][i] * scale_w23;
  }
  max_w = 0;
  for (int i = 0; i < s3_new; ++i)
    for (int j = 0; j < s4_new; ++j)
      max_w = max(max_w, abs(nn_w34_new[i][j]));
  float scale_w34 = std::numeric_limits<late_weight_t>::max() / max_w;
  for (int j = 0; j < s4_new; ++j) {
    nn_b4_new_q[j] = nn_b4_new[j] * scale_w34;
    for (int i = 0; i < s3_new; ++i)
      nn_w43_new_q[j][i] = nn_w43_new[j][i] * scale_w34;
  }

  float max_a_global = 0;
//  for (int i = 0; i < s0; ++i)
//    for (int j = 0; j < s1_new; ++j)
//      w = nn_w01_new[i][j], max_w = max(max_w, abs(w));
  //double mean = sum / (s0 * s1_new), std = sqrt(sum_sq/(s0*s1_new) - mean*mean);
  std::vector<std::string> calibration_fens = {"r1bqkbnr/ppppp1p1/2n5/4Pp1p/2P5/P7/1P1P1PPP/RNBQKBNR w KQkq - "};
  for (const board b : calibration_fens) {
    nnue_state_t<false> nnue_state(b);
    for (int i = 0; i < s1_new; ++i)
      max_a_global = max(max_a_global, relu(nnue_state.a1_pre_relu[0][i]));
  }
  //int bs1 = round(log2(63 / max_a_global));
  float scale_a1 = 63 / max_a_global;
  // layer1 is not quantized, after layer1 we get float32 activations that we *=scale_a1 and int() to make them int8_t in -63..63 a.s.
  // this is a good start, then we multiply by int8-quantized weights l1->l2, get our int32 accumulators
  // we got our int32 a2, which we further >>=bs1 to make them int8_t in -63..63
  max_a_global = 0;
  for (const board b : calibration_fens) {
    nnue_state_t<false> nnue_state(b);
    float a1[s1_new];
    for (int i = 0; i < s1_new; ++i)
      a1[i] = relu(nnue_state.a1_pre_relu[0][i]);
    // ...
  }

//  q_step_01 = max_w / std::numeric_limits<weight_t>::max();
//  sum = 0, sum_sq = 0, max_w = 0;
//  for (int i = 0; i < s1_new; ++i)
//    for (int j = 0; j < s2_new; ++j)
//      w = nn_w12_new[i][j], max_w = max(max_w, abs(w));
//  q_step_12 = max_w / std::numeric_limits<late_weight_t>::max();
//  max_w = 0;
//  for (int i = 0; i < s2_new; ++i)
//    for (int j = 0; j < s3_new; ++j)
//      max_w = max(max_w, abs(w));
//  q_step_23 = max_w / std::numeric_limits<late_weight_t>::max();
//  std::cout << "mean " << mean << " std " << std << " chose q_steps " << q_step_01 << " " << q_step_12 << " " << q_step_23 << " " << std::endl;
//  for (int j = 0; j < s1_new; ++j) {
//    nn_b1_q[j] = clamp(round(nn_b1_new[j] / q_step_01), std::numeric_limits<accumulator_t>::min() + 1, std::numeric_limits<accumulator_t>::max() - 1);
//    for (int i = 0; i < s0; ++i)
//       nn_w01_q[i][j] = clamp(round(nn_w01_new[i][j] / q_step_01), std::numeric_limits<weight_t>::min() + 1, std::numeric_limits<weight_t>::max() - 1);
//  }
//  for (int j = 0; j < s2_new; ++j) {
//    nn_b2_new_q[j] = clamp(round(nn_b2_new[j] / q_step_12), std::numeric_limits<int>::min() + 1, std::numeric_limits<int>::max() - 1);
//    for (int i = 0; i < s1_new; ++i)
//      nn_w21_new_q[j][i] = clamp(round(nn_w21_new[j][i] / q_step_12), std::numeric_limits<late_weight_t>::min() + 1, std::numeric_limits<late_weight_t>::max() - 1);
//  }
//  for (int j = 0; j < s3_new; ++j) {
//    nn_b3_new_q[j] = clamp(round(nn_b3_new[j] / q_step_23), std::numeric_limits<int>::min() + 1, std::numeric_limits<int>::max() - 1);
//    for (int i = 0; i < s2_new; ++i)
//      nn_w32_new_q[j][i] = clamp(round(nn_w32_new[j][i] / q_step_23), std::numeric_limits<late_weight_t>::min() + 1, std::numeric_limits<late_weight_t>::max() - 1);
//  }
//
//  board calibration_b("r1bqkbnr/ppppp1p1/2n5/4Pp1p/2P5/P7/1P1P1PPP/RNBQKBNR w KQkq - ");
//  float a1[s1_new], a2[s2_new], a3[s3_new];
//  const auto& a1pre = nnue_state_t<false>(b).a1_pre_relu;
//  for (int i = 0; i < s1_new; ++i)
//    a1[i] = relu(a1pre);
//  // we want max a1 to be around 63
//  //int bs1 = round(log2(63 / max_a));
//
//  for (int j = 0; j < s2_new; ++j)
//    a2[j] = b2[j];
//  for (int i = 0; i < s1_new; ++i)
//    for (int j = 0; j < s2_new; ++j)
//      a2[j] += nn_w21_new[j][i] * a1[i];
//  for (int j = 0; j < s2_new; ++j)
//    a2[j] = max(0.f, a2[j]);
//  for (int j = 0; j < s3_new; ++j)
//    a3[j] = b3[j];
//  for (int i = 0; i < s2_new; ++i)
//    for (int j = 0; j < s3_new; ++j)
//      a3[j] += nn_w32_new[j][i] * a2[i];
//  for (int j = 0; j < s3_new; ++j)
//    a3[j] = max(0.f, a3[j]);
//  float output = b4[0];
//  for (int k = 0; k < s3_new; ++k)
//    output += a3[k] * nn_w34_new[k][0];
//  float max_a = 0;
//  for (int i = 0; i < s1_new; ++i)
//    max_a = max(max_a, abs(a1[i]));
}

template<bool use_new>
float nn_eval(board b, const nnue_state_t<use_new>& nnue_state) {
  if constexpr (use_new) {
    return forward_pass<use_new>(b, nnue_state, nn_w01_new, nn_b1_new, nn_w12_new, nn_b2_new, nn_w23_new, nn_b3_new, nn_w34_new, nn_b4_new, nn_w10_new, nn_w21_new, nn_w32_new, nn_w43_new);
  } else {
    return forward_pass<use_new>(b, nnue_state, nn_w01_old, nn_b1_old, nn_w12_old, nn_b2_old, nn_w23_old, nn_b3_old, nn_w34_old, nn_b4_old, nn_w10_old, nn_w21_old, nn_w32_old, nn_w43_old);
  }
}

#endif
