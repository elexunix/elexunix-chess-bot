#include <cassert>
#include <fstream>
#include <immintrin.h>
#ifndef INCLUDE_NN_EVAL
#define INCLUDE_NN_EVAL
#include "board.h"

const std::string models_dir = "/home/elexunix/aprog/cpp/tests/tests06__/test0655-chess/bitmask-v9-nnue/models/";
const std::string filename_old = models_dir+"model_v9_128_8_8.bin",//"model9h12.bin",
                  //filename_new = "model10b0_tmp.bin";
                  filename_new = models_dir+"model_v9_128_8_8.bin";
float q_step_01, q_step_12, q_step_23, q_step_34;

constexpr int s0 = 768 + 5;

constexpr int s0_old = 773, s1_old = 128, s2_old = 8, s3_old = 8, s4_old = 1;
alignas(32) static float nn_w01_old[s0][s1_old], nn_w10_old[s1_old][s0_old], nn_b1_old[s1_old],
             nn_w12_old[s1_old][s2_old], nn_w21_old[s2_old][s1_old], nn_b2_old[s2_old],
             nn_w23_old[s2_old][s3_old], nn_w32_old[s3_old][s2_old], nn_b3_old[s3_old],
             nn_w34_old[s3_old][s4_old], nn_w43_old[s4_old][s3_old], nn_b4_old[s4_old];

constexpr int s0_new = 773/*772+40960*/, s1_new = 128, s2_new = 8, s3_new = 8, s4_new = 1;
alignas(32) static float nn_w01_new[s0_new][s1_new], nn_w10_new[s1_new][s0_new], nn_b1_new[s1_new],
             nn_w12_new[s1_new][s2_new], nn_w21_new[s2_new][s1_new], nn_b2_new[s2_new],
             nn_w23_new[s2_new][s3_new], nn_w32_new[s3_new][s2_new], nn_b3_new[s3_new],
             nn_w34_new[s3_new][s4_new], nn_w43_new[s4_new][s3_new], nn_b4_new[s4_new];
// ^T @.1s 77079:75865 +3 Elo

template<typename T>
T relu(T x) { return max(T(0), x); }

template<bool use_new>
float nn_eval(board b);

template<bool use_new>
struct nnue_state_t {  //   @3s 304.5:232.5 +47 Elo, then dual_nnue:nnue @.3s 184:125 +67 Elo
  static constexpr int s1 = use_new ? s1_new : s1_old;
  const float (*nn_w01)[s1], *nn_b1;
  alignas(32) float a1_pre_relu[2][s1];  // 0 for side2move==WHITE, 1 for BLACK

  static constexpr bool use_halfKP = false;

  nnue_state_t(const board& b) /*: nn_w01(use_new ? nn_w01_new : nn_w01_old), nn_b1(use_new ? nn_b1_new : nn_b1_old)*/ {  //, const float nn_w01[s0][s1], const float nn_b1[s1]) : nn_w01(nn_w01), nn_b1(nn_b1) {
    if constexpr (use_new)
      nn_w01 = nn_w01_new, nn_b1 = nn_b1_new;
    else
      nn_w01 = nn_w01_old, nn_b1 = nn_b1_old;
    int white_king = b.white_king_i<<3|b.white_king_j, black_king = b.black_king_i<<3|b.black_king_j, white_king_summand = 772+white_king*640, black_king_summand = 772+(black_king^56)*640;
    uint64_t masks[12] = {b.white_pawns, b.white_knights, b.white_bishops, b.white_rooks, b.white_queens, 1ull<<white_king, b.black_pawns, b.black_knights, b.black_bishops, b.black_rooks, b.black_queens, 1ull<<black_king};
    for (int j = 0; j < s1; ++j)
      a1_pre_relu[0][j] = nn_b1[j];
    for (int j = 0; j < s1; ++j)
      a1_pre_relu[1][j] = nn_b1[j];
    for (int i = 0; i < 12; ++i) {
      int i_fl = i<6 ? i+6 : i-6;
      for (uint64_t pieces = masks[i]; pieces; pieces &= pieces - 1) {
        int sq = __builtin_ctzll(pieces), iA = i<<6 | sq, iB = i_fl<<6 | sq^56, i1 = white_king_summand + (i<<6 | sq), i2 = white_king_summand + (i-1<<6 | sq), i3 = black_king_summand + (i_fl<<6 | sq^56), i4 = black_king_summand + (i_fl-1<<6 | sq^56);
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[0][j] += nn_w01[iA][j];
        if (use_halfKP && i < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] += nn_w01[i1][j];
        if (use_halfKP && 6 <= i && i < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] += nn_w01[i2][j];
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[1][j] += nn_w01[iB][j];
        if (use_halfKP && i_fl < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] += nn_w01[i3][j];
        if (use_halfKP && 6 <= i_fl && i_fl < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] += nn_w01[i4][j];
      }
    }

    if (!b.king04_moved && !b.rook07_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[768][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[770][j];
    }
    if (!b.king04_moved && !b.rook00_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[769][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[771][j];
    }
    if (!b.king74_moved && !b.rook77_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[770][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[768][j];
    }
    if (!b.king74_moved && !b.rook70_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[771][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[769][j];
    }

    if (!use_halfKP) {
      float ep_val = 1;
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[772][j] * ep_val;
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[772][j] * ep_val;
    }
  }

  void update(const board &b1, const board &b2) {
    //assert(0 <= m.i0 && m.i0 < 8 && 0 <= m.j0 && m.j0 < 8 && 0 <= m.i1 && m.i1 < 8 && 0 <= m.j1 && m.j1 < 8);
    uint64_t b1_white_king = b1.white_king_i<<3|b1.white_king_j, b1_black_king = b1.black_king_i<<3|b1.black_king_j, b1_white_king_summand = 772+b1_white_king*640, b1_black_king_summand = 772+(b1_black_king^56)*640,
             b2_white_king = b2.white_king_i<<3|b2.white_king_j, b2_black_king = b2.black_king_i<<3|b2.black_king_j, b2_white_king_summand = 772+b2_white_king*640, b2_black_king_summand = 772+(b2_black_king^56)*640;
    uint64_t masks_b1[12] = {b1.white_pawns, b1.white_knights, b1.white_bishops, b1.white_rooks, b1.white_queens, 1ull<<b1_white_king,
                             b1.black_pawns, b1.black_knights, b1.black_bishops, b1.black_rooks, b1.black_queens, 1ull<<b1_black_king},
             masks_b2[12] = {b2.white_pawns, b2.white_knights, b2.white_bishops, b2.white_rooks, b2.white_queens, 1ull<<b2_white_king,
                             b2.black_pawns, b2.black_knights, b2.black_bishops, b2.black_rooks, b2.black_queens, 1ull<<b2_black_king};
    for (int i = 0; i < 12; ++i) {
      int i_fl = i<6 ? i+6 : i-6;
      // assume for now kings have not moved, everything revolves around b1_white_king and b1_black_king
      for (uint64_t mask = masks_b1[i] & ~masks_b2[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask), iA = i<<6 | sq, iB = i_fl<<6 | sq^56, i1 = b1_white_king_summand + (i<<6 | sq), i2 = b1_white_king_summand + (i-1<<6 | sq), i3 = b1_black_king_summand + (i_fl<<6 | sq^56), i4 = b1_black_king_summand + (i_fl-1<<6 | sq^56);
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[0][j] -= nn_w01[iA][j];
        if (use_halfKP && i < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] -= nn_w01[i1][j];
        if (use_halfKP && 6 <= i && i < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] -= nn_w01[i2][j];
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[1][j] -= nn_w01[iB][j];
        if (use_halfKP && i_fl < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] -= nn_w01[i3][j];
        if (use_halfKP && 6 <= i_fl && i_fl < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] -= nn_w01[i4][j];
      }
      for (uint64_t mask = masks_b2[i] & ~masks_b1[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask), iA = i<<6 | sq, iB = i_fl<<6 | sq^56, i1 = b1_white_king_summand + (i<<6 | sq), i2 = b1_white_king_summand + (i-1<<6 | sq), i3 = b1_black_king_summand + (i_fl<<6 | sq^56), i4 = b1_black_king_summand + (i_fl-1<<6 | sq^56);
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[0][j] += nn_w01[iA][j];
        if (use_halfKP && i < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] += nn_w01[i1][j];
        if (use_halfKP && 6 <= i && i < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[0][j] += nn_w01[i2][j];
        for (int j = 0; j < s1; ++j)
          a1_pre_relu[1][j] += nn_w01[iB][j];
        if (use_halfKP && i_fl < 5)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] += nn_w01[i3][j];
        if (use_halfKP && 6 <= i_fl && i_fl < 11)
          for (int j = 0; j < s1; ++j)
            a1_pre_relu[1][j] += nn_w01[i4][j];
      }
      // if kings have moved...
      if (use_halfKP && b2_white_king != b1_white_king) {
        for (uint64_t mask = masks_b2[i]; mask; mask &= mask - 1) {
          int sq = __builtin_ctzll(mask), i1 = b1_white_king_summand + (i<<6 | sq), i2 = b2_white_king_summand + (i<<6 | sq), i3 = b1_white_king_summand + (i-1<<6 | sq), i4 = b2_white_king_summand + (i-1<<6 | sq);
          if (i < 5)
            for (int j = 0; j < s1; ++j)
              a1_pre_relu[0][j] += nn_w01[i2][j] - nn_w01[i1][j];
          if (6 <= i && i < 11)
            for (int j = 0; j < s1; ++j)
              a1_pre_relu[0][j] += nn_w01[i4][j] - nn_w01[i3][j];
        }
      }
      if (use_halfKP && b2_black_king != b1_black_king) {
        for (uint64_t mask = masks_b2[i]; mask; mask &= mask - 1) {
          int sq56 = __builtin_ctzll(mask)^56, i1 = b1_black_king_summand + (i_fl<<6 | sq56), i2 = b2_black_king_summand + (i_fl<<6 | sq56), i3 = b1_black_king_summand + (i_fl-1<<6 | sq56), i4 = b2_black_king_summand + (i_fl-1<<6 | sq56);
          if (i_fl < 5)
            for (int j = 0; j < s1; ++j)
              a1_pre_relu[1][j] += nn_w01[i2][j] - nn_w01[i1][j];
          if (6 <= i_fl && i_fl < 11)
            for (int j = 0; j < s1; ++j)
              a1_pre_relu[1][j] += nn_w01[i4][j] - nn_w01[i3][j];
        }
      }
    }
    int dc1 = (!b2.king04_moved && !b2.rook07_moved) - (!b1.king04_moved && !b1.rook07_moved), dc2 = (!b2.king04_moved && !b2.rook00_moved) - (!b1.king04_moved && !b1.rook00_moved),
        dc3 = (!b2.king74_moved && !b2.rook77_moved) - (!b1.king74_moved && !b1.rook77_moved), dc4 = (!b2.king74_moved && !b2.rook70_moved) - (!b1.king74_moved && !b1.rook70_moved),
        dep = (b2.en_passant_sq != 0) - (b1.en_passant_sq != 0);
    if (dc1) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc1 * nn_w01[768][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc1 * nn_w01[770][j];
    }
    if (dc2) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc2 * nn_w01[769][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc2 * nn_w01[771][j];
    }
    if (dc3) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc3 * nn_w01[770][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc3 * nn_w01[768][j];
    }
    if (dc4) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc4 * nn_w01[771][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc4 * nn_w01[769][j];
    }
  }

  /*nnue_state_t(const board& b) {  //, const float nn_w01[s0][s1], const float nn_b1[s1]) : nn_w01(nn_w01), nn_b1(nn_b1) {
    if constexpr (use_new)
      nn_w01 = nn_w01_new, nn_b1 = nn_b1_new;
    else
      nn_w01 = nn_w01_old, nn_b1 = nn_b1_old;
    int white_king = b.white_king_i<<3|b.white_king_j, black_king = b.black_king_i<<3|b.black_king_j, white_king_summand = 772+white_king*640, black_king_summand = 772+(black_king^56)*640;
    uint64_t masks[12] = {b.white_pawns, b.white_knights, b.white_bishops, b.white_rooks, b.white_queens, 1ull<<white_king, b.black_pawns, b.black_knights, b.black_bishops, b.black_rooks, b.black_queens, 1ull<<black_king};
    for (int j = 0; j < s1; ++j)
      a1_pre_relu[0][j] = nn_b1[j];
    for (int j = 0; j < s1; ++j)
      a1_pre_relu[1][j] = nn_b1[j];
    for (int i = 0; i < 12; ++i) {
      int i_fl = i<6 ? i+6 : i-6;
      for (uint64_t pieces = masks[i]; pieces; pieces &= pieces - 1) {
        int sq = __builtin_ctzll(pieces), sq56 = sq^56;
        for (int j = 0; j < s1; ++j) {
          a1_pre_relu[0][j] += nn_w01[i<<6 | sq][j];
          if (use_halfKP && i < 5)
            a1_pre_relu[0][j] += nn_w01[white_king_summand + (i<<6 | sq)][j];
          if (use_halfKP && 6 <= i && i < 11)
            a1_pre_relu[0][j] += nn_w01[white_king_summand + (i-1<<6 | sq)][j];
          a1_pre_relu[1][j] += nn_w01[i_fl<<6 | sq^56][j];
          if (use_halfKP && i_fl < 5)
            a1_pre_relu[1][j] += nn_w01[black_king_summand + (i_fl<<6 | sq^56)][j];
          if (use_halfKP && 6 <= i_fl && i_fl < 11)
            a1_pre_relu[1][j] += nn_w01[black_king_summand + (i_fl-1<<6 | sq^56)][j];
        }
      }
    }

    if (!b.king04_moved && !b.rook07_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[768][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[770][j];
    }
    if (!b.king04_moved && !b.rook00_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[769][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[771][j];
    }
    if (!b.king74_moved && !b.rook77_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[770][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[768][j];
    }
    if (!b.king74_moved && !b.rook70_moved) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[771][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[769][j];
    }

    if (!use_halfKP) {
      float ep_val = 1;
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += nn_w01[772][j] * ep_val;
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += nn_w01[772][j] * ep_val;
    }
  }

  void update(const board &b1, const board &b2) {
    //assert(0 <= m.i0 && m.i0 < 8 && 0 <= m.j0 && m.j0 < 8 && 0 <= m.i1 && m.i1 < 8 && 0 <= m.j1 && m.j1 < 8);
    uint64_t b1_white_king = b1.white_king_i<<3|b1.white_king_j, b1_black_king = b1.black_king_i<<3|b1.black_king_j, b1_white_king_summand = 772+b1_white_king*640, b1_black_king_summand = 772+(b1_black_king^56)*640,
             b2_white_king = b2.white_king_i<<3|b2.white_king_j, b2_black_king = b2.black_king_i<<3|b2.black_king_j, b2_white_king_summand = 772+b2_white_king*640, b2_black_king_summand = 772+(b2_black_king^56)*640;
    uint64_t masks_b1[12] = {b1.white_pawns, b1.white_knights, b1.white_bishops, b1.white_rooks, b1.white_queens, 1ull<<b1_white_king,
                             b1.black_pawns, b1.black_knights, b1.black_bishops, b1.black_rooks, b1.black_queens, 1ull<<b1_black_king},
             masks_b2[12] = {b2.white_pawns, b2.white_knights, b2.white_bishops, b2.white_rooks, b2.white_queens, 1ull<<b2_white_king,
                             b2.black_pawns, b2.black_knights, b2.black_bishops, b2.black_rooks, b2.black_queens, 1ull<<b2_black_king};
    for (int i = 0; i < 12; ++i) {
      int i_fl = i<6 ? i+6 : i-6;
      // assume for now kings have not moved, everything revolves around b1_white_king and b1_black_king
      for (uint64_t mask = masks_b1[i] & ~masks_b2[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask), sq56 = sq^56;
        for (int j = 0; j < s1; ++j) {
          a1_pre_relu[0][j] -= nn_w01[i<<6 | sq][j];
          if (use_halfKP && i < 5)
            a1_pre_relu[0][j] -= nn_w01[b1_white_king_summand + (i<<6 | sq)][j];
          if (use_halfKP && 6 <= i && i < 11)
            a1_pre_relu[0][j] -= nn_w01[b1_white_king_summand + (i-1<<6 | sq)][j];
          a1_pre_relu[1][j] -= nn_w01[i_fl<<6 | sq56][j];
          if (use_halfKP && i_fl < 5)
            a1_pre_relu[1][j] -= nn_w01[b1_black_king_summand + (i_fl<<6 | sq56)][j];
          if (use_halfKP && 6 <= i_fl && i_fl < 11)
            a1_pre_relu[1][j] -= nn_w01[b1_black_king_summand + (i_fl-1<<6 | sq56)][j];
        }
      }
      for (uint64_t mask = masks_b2[i] & ~masks_b1[i]; mask; mask &= mask - 1) {
        int sq = __builtin_ctzll(mask), sq56 = sq^56;
        for (int j = 0; j < s1; ++j) {
          a1_pre_relu[0][j] += nn_w01[i<<6 | sq][j];
          if (use_halfKP && i < 5)
            a1_pre_relu[0][j] += nn_w01[b1_white_king_summand + (i<<6 | sq)][j];
          if (use_halfKP && 6 <= i && i < 11)
            a1_pre_relu[0][j] += nn_w01[b1_white_king_summand + (i-1<<6 | sq)][j];
          a1_pre_relu[1][j] += nn_w01[i_fl<<6 | sq56][j];
          if (use_halfKP && i_fl < 5)
            a1_pre_relu[1][j] += nn_w01[b1_black_king_summand + (i_fl<<6 | sq56)][j];
          if (use_halfKP && 6 <= i_fl && i_fl < 11)
            a1_pre_relu[1][j] += nn_w01[b1_black_king_summand + (i_fl-1<<6 | sq56)][j];
        }
      }
      // if kings have moved...
      if (use_halfKP && b2_white_king != b1_white_king) {
        for (uint64_t mask = masks_b2[i]; mask; mask &= mask - 1) {
          int sq = __builtin_ctzll(mask);
          for (int j = 0; j < s1; ++j) {
            if (i < 5)
              a1_pre_relu[0][j] += nn_w01[b2_white_king_summand + (i<<6 | sq)][j] - nn_w01[b1_white_king_summand + (i<<6 | sq)][j];
            if (6 <= i && i < 11)
              a1_pre_relu[0][j] += nn_w01[b2_white_king_summand + (i-1<<6 | sq)][j] - nn_w01[b1_white_king_summand + (i-1<<6 | sq)][j];
          }
        }
      }
      if (use_halfKP && b2_black_king != b1_black_king) {
        for (uint64_t mask = masks_b2[i]; mask; mask &= mask - 1) {
          int sq = __builtin_ctzll(mask), sq56 = sq^56;
          for (int j = 0; j < s1; ++j) {
            if (i_fl < 5)
              a1_pre_relu[1][j] += nn_w01[b2_black_king_summand + (i_fl<<6 | sq56)][j] - nn_w01[b1_black_king_summand + (i_fl<<6 | sq56)][j];
            if (6 <= i_fl && i_fl < 11)
              a1_pre_relu[1][j] += nn_w01[b2_black_king_summand + (i_fl-1<<6 | sq56)][j] - nn_w01[b1_black_king_summand + (i_fl-1<<6 | sq56)][j];
          }
        }
      }
    }
    int dc1 = (!b2.king04_moved && !b2.rook07_moved) - (!b1.king04_moved && !b1.rook07_moved), dc2 = (!b2.king04_moved && !b2.rook00_moved) - (!b1.king04_moved && !b1.rook00_moved),
        dc3 = (!b2.king74_moved && !b2.rook77_moved) - (!b1.king74_moved && !b1.rook77_moved), dc4 = (!b2.king74_moved && !b2.rook70_moved) - (!b1.king74_moved && !b1.rook70_moved),
        dep = (b2.en_passant_sq != 0) - (b1.en_passant_sq != 0);
    if (dc1) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc1 * nn_w01[768][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc1 * nn_w01[770][j];
    }
    if (dc2) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc2 * nn_w01[769][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc2 * nn_w01[771][j];
    }
    if (dc3) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc3 * nn_w01[770][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc3 * nn_w01[768][j];
    }
    if (dc4) {
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[0][j] += dc4 * nn_w01[771][j];
      for (int j = 0; j < s1; ++j)
        a1_pre_relu[1][j] += dc4 * nn_w01[769][j];
    }
  }*/
};

template<bool use_new, int s0=use_new?s0_new:s0_old, int s1=use_new?s1_new:s1_old, int s2=use_new?s2_new:s2_old, int s3=use_new?s3_new:s3_old, int s4=use_new?s4_new:s4_old>
float forward_pass(board b, const nnue_state_t<use_new>& nnue_state, const float (&w01)[s0][s1], const float (&b1)[s1], const float (&w12)[s1][s2], const float (&b2)[s2],
                                                                     const float (&w23)[s2][s3], const float (&b3)[s3], const float (&w34)[s3][s4], const float (&b4)[s4],
                                                                     const float (&w10)[s1][s0],                        const float (&w21)[s2][s1],
                                                                     const float (&w32)[s3][s2],                        const float (&w43)[s4][s3]) {
  bool flip = b.side2move == BLACK;
  alignas(32) float a1[s1], a2[s2], a3[s3];
  for (int j = 0; j < s1; ++j)
    a1[j] = max(0.f, nnue_state.a1_pre_relu[flip][j]);

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

  output = clamp(output, -.999f, .999f);  // fix clamp here @.3s 2505:2291 mb +16 Elo, @3s 2547:2331 +15 Elo
  return 400.f * -log(2.f / (output + 1.f) - 1.f);
}

template<int s0, int s1, int s2, int s3, int s4>
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
  read_net<s0_old,s1_old,s2_old,s3_old,s4_old>(filename_old, nn_w01_old, nn_b1_old, nn_w12_old, nn_b2_old, nn_w23_old, nn_b3_old, nn_w34_old, nn_b4_old, nn_w10_old, nn_w21_old, nn_w32_old, nn_w43_old);
  read_net<s0_new,s1_new,s2_new,s3_new,s4_new>(filename_new, nn_w01_new, nn_b1_new, nn_w12_new, nn_b2_new, nn_w23_new, nn_b3_new, nn_w34_new, nn_b4_new, nn_w10_new, nn_w21_new, nn_w32_new, nn_w43_new);
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
