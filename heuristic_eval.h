#include <cassert>
#ifndef INCLUDE_HEURISTIC_EVAL
#define INCLUDE_HEURISTIC_EVAL
#include "board.h"
#include "nn_eval.h"


//static const int cost_pawn = 10, cost_knight = 30, cost_bishop = 30, cost_rook = 50, cost_queen = 90;
constexpr int cost_pawn = 100, cost_knight = 300, cost_bishop = 300, cost_rook = 500, cost_queen = 900;
constexpr int material0_of_player = 8 * cost_pawn + 2 * cost_knight + 2 * cost_bishop + 2 * cost_rook + cost_queen;

constexpr uint64_t not_left_column = 0xfefefefefefefefe, not_right_column = 0x7f7f7f7f7f7f7f7f, not_top_row = 0x00ffffffffffffff, not_bottom_row = 0xffffffffffffff00, center_mask = 0x00003c3c3c3c0000,
                   white_cells = 0xaaaaaaaaaaaaaaaa, black_cells = 0x5555555555555555, row7 = 0x00ff000000000000, row2 = 0x000000000000ff00,
                   files[8] = {0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808, 0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080};
static_assert(black_cells == ~white_cells);

void precalc_heuristic_eval() {}
//int floodfill(uint64_t rooks, uint64_t obstacles) {
//  // hope if you are here, this is an endgame
//  uint64_t free = ~obstacles, reachable = rooks, old_reachable = 0;
//  while (reachable != old_reachable) {
//    old_reachable = reachable;
//    reachable |= (reachable & not_left_column) >> 1 & free;
//    reachable |= (reachable & not_right_column) << 1 & free;
//    reachable |= (reachable & not_top_row) << 8 & free;
//    reachable |= (reachable & not_bottom_row) >> 8 & free;
//  }
//  return __builtin_popcountll(reachable);
//}

template<bool use_new>
int board::pst_summands(int game_phase /*0 to 256*/) const {
  const uint64_t my_pawns   = side2move == WHITE ? white_pawns   : black_pawns;
  const uint64_t my_knights = side2move == WHITE ? white_knights : black_knights;
  const uint64_t my_bishops = side2move == WHITE ? white_bishops : black_bishops;
  const uint64_t my_rooks   = side2move == WHITE ? white_rooks   : black_rooks;
  const uint64_t my_queens  = side2move == WHITE ? white_queens  : black_queens;
  const auto my_king_i = side2move == WHITE ? white_king_i : black_king_i, my_king_j = side2move == WHITE ? white_king_j : black_king_j;
  __attribute__((assume(0 <= white_king_i < 8 and 0 <= white_king_j < 8 and 0 <= black_king_i < 8 and 0 <= black_king_j < 8)));
  int my_king = my_king_i<<3|my_king_j;
  int flip = side2move == WHITE ? 0 : 56;
  int score = 0, mg_score = 0, eg_score = 0;
//  static const int8_t nn_pst_white_pawns[64] = {0,0,0,0,0,0,0,0,-2,6,-3,-1,-14,5,12,-2,7,8,-3,-8,-1,-6,10,0,1,9,-1,2,9,-4,21,3,4,8,-3,3,5,-12,16,6,4,-5,38,-11,-13,28,3,12,46,13,25,27,62,39,40,49,0,0,0,0,0,0,0,0},
//                      nn_pst_white_knights[64] = {-51,-3,-1,3,6,23,2,-80,11,5,-2,5,4,3,1,-41,9,11,-12,-20,3,-10,18,36,33,31,-25,5,1,-5,27,18,57,24,7,-8,4,-2,25,24,71,36,11,-4,-11,-34,-15,37,11,64,32,-8,-15,31,-7,-56,-12,-76,-23,126,82,-41,90,-72},
//                      nn_pst_white_bishops[64] = {4,15,-9,-4,-6,-7,-4,-14,70,0,25,-5,8,-2,7,18,2,11,-15,-7,-14,6,8,10,-13,-13,-9,-5,-9,-16,-1,-27,-12,22,11,4,37,-23,3,11,8,18,-18,10,25,-15,16,5,-20,15,39,-16,6,71,-20,34,69,13,-30,33,95,3,99,60},
//                      nn_pst_white_rooks[64] = {-10,-3,1,2,-2,2,11,-9,-13,-14,-2,-3,4,2,20,-14,8,-8,31,-37,38,-6,3,-5,15,8,21,4,6,-21,18,4,13,31,12,8,-14,2,22,17,76,45,38,38,20,53,20,20,57,66,19,32,27,81,37,24,42,39,58,44,23,33,42,21},
//                      nn_pst_white_king[64] = {-91,-53,-49,-51,-50,-44,-50,-74,-20,-48,-53,-43,-45,-33,-53,-66,-3,-24,-40,-45,-53,-31,-44,-90,-61,-15,-39,-56,-13,-52,-23,9,28,36,-2,-3,-30,-27,-25,-46,-32,-10,-65,-34,12,-63,-18,18,-3,103,-55,23,89,127,17,127,127,-13,10,66,-112,127,127,127};
//
//  int score = 0;
//  static const int8_t pst_pawns[64] = {0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, -7, -7, 3, 3, 2, 2, -2, -3, 0, 0, -3, -2, 2, 0, 0, 0, 7, 7, 0, 0, 0, 2, 2, 3, 8, 8, 3, 2, 2, 3, 3, 7, 10, 10, 7, 3, 3, 22, 22, 22, 22, 22, 22, 22, 22, 0, 0, 0, 0, 0, 0, 0, 0};
//  static const int8_t pst_knights[64] = {-17,-13,-10,-10,-10,-10,-13,-17, -13, -7, 0, 2, 2, 0, -7,-13, -10, 2, 3, 5, 5, 3, 2,-10, -10, 0, 5, 7, 7, 5, 0,-10, -10, 2, 5, 7, 7, 5, 2,-10, -10, 0, 3, 5, 5, 3, 0,-10, -13, -7, 0, 0, 0, 0, -7,-13, -17,-13,-10,-10,-10,-10,-13,-17};
//  static const int8_t pst_bishops[64] = {-7, -3, -3, -3, -3, -3, -3, -7, -3, 2, 0, 0, 0, 0, 2, -3, -3, 3, 3, 3, 3, 3, 3, -3, -3, 0, 2, 3, 3, 2, 0, -3, -3, 2, 2, 3, 3, 2, 2, -3, -3, 0, 3, 3, 3, 3, 0, -3, -3, 0, 0, 0, 0, 0, 0, -3, -7, -3, -3, -3, -3, -3, -3, -7};
//  static const int8_t pst_rooks[64] = {0, 0, 0, 2, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, -2, 2, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0};
//  static const int8_t pst_queens[64] = {-7, -3, -3, -2, -2, -3, -3, -7, -3, 0, 2, 0, 0, 0, 0, -3, -3, 2, 2, 2, 2, 2, 0, -3, 0, 0, 2, 2, 2, 2, 0, -2, -2, 0, 2, 2, 2, 2, 0, -2, -3, 0, 2, 2, 2, 2, 0, -3, -3, 0, 0, 0, 0, 0, 0, -3, -7, -3, -3, -2, -2, -3, -3, -7};
//  static const int8_t pst_mg_king[64] = {7, 10, 3, 0, 0, 3, 10, 7, 7, 7, 0, 0, 0, 0, 7, 7, -3, -7, -7, -7, -7, -7, -7, -3, -7,-10,-10,-13,-13,-10,-10, -7, -10,-13,-13,-17,-17,-13,-13,-10, -10,-13,-13,-17,-17,-13,-13,-10, -10,-13,-13,-17,-17,-13,-13,-10, -10,-13,-13,-17,-17,-13,-13,-10};
//  static const int8_t pst_eg_king[64] = {-17,-10,-10,-10,-10,-10,-10,-17, -10,-10, 0, 0, 0, 0,-10,-10, -10, -3, 7, 10, 10, 7, -3,-10, -10, -3, 10, 13, 13, 10, -3,-10, -10, -3, 10, 13, 13, 10, -3,-10, -10, -3, 7, 10, 10, 7, -3,-10, -10, -7, -3, 0, 0, -3, -7,-10, -17,-13,-10, -7, -7,-10,-13,-17};
  //int mg_score = pst_mg_king[my_king ^ flip], eg_score = pst_eg_king[my_king ^ flip];
  static const int8_t linreg_pst_pawns[64] = {-5,-5,-5,-5,-5,-5,-5,-5,-2,1,0,-4,1,-1,-1,-4,-2,1,-1,-3,0,-2,1,-1,-2,0,1,1,-1,-1,0,0,-2,2,-1,2,0,-4,-1,-2,4,2,3,9,3,1,-1,-3,6,16,12,9,12,7,11,14,-5,-5,-5,-5,-5,-5,-5,-5},  // for wrong target, must retrain
                      linreg_pst_knights[64] = {1,-2,-2,3,1,1,-1,-3,-7,-3,1,2,4,-2,-3,-7,0,0,0,3,4,2,3,2,2,4,0,3,2,3,-4,-6,9,3,-4,3,-1,-4,5,8,-2,6,-2,0,2,0,4,-6,0,-6,4,6,1,9,-10,7,-9,8,-1,-7,-4,-17,8,-13},
                      linreg_pst_bishops[64] = {-3,-2,-2,-7,-5,-3,-5,-5,-10,3,3,0,-5,3,-2,-4,-2,10,-1,-1,1,-3,-4,-5,-2,1,2,-2,-3,0,3,3,4,0,0,0,0,-3,-3,-8,3,7,5,2,2,0,-1,-3,2,-1,-4,9,10,0,3,-7,-2,29,-2,3,7,-4,3,-3},
                      linreg_pst_rooks[64] = {-6,-4,-4,-1,-4,-2,0,-4,-4,-1,-1,-1,-6,-2,-4,-4,-3,2,-5,-5,-2,-2,-5,-3,-4,3,4,-1,-1,-1,-1,-2,1,5,-4,-2,3,0,-2,-1,1,0,0,2,-2,-2,6,1,4,0,2,1,5,4,0,5,5,8,7,10,6,5,2,4},
                      linreg_pst_queens[64] = {-6,-1,-4,-3,-6,-7,-7,1,-2,1,-3,-1,-4,-7,-3,-4,-8,-1,-3,-3,1,-1,1,4,-5,-3,-6,0,-2,8,-2,-3,-1,-7,-2,-3,-3,3,2,7,5,-3,0,2,5,3,4,7,4,-1,2,-5,-3,6,6,0,4,12,13,9,-2,14,0,1},
                      linreg_pst_kings[64] = {0,0,-1,-3,-4,-2,-1,-6,-4,0,-3,0,2,2,-4,-4,-2,-3,-2,-1,-3,-4,-6,-4,0,-2,-2,2,1,-1,-2,0,1,0,0,5,5,1,-7,1,13,11,5,7,10,-2,-4,12,-4,5,4,6,-12,6,-13,-26,11,-2,-18,-10,23,-7,15,24},
                      hc_pst_eg_kings[64] = {
                      0,0,0,0,0,0,0,0,
                      0,10,10,10,10,10,10,0,
                      0,10,20,20,20,20,10,0,
                      0,10,20,40,40,20,10,0,
                      0,10,20,40,40,20,10,0,
                      0,10,20,20,20,20,10,0,
                      0,10,10,10,10,10,10,0,
                      0,0,0,0,0,0,0,0};
  if (false) {
//    for (uint64_t pawns = my_pawns; pawns; pawns &= pawns - 1) {
//      int pawn = __builtin_ctzll(pawns);
//      score += linreg_pst_pawns[pawn ^ flip];
//    }
//    for (uint64_t knights = my_knights; knights; knights &= knights - 1) {
//      int knight = __builtin_ctzll(knights);
//      score += linreg_pst_knights[knight ^ flip];
//    }
//    for (uint64_t bishops = my_bishops; bishops; bishops &= bishops - 1) {
//      int bishop = __builtin_ctzll(bishops);
//      score += linreg_pst_bishops[bishop ^ flip];
//    }
//    for (uint64_t rooks = my_rooks; rooks; rooks &= rooks - 1) {
//      int rook = __builtin_ctzll(rooks);
//      score += linreg_pst_rooks[rook ^ flip];
//    }
//    for (uint64_t queens = my_queens; queens; queens &= queens - 1) {
//      int queen = __builtin_ctzll(queens);
//      score += linreg_pst_queens[queen ^ flip];
//    }
//    score += linreg_pst_kings[my_king ^ flip];

      //mg_score -= 20 * __builtin_popcountll((my_knights|my_bishops) & (side2move==WHITE?0xff:0xffll<<56));  // @1s 1455.5:1447.5 mb +1 Elo
  }
  eg_score += 2 * hc_pst_eg_kings[my_king ^ flip];  // @0.1s 6033.5:5691.5 +10 Elo, 4:2 @0.1s 1225.5:1073.5 +23 Elo, 2:0 @1s 232.5:170.5 +54 Elo, @1s 1143:1094 mb +8 Elo p_>=.865
  return score + ((256-game_phase) * mg_score + game_phase * eg_score) / 256;
}

template<bool use_new>
int board::one_sided_score(int game_phase) const {
  __attribute__((assume(0 <= white_king_i < 8 and 0 <= white_king_j < 8 and 0 <= black_king_i < 8 and 0 <= black_king_j < 8)));
  bool is_white2move = side2move == WHITE;
  const uint64_t my_pawns   = is_white2move ? white_pawns   : black_pawns;
  const uint64_t my_knights = is_white2move ? white_knights : black_knights;
  const uint64_t my_bishops = is_white2move ? white_bishops : black_bishops;
  const uint64_t my_rooks   = is_white2move ? white_rooks   : black_rooks;
  const uint64_t my_queens  = is_white2move ? white_queens  : black_queens;
  const auto my_king_i = is_white2move ? white_king_i : black_king_i, my_king_j = is_white2move ? white_king_j : black_king_j;
  //const int my_queen_sq = __builtin_ctzll(my_queens), my_queen_i = my_queen_sq >> 8, my_queen_j = my_queen_sq & 7;

  const uint64_t his_pawns   = is_white2move ? black_pawns   : white_pawns;
  const uint64_t his_knights = is_white2move ? black_knights : white_knights;
  const uint64_t his_bishops = is_white2move ? black_bishops : white_bishops;
  const uint64_t his_rooks   = is_white2move ? black_rooks   : white_rooks;
  const uint64_t his_queens  = is_white2move ? black_queens  : white_queens;
  const auto his_king_i = is_white2move ? black_king_i : white_king_i, his_king_j = is_white2move ? black_king_j : white_king_j;
  //const int his_queen_sq = __builtin_ctzll(his_queens), his_queen_i = his_queen_sq >> 8, his_queen_j = his_queen_sq & 7;

  int my_king_sq = my_king_i<<3|my_king_j;
  int his_king_sq = his_king_i<<3|his_king_j;
  uint64_t my_kings = 1ull<<my_king_sq, my_something = my_pawns | my_knights | my_bishops | my_rooks | my_queens | my_kings;
  uint64_t his_kings = 1ull<<his_king_sq;
  uint64_t his_something = his_pawns | his_knights | his_bishops | his_rooks | his_queens | his_kings;  // bugfix my_kings->his_kings @0.1s 23307.5:22641.5 +5 Elo

  // @0.4s 139:198 -61 Elo???

  int my_cnt_possible_moves = list_possible_moves<desire_type_t::cnt_possible_moves>();
  uint64_t I_attack = list_possible_moves<desire_type_t::attack_mask>();
  uint64_t I_perforate = list_possible_moves<desire_type_t::perforation_mask>();
  uint64_t my_king_nhood = board::king_nhoods[my_king_sq], my_king_nhood2 = board::king_nhoods2[my_king_sq];
  //uint64_t his_king_nhood = board::king_nhoods[his_king_sq];//, his_king_nhood2 = board::king_nhoods2[his_king_sq];  // 0:some_king_sq @0.1s 2609:2394 +15 Elo, his:0 @0.1s 566:680 -32 Elo

  int material = (false ? cost_pawn - 20 : cost_pawn) * __builtin_popcountll(my_pawns)
      + (false ? cost_knight : cost_knight) * __builtin_popcountll(my_knights)
      + (false ? cost_bishop + 30 : cost_bishop) * __builtin_popcountll(my_bishops)
      + (false ? cost_rook : cost_rook) * __builtin_popcountll(my_rooks)
      + (false ? cost_queen + 50 : cost_queen) * __builtin_popcountll(my_queens);  // Q+=100 @0.1s 17059:17082 mb -0 Elo

  bool can_castle_left, can_castle_right;
  if (is_white2move) {
    can_castle_left = !king04_moved && !rook00_moved && !((my_something | his_something) & 0x0e);
    can_castle_right = !king04_moved && !rook07_moved && !((my_something | his_something) & 0x60);
  } else {
    can_castle_left = !king74_moved && !rook70_moved && !((my_something | his_something) & 0x0eull<<56);
    can_castle_right = !king74_moved && !rook77_moved && !((my_something | his_something) & 0x60ull<<56);
  }

  int score = material + 75 * (__builtin_popcountll(my_bishops) >= 2) + 0 * (12 * can_castle_left)/*@2s 24832:25261 mb -3 Elo*/, mg_score = 0, eg_score = 0;  // 50 vs 75 @1s 16055.5:16115.5; much later @0.1s 35678:35779 mb -0 Elo, 100:75 @0.1s 53370.5:54389.5 -3 Elo, 63:75 @0.1s 75450.5:74250.5 +3 Elo

  score += (true ? 15 : 20) * __builtin_popcountll(my_something & center_mask);  // 15:20 @0.1s 8559.5:8555.5 mb +0 Elo, @1s 4737:4440 +11 Elo
  score += 10 * __builtin_popcountll(I_attack);  // 10 vs 20 @3s 432.5:345.5 +39 Elo, @30s 4082:3807 +12 Elo, 5 vs 10 @3s 2053:2257 -16 Elo
  uint64_t his_pawn_attacks = is_white2move ? his_pawns>>7 & not_left_column | his_pawns>>9 & not_right_column : his_pawns<<7 & not_right_column | his_pawns<<9 & not_left_column;

  // @3s 51335.5:50348.5 +3 Elo, but of course was hoped to give like 15-20
  score -= 10 * __builtin_popcountll(I_attack & his_pawn_attacks);
  int knight_mobility = 0;
  for (uint64_t knights = my_knights; knights; knights &= knights - 1) {
    knight_mobility += __builtin_popcountll(knight_masks[__builtin_ctzll(knights)] & ~his_pawn_attacks);
  }
  score += 4 * knight_mobility;  // 2 vs 4 @3s 9846.5:10284.5 -8 Elo

  score -= 10 * __builtin_popcountll(my_pawns & my_pawns<<8);

  // @2.5s 81625:80899 mb +2 Elo p=.964 continued testing below, @30s 1932:1936 mb -0 Elo
  uint64_t my_pawn_attacks = is_white2move ? my_pawns<<7 & not_right_column | my_pawns<<9 & not_left_column : my_pawns>>7 & not_left_column | my_pawns>>9 & not_right_column;
  uint64_t his_half = is_white2move ? 0xffffffff00000000 : 0x00000000ffffffff;
  //score += 20 * __builtin_popcountll(my_knights & my_pawn_attacks & his_half & ~his_pawn_attacks);  // @2s 22761:22666 mb +1 Elo
  score += 5 * __builtin_popcountll(my_pawn_attacks & my_pawns);  // 10:5 @3s 2265.5:2328.5 mb -5 Elo p_>=.176

  {  // @0.1s 7034:6676 +9 Elo; @1s 13939.5:13426.5 +7 Elo
    score += 120 * __builtin_popcountll(my_pawn_attacks & his_queens);  // tests wo nn: 30:0 @0.1s 10056:9619 +8 Elo, 60:30 @0.1s 17339.5:16762.5 +6 Elo, 120:60 @0.1s 27825:27100 +5 Elo, 240:120 @0.1s 13930:13921 mb +0 Elo
    score += 60 * __builtin_popcountll(my_pawn_attacks & his_rooks);  // BEING TESTED, some paused
  }

  if (true) {
    score += 20 * bool(I_attack & my_kings);  // 30 vs 20 @3s 4149.5:3998 mb +6 Elo, @30s 4014.5:4078.5 mb -3 Elo, my:his @0.1s 68676:69824 -3 Elo
    score += 50 * bool(I_perforate & my_kings);  // mystery: 30:50 @0.1s 587.5:487.5 +32 Elo, @1s 6941.5:6582.5 +9 Elo, @30s 220.5:247.5 mb -20 Elo p_>=.106
  // later 30:50 @1s 22998:22581 mb +3 Elo p=0.974 @10s 3350:3326 mb +1 Elo p=0.615; remove everything :20+30@10s 1121:1105 mb +2 Elo p=.632 @200s 159:160 mb -1 Elo p=0.477
  }
  score += 40 * bool(I_perforate & his_kings);
  // 20:0 @.1s 18509:17917 +6 Elo, @1s 6347.5:6002.5 +10 Elo; 40:20 @.1s 27569.5:26845.5 +5 Elo @2s 11396:10934 +7 Elo @20s 4071.5:3908.5 mb +7 Elo p=0.965

  //score += 3 * bool(I_perforate & his_kings);  // 50 @0.1s 390.5:484.5 -37 Elo, 20 @0.1s 4883.5:5195.5 -11 Elo, 3 @0.1s 211051:213062 -2 Elo, 5 @0.1s 33482:33815 mb -2 Elo, 10 @0.1s 33482:33815 mb -2 Elo
  //score += 10 * bool(I_perforate & his_queens);
  //score -= 20 * (false && material == material0_of_player && my_queen_sq != 4 && my_queen_sq != 60);  // @5s 21962.5:22282.5 mb -2 Elo p_>=.06
  //int q = __builtin_popcountll(I_perforate & his_king_nhood);
  //score += 5*q + q*q/4;  // 5q+q²/4 vs 10q @3s 12029.5:11555.5 +7 Elo, @30s 4122:3846 +12 Elo
  //score += 1 * q;  // 1:0 @0.1s 79321.5:80546.5 -3 Elo, 2:0 @0.1s 18482.5:19082.5 -6 Elo, 3:0 @0.1s 2779.5:3010.5 -14 Elo, 5:0 @0.1s 986.5:1129.5 -24 Elo, 10:0 @0.1s 2356.5:2666.5 -21 Elo, 20:0 @0.1s 150.5:208.5 -57 Elo

  if (false) {  // @3s 16978.5:16971.5 mb +0 Elo, @30s 710.5:714.5 mb -1 Elo; probably due to cost
    for (uint64_t knights = his_knights; knights; knights &= knights - 1) {
      uint64_t can_move = knight_masks[__builtin_ctzll(knights)] & ~his_something;
      score += 3 * __builtin_popcountll(I_attack & can_move);
      if ((can_move & ~I_attack) == 0) {
        score += 30;
      }
    }
  }
  score += 5 * __builtin_popcountll(I_perforate & his_something);
  score += 3 * (bool(my_bishops & white_cells) * __builtin_popcountll(his_something & white_cells) + bool(my_bishops & black_cells) * __builtin_popcountll(his_something & black_cells));  // will try

  if (false) {
    #pragma unroll(8)
    for (uint64_t file : files) {
      if ((my_rooks & file) && !(my_pawns & file)) {
        score += 15 + 15 * !(his_pawns & file);
      }
    }
  }
  if (false) {
    for (uint64_t rooks = my_rooks; rooks; rooks &= rooks - 1) {
      uint64_t file = files[__builtin_ctzll(rooks) & 7];
      if (!(my_pawns & file)) {
        mg_score += 15 + 15 * !(his_pawns & file);
      }
    }
  }
  if (false) {
    // @30s 11138:11515 mb -6 Elo p_>=0.006
    #pragma unroll(8)
    for (uint64_t file : files) {
      uint64_t adj_files = (file & not_left_column) >> 1 | (file & not_right_column) << 1;
      if ((my_pawns & file) && !(my_pawns & adj_files)) {
        score -= 15;
      }
    }
  }
  if (false) {  // for both 40 1461.5:1462.5 mb +0 Elo
    if (my_rooks & (is_white2move ? row7 : row2)) {
      mg_score += 30;//, eg_score += 40;
    }
  }

  // @3s 328:252 +46 Elo, @30s 175:121 +64 Elo
  bool lost_rights = is_white2move ? king04_moved && (!false||rook00_moved&&rook07_moved) : king74_moved && (!false||rook70_moved&&rook77_moved);  // this change @3s 3277:3534 -13 Elo, @30s 121.5:121.5 mb -0 Elo
  bool is_tucked = my_king_j == 1 || my_king_j == 2 || my_king_j == 6;

  if (lost_rights && !is_tucked) {
    mg_score -= 80;
  } else if (is_tucked) {
    int pawn_shield = __builtin_popcountll(my_pawns & my_king_nhood);
    mg_score += 35 * pawn_shield;  // 25 vs 15 @3s 1896:1711 +18 Elo, @30s 2064.5:1870.5 +17 Elo, 35 vs 25 @3s 3418:3166 +13 Elo, @22s 13407.5:12905.5 +7 Elo
  }

  // @2s 2788:2559 +15 Elo, @30s 1427:1266 +21 Elo
  if (is_white2move) {
    mg_score -= 20 * bool(my_knights & 0x02);
    mg_score -= 20 * bool(my_knights & 0x40);
    mg_score -= 20 * bool(my_bishops & 0x04);
    mg_score -= 20 * bool(my_bishops & 0x20);
  } else {
    mg_score -= 20 * bool(my_knights & 0x02ull<<56);
    mg_score -= 20 * bool(my_knights & 0x40ull<<56);
    mg_score -= 20 * bool(my_bishops & 0x04ull<<56);
    mg_score -= 20 * bool(my_bishops & 0x20ull<<56);
  }

  //for (uint64_t rooks = my_rooks; rooks; rooks &= rooks - 1) {}

  // this block @30s 1429.5:1268.5 +21 Elo
  int flip = is_white2move ? 0 : 56;
  static const int p_mg[8]  = { 0,  0,  0,  0, 20, 50, 150, 0},  // make uint8_t @0.1s 44240.5:44350.5 mb -0 Elo
                   p_eg[8]  = {10, 10, 10, 10, 30, 60, 160, 0},
                   pp_mg[8] = { 0,  5, 10, 20, 35, 60, 100, 0},
                   pp_eg[8] = { 0, 10, 20, 40, 75,150, 250, 0};
  // (1,0)vs(0,0): @30s 5403.5:5086.5 +11 Elo
  // (0,1)vs(0,0): @30s 92:54 +93 Elo, also @30s 2649:2428 +15 Elo
  // (1,1)vs(1,0): @30s 1295.5:1458.5 -21 Elo
  for (uint64_t pawns = my_pawns; pawns; pawns &= pawns - 1) {
    int pawn = __builtin_ctzll(pawns), rank = (pawn ^ flip) >> 3;
    mg_score += p_mg[rank] >> 1, eg_score += p_eg[rank] >> 1;
    if (!(board::passed_pawn_check[side2move][pawn] & his_pawns)) {
      mg_score += pp_mg[rank] >> 0, eg_score += pp_eg[rank] >> 0;
    }
  }

  score += pst_summands<use_new>(game_phase);  // was once divided by 3, 2 vs 1 @3s 1678.5:1503.5 +19 Elo, @30s 69.5:37.5 +107 Elo, 3 vs 2 @30s 110.5:68.5 +83 Elo, 4 vs 3 456.5:459.5 mb -1 Elo
  return score + mg_score + ((eg_score-mg_score) *game_phase /256);
}

template<bool use_new>
int board::hc_heuristic_score(const nnue_state_t<use_new>& nnue_state) const {
  static constexpr int weight_pawn = 0, weight_knight = 1, weight_bishop = 1, weight_rook = 3, weight_queen = 5,
                   initial_total_weight = 16 * weight_pawn + 4 * weight_knight + 4 * weight_bishop + 4 * weight_rook + 2 * weight_queen;
  int current_total_weight = (__builtin_popcountll(white_pawns | black_pawns)) * weight_pawn  // vs __builtin_popcountll(a)+__builtin_popcountll(b) @0.1s 15231:14691 +6 Elo
                           + (__builtin_popcountll(white_knights | black_knights)) * weight_knight
                           + (__builtin_popcountll(white_bishops | black_bishops)) * weight_bishop
                           + (__builtin_popcountll(white_rooks | black_rooks)) * weight_rook
                           + (__builtin_popcountll(white_queens | black_queens)) * weight_queen;
  int game_phase = clamp((initial_total_weight-current_total_weight << 8) / initial_total_weight, 0, 256);  // making this <<'y and new weights and score→eg_score for pawn reward and fixing std:: resulted in 356820:356022 mb +0 Elo
  board b2(*this);
  b2.toggle_identity();
  int score = one_sided_score<use_new>(game_phase) - b2.one_sided_score<use_new>(game_phase);
  //int exchanges = (score>0?+1:-1) * 30 * (initial_total_weight - current_total_weight) / initial_total_weight;  // @0.2s 21558.5:20920.5 +5 Elo, @2s 3992:4051 mb -3 Elo, 200:100 @0.1s 54417.5:53404.5 +3 Elo
  // exchanges 200 @0.1s 3487.5:3753.5 -13 Elo, @1s 833:963 -25 Elo
  int stretch = 256 + game_phase/4;  // @.1s 28998:29332 mb -2 Elo p=0.084 @1s 4001:3936 mb +3 Elo p=0.770

  //return score;
  return use_new ? (nn_eval<use_new>(*this, nnue_state) * 7 + score) / 4 : (nn_eval<use_new>(*this, nnue_state) * 7 + score) / 4;  // (when nn v5) score:avg2 @.3s 100.5:205.5 -124 Elo, @3s 84.5:217.5 -164 Elo, nn:questo @.3s 108:204 -110 Elo, @3s 113.5:191.5 -91 Elo
  // /4:/2 @10s 5249:4937 +11 Elo
  // 3 1 /4 :7 1 /4 @0.1s 8914.5:9333.5 -8 Elo @1s 743.5:869.5 -27 Elo
  // 3 1 /2 :7 1 /4 @0.1s 109509:110179 mb -1 Elo p=0.077 @1s 1991.5:2191.5 -17 Elo

  // /4:/2 @0.1s 970367.5:971517.5 mb -0 Elo p=0.204 @1s 4522.5:4234.5 +11 Elo
  // 75%:50% @0.1s 200.5:140.5 +62 Elo, @1s 184:120 +74 Elo, @10s 203:97 +128 Elo
  // (v9f9) nn trust 50%:75% @2s 110:190 -95 Elo, 87.5%:75% @.2s 7551.5:7173.5 +9 Elo @2s 932:803 +26 Elo @20s 694.5:583.5 +30 Elo
  // (7n+1s)/4:(7n+1s)/8 @.2s 1702.5:1523.5 +19 Elo @2s 3553:3298 +13 Elo; (7n+1s)/2:(7n+1s)/4 @.1s 2101:2309 -16 Elo; (3n+1s)/2:(7n+1s)/4 @2s 1216:1375 -21 Elo
  // (3n+1s)/1:(7n+1s)/2 @.2s 10199:9764 +8 Elo @2s 4584:4885 -11 Elo @20s 221:291 -48 Elo
  // (15n+1s)/4:(7n+1s)/2  @.2s 11338.5:11810.5 -7 Elo
  // oh shoot this^ was together with smth else
  // (15n+1s)/4:(7n+1s)/2 @.2s 2253:2467 -16 Elo @1s 23074:23061 mb +0 Elo p=0.524 @10s 7681:7386 mb +7 Elo p=0.991 @100s 713:651 mb +16 Elo p=0.953
  // ([9f9]*7+[hc])/2:(15[9f9]+1[hc])/4 @1s 13024:13258 mb -3 Elo p=0.074 @10s 864:881 mb -3 Elo p=0.342
  // we are proceeding with 7, bc no diff, and 15 is more extreme


  //return use_new ? (nn_eval<use_new>(*this, nnue_state) + score) / 2 : score;
  //return use_new ? nn_eval(*this) : score;
}
  //return use_new ? score + nn_eval<use_new>(*this)/2 : (nn_eval<use_new>(*this) + score) / 2;


#endif
