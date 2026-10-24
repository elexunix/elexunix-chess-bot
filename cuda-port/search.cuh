#ifndef INCLUDE_SEARCH
#define INCLUDE_SEARCH
#include <cassert>
#include <utility>
#include <vector>
#include "board.cuh"


const int inf = 2e4;//unk = 30247;
const int max_depth = 50;


double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    std::cout << "SYSCALL FAILED" << std::endl;
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

struct KillerMoves {
  move moves[max_depth];
};

struct tt_entry {
  uint64_t hash;
  int depth = -1, lb = -inf, ub = +inf;
  move bestmove;
};

template<typename T, typename U>
struct pair {
  T first;
  U second;
};

constexpr int max_tt_size = 1024;
template<bool use_new>
class TT {  // @.3s 187:118 +80 Elo
  tt_entry data[max_tt_size];
  int size;
  uint64_t size_mask;  // like 0x0000ffff if size=2^16
public:
  __host__ __device__ TT(int size) : size(size) {
    assert(size == max_tt_size);
    //assert(size < 3e9);
    for (size_mask = 1; size_mask < size; size_mask *= 2);
    assert(size == size_mask);
    size_mask--;
  }
  __host__ __device__ pair<const tt_entry, bool> find(const board& b) {
    uint64_t hash = b.get_hash();
    const tt_entry& entry = data[hash & size_mask];
    return {entry, entry.hash == hash};
  }
  __host__ __device__ void inform(const board& b, int depth, int lb, int ub, move bestmove) {
    uint64_t hash = b.get_hash();
    tt_entry& entry = data[hash & size_mask];
    if (entry.depth > depth) {
      return;
    }
    if (entry.hash != hash || entry.depth < depth) {
      entry = {hash, depth, lb, ub, bestmove};
    } else {
      if (lb > entry.lb) {
        entry.lb = lb, entry.bestmove = bestmove;
      }
      entry.ub = min(entry.ub, ub);
    }
  }
  void clear() {
    size = 0;
  }
};


template<bool use_new>
__device__ int quiescence(const board& b, nnue_state_t<use_new>& nnue_state, int alpha, int beta, int max_diving_depth) {  // this guy is eager to terminate
  __attribute__((assume(alpha < beta)));
  const int mate = 9e3 * pow(1.01, max_diving_depth);
  if (b.can_eat_king()) {
    return mate;  // the previous player left his king under check, we won
  }
  int standing_pat = b.hc_heuristic_score<use_new>(nnue_state);
  if (max_diving_depth <= 0) {
    return standing_pat;
  }
  const bool in_check = b.is_check(), quick = !in_check;  // @30s 80:45 +100 Elo
  int best_score = -inf;  // only to detect checkmates
  if (quick) {
    alpha = max(alpha, standing_pat);
  } else {
    best_score = -mate;
  }
  if (alpha >= beta) {
    return beta;
  }
  for (move m: (in_check ? b.list_possible_moves<board::desire_type_t::vector_of_moves, false>() : b.list_possible_moves<board::desire_type_t::vector_of_moves, true>())) {
    board b2(b);
    int value;
    if (b2.make_move(m))
      value = mate;
    else
      nnue_state.update(b, b2), value = -quiescence<use_new>(b2, nnue_state, -beta, -alpha, max_diving_depth - 1), nnue_state.update(b2, b);
    if (value >= beta) {
      return beta;
    }
    if (value > best_score) {
      best_score = value;
    }
    if (value > alpha) {
      alpha = value;
    }
  }
  return in_check && best_score <= -9e3 ? best_score : alpha;
}


template<bool use_new, bool is_outer, int split_depth>
__device__ int compute_alphabeta_2(const board& b, move& put_bestmove, int depth, int alpha, int beta, curandState_t& curand_state, hashes_vec_t& hash_history, TT<use_new>& ttable, KillerMoves& killer_moves, nnue_state_t<use_new> nnue_state) {
  const int alpha0 = alpha;
  const bool tt_interaction = false;//!is_outer && depth >= 1;  // @3s 309.0:211.0 +66 Elo, 1 vs 2 @3s 99.5:149.5 -71 Elo, 3 vs 2 @3s 138:90 +74 Elo, 3 vs none @3s 67:35 +113 Elo, @30s 84.5:15.5 +295 Elo; MUCH later 3 vs 2 @3s 89:52 +93 Elo, @30s 462.5:371.5 +38 Elo, 1 vs 2 @3s 140.5:91.5 +75 Elo @30s 421:336 +39 Elo
  const int mate = 1e4 * pow(1.01, depth);
//  if (!is_outer && depth >= 2 && --left_nodes <= 0) {
//    return unk;
//  }

  if (depth >= 2 && b.is_stalemate()) {  // @10s 1023:887 +25 Elo
    return clamp(0, alpha, beta);
  }

  const bool maintain_hashes = depth >= 3;
  if (maintain_hashes) {
    if (!is_outer && hash_history.size() >= 10) {
      //#pragma unroll(10)
      // start from start @30s 376:466 -37 Elo, if also d≥4 @30s 287:366 -42 Elo, d≥5 93.5:123.5 mb -48 Elo
      for (auto it = false ? hash_history.begin() + hash_history.size() % 2 : hash_history.end() - 10; it < hash_history.end(); it += 2) {
        if (*it == b.get_hash()) {
          return 0;
        }
      }
    }
  }
  /*if (!is_outer && depth >= 4 && (put_bestmove = b.get_unique_legal_move()) != move()) {
    move useless;
    if (maintain_hashes) {
      hash_history.push_back(b.get_hash());
    }
    board b2(b);
    int value = b2.make_move(put_bestmove) ? mate : -compute_alphabeta_2<use_new, is_outer, split_depth>(b2, useless, depth - 1, -beta, -alpha, curand_state, hash_history, ttable, killer_moves);
    if (maintain_hashes) hash_history.pop_back();
    return value;
  }*/
  if (depth <= 0) {
    // quiescence vs just eval @30s 72.5:27.5 +168 Elo
    return quiescence<use_new>(b, nnue_state, alpha, beta, 4);  // 4 vs 2 @30s 18:82 -263 Elo; much later&after qs bugfix @3s 345:267 +45 Elo, @30s 79:44 +102 Elo; 6 vs 4 @3s 142:198 -58 Elo, @30s 36:68 -110 Elo
  }
  if (tt_interaction) {
    auto [entry, found] = ttable.find(b);
    if (found && entry.depth >= depth && (entry.depth - depth) % 1 == 0) {  // previously important (≥100 Elo): the parity condition, bc the search is still very odd-even unstable; much later removing it @3s 1081:942 +24 Elo, @30s 2180:1980 +17 Elo
      if (entry.lb == entry.ub) {
        put_bestmove = entry.bestmove;
        return entry.lb;
      }
      if (entry.lb >= beta) {
        put_bestmove = entry.bestmove;
        return beta;
      }
      if (entry.ub <= alpha) {
        put_bestmove = entry.bestmove;
        return alpha;
      }
      alpha = max(alpha, entry.lb);
    }
  }
  __attribute__((assume(alpha < beta)));
  if (maintain_hashes)
    hash_history.push_back(b.get_hash());

  // d≥2 vs d≥3 @3s 1063.5:925.5 +24 Elo, @25s 561.5:462.5 +34 Elo
  const bool can_do_nmp = !is_outer && depth >= 2 && !b.is_check() && b.has_sufficient_material_for_nmp() /*some protection against zugzwang*/ ;//&& (false ? b.hc_heuristic_score<use_new>() >= beta : true); // enabling first is @1.5s 11870:11872 +0 Elo, later @3s 2190.5:2186.5 mb +0 Elo
  if (can_do_nmp) {  // without pop_back: @3s 649:766 -29 Elo, @30s 33:67 -123 Elo; correct: @3s 13866.5:13355.5 +7 Elo, @30s 185:175 mb +10 Elo
    const int R = 2;  // increasing reduces Elo
    board b2(b);
    b2.make_null_move();
    move useless;
    int null_score = -compute_alphabeta_2<use_new, false, split_depth>(b2, useless, depth - 1 - R, -beta, -beta + 1, curand_state, hash_history, ttable, killer_moves, nnue_state);
//    if (null_score == -unk) {
//      if (maintain_hashes) hash_history.pop_back();
//      return unk;
//    }
    if (null_score >= beta) {
      if (maintain_hashes)
        hash_history.pop_back();
      return !false || null_score >= 9000 ? beta : null_score;  // meno fiducia se restituisce "mate", @1.5s 13532.5:13438.5 mb +1 Elo, later @3s 1332:1339 mb -1 Elo; @3s 1255.5:1292.5 mb -5 Elo; BEING TESTED
    }
  }

  moves_vec_t possible_moves(b.list_possible_moves<board::desire_type_t::vector_of_moves>());
  if (is_outer) {
    moves_vec_t valid_moves;
    for (move m : possible_moves) {
      board next = b;
      (bool)next.make_move(m);
      if (!next.can_eat_king()) {  // Se dopo la mia mossa l'avversario non può mangiare il mio re, la mossa è valide
        valid_moves.push_back(m);
      }
    }
    possible_moves = valid_moves;
  }
  if (possible_moves.empty()) {
    if (maintain_hashes)
      hash_history.pop_back();
    return b.is_check() ? -mate : 0;
  }
  /*for (move m : possible_moves) {
    assert(m.i1 >= 0 && m.i1 < 8 && m.j1 >= 0 && m.j1 < 8);
    assert(m.i0 >= 0 && m.i0 < 8 && m.j0 >= 0 && m.j0 < 8);
  }*/
  /*if (false && depth >= 4) {  // d≥1 @30s 188:238 mb -41 Elo, d≥2 @30s 941.5:1080.5 -24 Elo; later d≥4 MAYBE 130110:129412 mb +1 Elo but sus
    uint64_t my_pawns     = b.side2move == WHITE ? b.white_pawns   : b.black_pawns;
    uint64_t my_knights   = b.side2move == WHITE ? b.white_knights : b.black_knights;
    uint64_t my_bishops   = b.side2move == WHITE ? b.white_bishops : b.black_bishops;
    uint64_t my_rooks     = b.side2move == WHITE ? b.white_rooks   : b.black_rooks;
    uint64_t my_queens    = b.side2move == WHITE ? b.white_queens  : b.black_queens;
    uint64_t my_kings     = 1ull << (b.side2move == WHITE ? b.white_king_i<<3|b.white_king_j : b.black_king_i<<3|b.black_king_j);
    uint64_t his_pawns    = b.side2move == WHITE ? b.black_pawns   : b.white_pawns;
    uint64_t his_knights  = b.side2move == WHITE ? b.black_knights : b.white_knights;
    uint64_t his_bishops  = b.side2move == WHITE ? b.black_bishops : b.white_bishops;
    uint64_t his_rooks    = b.side2move == WHITE ? b.black_rooks   : b.white_rooks;
    uint64_t his_queens   = b.side2move == WHITE ? b.black_queens : b.white_queens;
    uint64_t his_kings    = 1ull << (b.side2move == WHITE ? b.black_king_i<<3|b.black_king_j : b.white_king_i<<3|b.white_king_j);
    uint64_t his_something = his_pawns | his_knights | his_bishops | his_rooks | his_queens | his_kings;
    std::sort(possible_moves.begin(), possible_moves.end(), [&](const move& m1, const move& m2) -> bool {
      uint64_t target1 = 1ull << (m1.i1<<3|m1.j1), target2 = 1ull << (m2.i1<<3|m2.j1);
      if (!bool(target1 & his_something) && !bool(target2 & his_something)) {
        return false;
      } else if (!bool(target1 & his_something) || !bool(target2 & his_something)) {
        return bool(target1 & his_something);
      }
      if (bool(target1 & his_kings) ^ bool(target2 & his_kings)) {
        return target1 & his_kings;
      } else if (bool(target1 & his_queens) ^ bool(target2 & his_queens)) {
        return target1 & his_queens;
      } else if (bool(target1 & his_rooks) ^ bool(target2 & his_rooks)) {
        return target1 & his_rooks;
      } else if (bool(target1 & his_bishops) ^ bool(target2 & his_bishops)) {
        return target1 & his_bishops;
      } else if (bool(target1 & his_knights) ^ bool(target2 & his_knights)) {
        return target1 & his_knights;
      } else if (bool(target1 & his_pawns) ^ bool(target2 & his_pawns)) {
        return target1 & his_pawns;
      } else {
        uint64_t source1 = 1ull << (m1.i0<<3|m1.j0), source2 = 1ull << (m2.i0<<3|m2.j0);
        if (bool(source1 & my_pawns) ^ bool(source2 & my_pawns)) {
         return source1 & my_pawns;
        } else if (bool(source1 & my_knights) ^ bool(source2 & my_knights)) {
          return source1 & my_knights;
        } else if (bool(source1 & my_bishops) ^ bool(source2 & my_bishops)) {
          return source1 & my_bishops;
        } else if (bool(source1 & my_rooks) ^ bool(source2 & my_rooks)) {
          return source1 & my_rooks;
        } else if (bool(source1 & my_queens) ^ bool(source2 & my_queens)) {
          return source1 & my_queens;
        } else if (bool(source1 & my_kings) ^ bool(source2 & my_kings)) {
          return source1 & my_kings;
        } else {
         return false;
        }
      }
    });
  }*/

  move best_move_guess;
  if (tt_interaction) {
    auto [entry, found] = ttable.find(b);
    if (found) {
      best_move_guess = entry.bestmove;
    }
  }
  if (best_move_guess == move() && depth >= 3) {
    int eval = compute_alphabeta_2<use_new, false, 0>(b, best_move_guess, depth - 2, -inf, +inf, curand_state, hash_history, ttable, killer_moves, nnue_state);
//    if (eval == unk) {
//      if (maintain_hashes) hash_history.pop_back();
//      return unk;
//    }
  }

  int promoted_count = 0;
  if (best_move_guess != move()) {
    for (move& m : possible_moves) {
      if (m == best_move_guess) {
        cuda::std::swap(possible_moves[0], m);
        promoted_count = 1;
        break;
      }
    }
  }
  if (!is_outer && depth > 0 && promoted_count < possible_moves.size()) {  // @3s 2554:2337 +15 Elo
    move killer = killer_moves.moves[depth];
    if (killer != move()) {
      for (int i = promoted_count; i < possible_moves.size(); ++i) {
        if (possible_moves[i] == killer && !b.is_capture(possible_moves[i])) {
          cuda::std::swap(possible_moves[i], possible_moves[promoted_count++]);
          break;
        }
      }
    }
  }

  move best_move_found;

  // process the first move sequentially to establish a good alpha bound
  {
    const auto& m = possible_moves[0];
    board tmp(b);
    move useless;
    int value;
    if (tmp.make_move(m))
      value = mate;
    else
      nnue_state.update(b, tmp), value = -compute_alphabeta_2<use_new, false, split_depth>(tmp, useless, depth - 1, -beta, -alpha, curand_state, hash_history, ttable, killer_moves, nnue_state), nnue_state.update(tmp, b);
//    if (value == -unk) {
//      if (maintain_hashes) hash_history.pop_back();
//      return unk;
//    }

    best_move_found = m;
    if (value >= beta) {
      if (maintain_hashes)
        hash_history.pop_back();
      if (tt_interaction)
        ttable.inform(b, depth, value, +inf, best_move_found);
      if (!is_outer && depth > 0 && !b.is_capture(best_move_found))
        killer_moves.moves[depth] = best_move_found;
      put_bestmove = best_move_found;
      return beta;
    }
    if (value > alpha)
      alpha = value;
    else if (is_outer && value == alpha)
      best_move_found = m;
  }
  int cnt_best = 1;

  auto search_child_node = [&b, &ttable, depth, beta, mate, maintain_hashes, &hash_history, &nnue_state](const move &m, int alpha, curandState_t& curand_state, int move_i, hashes_vec_t &local_hash_history, KillerMoves &killer_moves) -> int {
    alpha -= is_outer;  // in case of outer, we really care if it's exactly α or less
    constexpr bool use_pvs = false;  // idk why but PVS is -10 Elo. I was unable to find the bug. maybe the move ordering is not ripe yet -- let's focus on implementing other important things for now; much later: @.4s 40k:40k mb +0 Elo
    const bool use_lmr = move_i >= 2; // @.3s 273:34 +362 Elo, @3s 68:34 +120 Elo, @30s 80:45 +100 Elo, 4 vs 2 @30s 79:123 -77 Elo; much later 3 vs 2 @.3s 23553:23826 mb -2 Elo, @3s 897:905 mb -2 Elo, 1 vs 2 @.3s 6314:6668 -9 Elo, @3s 2647:2878 -15 Elo
    board tmp(b);
    move useless;
    int value = tmp.make_move(m) ? mate : 0;
    if (value == 0) {
      nnue_state.update(b, tmp);
      constexpr int new_split_depth = split_depth - 1 >= 0 ? split_depth - 1 : 0;
      /*if constexpr (use_pvs) {  // PVS
        value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1, -alpha - 1, -alpha, local_gen, local_hash_history, ttable, killer_moves);
        if (value > alpha && value < beta) {
          value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1, -beta, -alpha, local_gen, local_hash_history, ttable, killer_moves);
        }
      } else {
        value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1, -beta, -alpha, local_gen, local_hash_history, ttable, killer_moves);
      }*/
      if (use_lmr) {  // LMR
        const int reduction = 2;  // 2 vs 4 @.3s 4301:4019 +12 Elo, 4 vs 2 @3s 41.5:73.5 -99 Elo, 4 vs 2 @30s 37.5:69.5 -107 Elo; later 3 vs 2 @.3s 1101.5:1250.5 -22 Elo, @3s 565.5:676.5 -31 Elo, 1 vs 2 @.3s 1014.5:1162.5 -24 Elo, 4 vs 2 @.3s 84.5:132.5 -78 Elo, @3s 41.5:75.5 -104 Elo
        value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1 - reduction, -beta, -alpha, curand_state, local_hash_history, ttable, killer_moves, nnue_state);
//        if (value == -unk) {
//          if (maintain_hashes) hash_history.pop_back();
//          return unk;
//        }
        if (value > alpha) {
          value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1, -beta, -alpha, curand_state, local_hash_history, ttable, killer_moves, nnue_state);
        }
      } else {
        value = -compute_alphabeta_2<use_new, false, new_split_depth>(tmp, useless, depth - 1, -beta, -alpha, curand_state, local_hash_history, ttable, killer_moves, nnue_state);
      }
//      if (value == -unk) {
//        if (maintain_hashes) hash_history.pop_back();
//        return unk;
//      }
      nnue_state.update(tmp, b);
    }
    return value;
  };

  // process remaining moves
  {
    // sequential execution for remaining moves
    for (size_t i = 1; i < possible_moves.size(); ++i) {
      const auto& m = possible_moves[i];
      int value = search_child_node(m, alpha, curand_state, i, hash_history, killer_moves);
//      if (value == unk) {
//        if (maintain_hashes) hash_history.pop_back();
//        return unk;
//      }
      if (value > alpha) {
        alpha = value, best_move_found = m, cnt_best = 1;
        if (is_outer) { cnt_best = 1; }
      } else if (is_outer && value == alpha && curand(&curand_state) % ++cnt_best == 0) {
        best_move_found = m;
      }
      if (alpha >= beta) {
        if (maintain_hashes) { hash_history.pop_back(); }
        if (tt_interaction) { ttable.inform(b, depth, alpha, +inf, best_move_found); }
        if (!is_outer && depth > 0 && !b.is_capture(best_move_found)) {
          killer_moves.moves[depth] = best_move_found;
        }
        put_bestmove = best_move_found;
        return beta;
      }
    }
  }

  put_bestmove = best_move_found;
  if (maintain_hashes) hash_history.pop_back();
  if (tt_interaction) {
    int lower_bound = (alpha > alpha0) ? alpha : -inf;
    ttable.inform(b, depth, lower_bound, alpha, best_move_found);
  }
  return alpha;
}

#endif
