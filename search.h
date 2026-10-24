#ifndef INCLUDE_SEARCH
#define INCLUDE_SEARCH
#include <cassert>
#include <mutex>
#include <utility>
#include <vector>
#include "board.h"


const int inf = 2e4;//unk = 30247;
const int max_depth = 50;
constexpr int unk = -23947;

int16_t mate_values_s[max_depth], mate_values_qs[max_depth];
void precalc_search() {
  for (int d = 0; d < max_depth; ++d)
    mate_values_s[d] = 1e4 * pow(1.01, d);
  for (int d = 0; d < max_depth; ++d)
    mate_values_qs[d] = 9e3 * pow(1.01, d);
}


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

/*template<bool use_new>
struct tt_entry {
  uint64_t hash;
  int depth = -1, lb = -inf, ub = +inf;
  move bestmove;
};

template<bool use_new>
class TT {  // @.3s 187:118 +80 Elo
  std::vector<tt_entry<use_new>> data;
  int size;
  uint64_t size_mask;  // like 0x0000ffff if size=2^16
  static constexpr int n_mutexes = 256;  // must be a power of two
  static constexpr uint64_t n_mutexes_mask = n_mutexes - 1;
  mutable std::mutex m[n_mutexes];
public:
  static constexpr int tt_inf = 2e4;  // 3e4:2e4 @0.1s 7109:7125 mb -0 Elo p=.456, @1s @1s 8821:8891 mb -1 Elo p=.302 @10s 185.5:186.5 mb -1 Elo p=.479
  TT(int size) : size(size), data(size) {
    //assert(size < 3e9);
    for (size_mask = 1; size_mask < size; size_mask *= 2);
    assert(size == size_mask);
    size_mask--;
  }
  std::pair<const tt_entry<use_new>, bool> find(const board& b) {
    uint64_t hash = b.get_hash();
    std::scoped_lock sl{m[hash & n_mutexes_mask]};  // these four %->& @0.2s 299966:300414 mb -0 Elo
    const tt_entry<use_new>& entry = data[hash & size_mask];
    return {entry, entry.hash == hash};
  }
  void inform(const board& b, int depth, int lb, int ub, move bestmove) {
    uint64_t hash = b.get_hash();
    std::scoped_lock sl{m[hash & n_mutexes_mask]};
    tt_entry<use_new>& entry = data[hash & size_mask];
    if (entry.depth > depth)
      return;
    if (entry.hash != hash || entry.depth < depth) {
      entry = {hash, depth, lb, ub, bestmove};
    } else if (!use_new || entry.depth == depth) {
      if (false) {  // @.1s 18559.5:18482.5 mb +1 Elo p=.659, @1s 15217:15352 mb -2 Elo p=.214, @10s 3000.5:3055.5 mb -3 Elo p=.239
        if (lb > entry.ub || ub < entry.lb)
          entry.lb = lb, entry.ub = ub, entry.bestmove = entry.lb > -tt_inf ? bestmove : move();
        else {
          if (lb > entry.lb)
            entry.lb = lb, entry.bestmove = bestmove;
          if (ub < entry.ub)
            entry.ub = ub;
        }
      } else {
        if (lb > entry.lb)
          entry.lb = lb, entry.bestmove = bestmove;
        entry.ub = min(entry.ub, ub);
      }
    }
  }
  void clear() {
    for (int i = 0; i < n_mutexes; ++i)
      m[i].lock();
    data.assign(size, tt_entry<use_new>());
    for (int i = 0; i < n_mutexes; ++i)
      m[i].unlock();
  }
};*/

// lockless tt approach: EXTREME gains: (usual)1-threaded @.1s 464.5:324.5 +62 Elo @1s 399.5:207.5 +114 Elo @10s @10s 198.5:101.5 +117 Elo
//template<bool use_new>
//struct tt_entry {
//  // important: each field should fit in 1 register, so sizeof<=4 (but this class is sanitizer-safe even without that)
//  std::atomic<uint64_t> hash_xor_rest;
//  std::atomic<int> depth, lb, ub;
//  std::atomic<move> bestmove;
//  static_assert(sizeof(move) <= 8);
//  void store(uint64_t hash_, int depth_, int lb_, int ub_, move bestmove_) {
//    uint64_t metadata = uint64_t(depth_)<<48 ^ uint64_t(lb_)<<32 ^ uint64_t(ub_)<<16 ^ std::bit_cast<uint16_t>(bestmove_);
//    hash_xor_rest.store(hash_ ^ metadata, std::memory_order::relaxed);
//    depth.store(depth_);
//    lb.store(lb_);
//    ub.store(ub_);
//    bestmove.store(bestmove_);
//  }
//  bool probe(uint64_t hash_, int& put_depth, int& put_lb, int& put_ub, move& put_bestmove) {
//    uint64_t loaded_hash_xor_rest = hash_xor_rest.load(std::memory_order::relaxed);
//    int loaded_depth = depth.load(std::memory_order::relaxed), loaded_lb = lb.load(std::memory_order::relaxed), loaded_ub = ub.load(std::memory_order::relaxed);
//    move loaded_bestmove = bestmove.load(std::memory_order::relaxed);
//    if ((hash_ ^ uint64_t(loaded_depth)<<48 ^ uint64_t(loaded_lb)<<32 ^ uint64_t(loaded_ub)<<16 ^ std::bit_cast<uint16_t>(loaded_bestmove)) == loaded_hash_xor_rest) {  // then the entry makes sense, we are not reading mid-write
//      put_depth = loaded_depth, put_lb = loaded_lb, put_ub = loaded_ub, put_bestmove = loaded_bestmove;
//      return true;
//    }
//    return false;
//  }
//};
template<bool use_new>  // this implementation @.1s 26002:25296 +5 Elo
struct tt_entry {
  // important: each field should fit in 1 register, so sizeof<=4 (but this class is sanitizer-safe even without that)
  std::atomic<uint64_t> hash_xor_metadata;
  std::atomic<uint64_t> metadata;
  void store(uint64_t hash_, int depth_, int lb_, int ub_, move bestmove_) {
    uint64_t metadata_ = (uint64_t(depth_)&0xffff)<<48 ^ (uint64_t(lb_)&0xffff)<<32 ^ (uint64_t(ub_)&0xffff)<<16 ^ std::bit_cast<uint16_t>(bestmove_);
    hash_xor_metadata.store(hash_ ^ metadata_, std::memory_order::relaxed);
    metadata.store(metadata_, std::memory_order::relaxed);  // restored relaxed
  }
  bool probe(uint64_t hash_, int& put_depth, int& put_lb, int& put_ub, move& put_bestmove) {
    uint64_t loaded_hash_xor_metadata = hash_xor_metadata.load(std::memory_order::relaxed);
    uint64_t loaded_metadata = metadata.load(std::memory_order::relaxed);
    if ((hash_ ^ loaded_metadata) == loaded_hash_xor_metadata) {  // then the entry makes sense, we are not reading mid-write
      put_depth = int16_t(loaded_metadata>>48), put_lb = int16_t(loaded_metadata>>32), put_ub = int16_t(loaded_metadata>>16), put_bestmove = std::bit_cast<move>(uint16_t(loaded_metadata));
      return true;
    }
    return false;
  }
};  // w/o atomics (when single thread surprisingly +0 Elo)

template<bool use_new>
struct TT {
  std::vector<tt_entry<use_new>> data;
  int size;
  uint64_t size_mask;  // like 0x0000ffff if size=2^16
public:
  static constexpr int tt_inf = 2e4;  // 3e4:2e4 @0.1s 7109:7125 mb -0 Elo p=.456, @1s 8821:8891 mb -1 Elo p=.302 @10s 185.5:186.5 mb -1 Elo p=.479
  TT(int size) : size(size), data(size) {
    //assert(size < 3e9);
    for (size_mask = 1; size_mask < size; size_mask *= 2);
    assert(size == size_mask);
    size_mask--;
  }
  bool find(const board& b, int& put_depth, int& put_lb, int& put_ub, move& put_bestmove) {
    uint64_t hash = b.get_hash();
    return data[hash & size_mask].probe(hash, put_depth, put_lb, put_ub, put_bestmove);
  }
  void inform(const board& b, int depth, int lb, int ub, move bestmove) {
    uint64_t hash = b.get_hash();
    tt_entry<use_new>& entry = data[hash & size_mask];
    int entry_depth, entry_lb, entry_ub;
    move entry_bestmove;
    bool probe_succeeded = data[hash & size_mask].probe(hash, entry_depth, entry_lb, entry_ub, entry_bestmove);
    if (probe_succeeded && entry_depth > depth)
      return;
    if (!probe_succeeded || entry_depth < depth) {
      entry.store(hash, depth, lb, ub, bestmove);
    } else {
      if (lb > entry_lb)
        entry_lb = lb, entry_bestmove = bestmove;
      entry_ub = min(entry_ub, ub);
      entry.store(hash, entry_depth, entry_lb, entry_ub, entry_bestmove);
    }
  }
  void clear() {
    for (tt_entry<use_new>& entry : data)
      entry.store(0xdeadbeef, 0, 0, 0, move());  // will fail the reads
  }
};


// std::atomic<__uint128_t>::is_always_lock_free and std::atomic<__uint128_t>(0).is_lock_free() are falseon my cpu
//template<>  // fails (-500 Elo) bc 12 bit is too narrow to store inf
//struct tt_entry<true> {
//  std::atomic<uint64_t> packed;
//  void store(uint64_t hash_, int depth_, int lb_, int ub_, move bestmove_) {
//    uint64_t packed_ = uint64_t(uint16_t(hash_))<<48 | (uint64_t(depth_)&0xff)<<40 | (uint64_t(lb_)&0xfff)<<28 | (uint64_t(ub_)&0xfff)<<16 | std::bit_cast<uint16_t>(bestmove_);  // "uint48_t" actually; uint16_t<<48 would first cast to int
//    packed.store(packed_, std::memory_order::relaxed);
//  }
//  bool probe(uint64_t hash_, int& put_depth, int& put_lb, int& put_ub, move& put_bestmove) {
//    uint64_t loaded_packed = packed.load(std::memory_order::relaxed);
//    if (loaded_packed>>48 == uint16_t(hash_)) {
//      int raw_lb, raw_ub;
//      put_depth = int8_t(loaded_packed>>40), raw_lb = loaded_packed>>28&0xfff, raw_ub = loaded_packed>>16&0xfff, put_bestmove = std::bit_cast<move>(uint16_t(loaded_packed));
//      put_lb = raw_lb & 0x800 ? raw_lb | ~0xfff : raw_lb, put_ub = raw_ub & 0x800 ? raw_ub | ~0xfff : raw_ub;  // sign extend from 12 bits
//      return true;
//    }
//    return false;
//  }
//};
//template<>  // @.1s -19 Elo, probablement à cause de 8+21-bit collisions
//struct tt_entry<true> {
//  std::atomic<uint64_t> packed;
//  void store(uint64_t hash_, int depth_, int lb_, int ub_, move bestmove_) {
//    uint64_t packed_ = (hash_>>56&0xff)<<56 | (uint64_t(depth_)&0xff)<<48 | (uint64_t(lb_)&0xffff)<<32 | (uint64_t(ub_)&0xffff)<<16 | std::bit_cast<uint16_t>(bestmove_);
//    packed.store(packed_, std::memory_order::relaxed);
//  }
//  bool probe(uint64_t hash_, int& put_depth, int& put_lb, int& put_ub, move& put_bestmove) {
//    uint64_t loaded_packed = packed.load(std::memory_order::relaxed);
//    if (loaded_packed>>56 == hash_>>56) {
//      put_depth = int8_t(loaded_packed>>48), put_lb = int16_t(loaded_packed>>32&0xffff), put_ub = int16_t(loaded_packed>>16&0xffff), put_bestmove = std::bit_cast<move>(uint16_t(loaded_packed));
//      return true;
//    }
//    return false;
//  }
//};


template<bool use_new>
int quiescence(const board& b, nnue_state_t<use_new>& nnue_state, int alpha, int beta, int max_diving_depth, TT<use_new>& qttable) {  // this guy is eager to terminate
  __attribute__((assume(alpha < beta)));
  if (false && max_diving_depth < 0)
    max_diving_depth = 0;
  const int mate = false ? mate_values_qs[max_diving_depth] : 9e3 * pow(1.01, max_diving_depth);
  const bool qtt_interaction = max_diving_depth >= 2;  // 1:2 @.1s  17345:17256 mb +1 Elo, @1s 13066:13383 mb -4 Elo
  int alpha0 = alpha;
  if (b.can_eat_king())
    return mate;  // the previous player left his king under check, we won
  int standing_pat = b.hc_heuristic_score<use_new>(nnue_state);
  if (qtt_interaction) {  // @0.1s 186.5:125.5 +69 Elo, @1s 376.5:297.5 +41 Elo, @10s 795:676 +28 Elo
    bool found;
    int entry_depth, entry_lb, entry_ub;
    move dummy;
    if constexpr (true)
      found = qttable.find(b, entry_depth, entry_lb, entry_ub, dummy);
    else {
      auto [entry, found_] = qttable.find(b);
      if (found = found_)
        entry_depth = entry.depth, entry_lb = entry.lb, entry_ub = entry.ub;
    }
    //auto [entry, found] = qttable.find(b);
    if (found && entry_depth >= max_diving_depth) {
      if (entry_lb == entry_ub)
        return entry_lb;
      if (entry_lb >= beta)
        return beta;
      if (entry_ub <= alpha)
        return alpha;
      alpha = max(alpha, entry_lb);
    }
  }
  if (max_diving_depth <= 0)
    return standing_pat - 0;  // if is_check -=300:0 @.1s 10640:10774 mb -2 Elo p=0.179 @1s 1872:1920 mb -4 Elo p=0.217
  const bool in_check = b.is_check(), quick = !in_check;  // @30s 80:45 +100 Elo
  int best_score = -inf;  // only to detect checkmates
  if (quick)
    alpha = max(alpha, standing_pat);
  else
    best_score = -mate;
  if (alpha >= beta)
    return beta;
  std::vector<move> considered_moves = in_check ? b.list_possible_moves<board::desire_type_t::vector_of_moves, false>() : b.list_possible_moves<board::desire_type_t::vector_of_moves, true>();
  int cnt_considered_moves = considered_moves.size();
  const bool move_scores_active = max_diving_depth >= 2 && cnt_considered_moves > 0;  // 1:2 @.2s 7959.5:7568.5 +9 Elo
  // @0.06s 209:134 +77 Elo, @.2s 801.5:684.5 +27 Elo, @15s 606.5:503.5 +32 Elo
  if (move_scores_active) {  // after rewriting with arrs @.2s 11743:11272 +7 Elo
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
    int move_scores[cnt_considered_moves];
    for (int move_i = 0; move_i < cnt_considered_moves; ++move_i) {
      move m = considered_moves[move_i];
      uint64_t source = 1ull << (m.i0<<3|m.j0), target = 1ull << (m.i1<<3|m.j1);
      if (target & ~his_something)
        move_scores[move_i] = 0;
      else {
        int target_val = target & his_kings ? 6 : target & his_queens ? 5 : target & his_rooks ? 4 : target & his_bishops ? 3 : target & his_knights ? 2 : target & his_pawns ? 1 : 0,
            source_val = source & his_kings ? 6 : source & his_queens ? 5 : source & his_rooks ? 4 : source & his_bishops ? 3 : source & his_knights ? 2 : source & his_pawns ? 1 : 0;
        move_scores[move_i] = (target_val<<3 | 6-source_val)  // bitmasks here mb +0 Elo
          + (target_val >= source_val ? 60 : 0);  // @0.06s 185371.5:185177.5 mb +0 Elo p=0.630, pourquoi..
      }
    }
    int indices[cnt_considered_moves];
    std::pair<int, move> moves_[cnt_considered_moves];
    for (int move_i = 0; move_i < cnt_considered_moves; ++move_i) {
      move m = considered_moves[move_i];
      //if (m == best_move_guess)
      //  move_scores[move_i] = 200;
      indices[move_i] = move_i;
      moves_[move_i] = {-move_scores[move_i], m};
    }
    std::ranges::sort(moves_, moves_ + cnt_considered_moves);
    for (int move_i = 0; move_i < cnt_considered_moves; ++move_i)
      considered_moves[move_i ] = moves_[move_i].second;
  }
  for (move m : considered_moves) {
    board b2(b);
    int value;
    if (b2.make_move(m))
      value = mate;
    else
      nnue_state.update(b, b2), value = -quiescence<use_new>(b2, nnue_state, -beta, -alpha, max_diving_depth - 1, qttable), nnue_state.update(b2, b);
    if (value >= beta) {
      if (qtt_interaction)
        qttable.inform(b, max_diving_depth, value, +TT<use_new>::tt_inf, m);
      return beta;
    }
    if (value > best_score)
      best_score = value;
    if (value > alpha)
      alpha = value;
  }
  if (qtt_interaction)
    qttable.inform(b, max_diving_depth, alpha > alpha0 ? alpha : -TT<use_new>::tt_inf, alpha, move());
  return in_check && best_score <= -9e3 ? best_score : alpha;
}


template<bool use_new, bool is_outer, int split_depth, bool is_helper_thread>
int compute_alphabeta_2(const board& b, move& put_bestmove, int depth, int alpha, int beta, std::mt19937& gen, std::vector<uint64_t>& hash_history, TT<use_new>& ttable, TT<use_new>& qttable, KillerMoves& killer_moves, nnue_state_t<use_new>& nnue_state, const std::atomic<bool>& stop_search) {   // restore ref for nnue_state @.2s 22958:22294 +5 Elo
  if (depth >= 6 && stop_search)
    return unk;
  const int alpha0 = alpha;
  const bool tt_interaction = !is_outer && depth >= 0;  // @3s 309.0:211.0 +66 Elo, 1 vs 2 @3s 99.5:149.5 -71 Elo, 3 vs 2 @3s 138:90 +74 Elo, 3 vs none @3s 67:35 +113 Elo, @30s 84.5:15.5 +295 Elo; MUCH later 3 vs 2 @3s 89:52 +93 Elo, @30s 462.5:371.5 +38 Elo, 1 vs 2 @3s 140.5:91.5 +75 Elo @30s 421:336 +39 Elo; much later after big nnue @0.1s 544203.5:540981.5 +1 Elo, @2s 14857:14791 mb +1 Elo
  if (false && depth < 0)
    depth = 0;
  const int mate = false ? mate_values_s[depth] : 1e4 * pow(1.01, depth);
//  if (!is_outer && depth >= 2 && --left_nodes <= 0) {
//    return unk;
//  }
  if (!is_outer && depth >= 0 && b.can_eat_king())  // @0.1s 14587:14064 +6 Elo @1s 1552.5:1383.5 +20 Elo @10s 644:537 +32 Elo; 0:1 @.1s 40051.5:40348.5 mb -1 Elo p=0.137 @1s 11357.5:11130.5 mb +4 Elo p=0.934
    return mate;

  if (depth >= 2 && b.is_stalemate()) {  // @10s 1023:887 +25 Elo; 1:2 @0.1s 21581:22230 -5 Elo @1s 4203:4261 mb -2 Elo p=0.260
    return std::clamp(0, alpha, beta);
  }

  const bool maintain_hashes = depth >= (false ? 2 : 3);  // with old tt: 2:3 @.1s 66494:66368 mb +0 Elo, @1s 4297.5:4440.5 mb -6 Elo
  if (maintain_hashes) {
    if (!is_outer && hash_history.size() >= 10) {
      //#pragma unroll(10)
      // start from start @30s 376:466 -37 Elo, if also d≥4 @30s 287:366 -42 Elo, d≥5 93.5:123.5 mb -48 Elo
      for (auto it = false ? hash_history.begin() + hash_history.size() % 2 : hash_history.end() - 10; it < hash_history.end(); it += 2)  // d≥6⇒check from 0 @1s 4297.5:4440.5 mb -6 Elo p=.170
        if (*it == b.get_hash())
          return 0;
    }
  }
  if (depth <= 0)  // quiescence vs just eval @30s 72.5:27.5 +168 Elo
    return quiescence<use_new>(b, nnue_state, alpha, beta, 4, qttable);  // @.2s 50988:50564 mb +1 Elo p=0.908
  if (tt_interaction) {
    //auto [entry, found] = ttable.find(b);
    bool found;
    int entry_depth, entry_lb, entry_ub;
    move entry_bestmove;
    if constexpr (true)
      found = ttable.find(b, entry_depth, entry_lb, entry_ub, entry_bestmove);
    else {
      auto [entry, found_] = ttable.find(b);
      if (found = found_)
        entry_depth = entry.depth, entry_lb = entry.lb, entry_ub = entry.ub;
    }
    if (found && entry_depth >= depth) {  // previously important (≥100 Elo): the parity condition, bc the search is still very odd-even unstable; much later removing it @3s 1081:942 +24 Elo, @30s 2180:1980 +17 Elo
      if (entry_lb == entry_ub) {
        put_bestmove = entry_bestmove;
        return entry_lb;
      }
      if (entry_lb >= beta) {
        put_bestmove = entry_bestmove;
        return beta;
      }
      if (entry_ub <= alpha) {
        put_bestmove = entry_bestmove;
        return alpha;
      }
      alpha = max(alpha, entry_lb);
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
    b2.make_null_move<false>();
    move useless;
    int null_score = -compute_alphabeta_2<use_new, false, split_depth, is_helper_thread>(b2, useless, depth - 1 - R, -beta, -beta + 1, gen, hash_history, ttable, qttable, killer_moves, nnue_state, stop_search);
    if (null_score == -unk) {
      if (maintain_hashes) hash_history.pop_back();
      return unk;
    }
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
      if (!next.can_eat_king())  // Se dopo la mia mossa l'avversario non può mangiare il mio re, la mossa è valide
        valid_moves.push_back(m);
    }
    possible_moves = valid_moves;
  }
  if (possible_moves.empty()) {
    if (maintain_hashes)
      hash_history.pop_back();
    return b.is_check() ? -mate : 0;
  }
  int cnt_possible_moves = possible_moves.size(), move_scores[cnt_possible_moves];
  /*for (move m : possible_moves) {
    assert(m.i1 >= 0 && m.i1 < 8 && m.j1 >= 0 && m.j1 < 8);
    assert(m.i0 >= 0 && m.i0 < 8 && m.j0 >= 0 && m.j0 < 8);
  }*/
  const bool move_scores_active = depth >= 1 && cnt_possible_moves > 0;  // overall here + in main search @0.06s 1361:1205 +21 Elo, @.2s 552.5:453.5 +34 Elo, @2s 495.5:399.5 +37 Elo
  if (move_scores_active) {  // d≥1 @30s 188:238 mb -41 Elo, d≥2 @30s 941.5:1080.5 -24 Elo; later d≥4 MAYBE 130110:129412 mb +1 Elo but sus; much later after rewriting with arrs @.2s 11743:11272 +7 Elo
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
    /*std::sort(possible_moves.begin(), possible_moves.end(), [&](const move& m1, const move& m2) -> bool {
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
    });*/
    for (int move_i = 0; move_i < cnt_possible_moves; ++move_i) {
      move m = possible_moves[move_i];
      uint64_t source = 1ull << (m.i0<<3|m.j0), target = 1ull << (m.i1<<3|m.j1);
      if (target & ~his_something)
        move_scores[move_i] = 0;
      else {
        int target_val = target & his_kings ? 6 : target & his_queens ? 5 : target & his_rooks ? 4 : target & his_bishops ? 3 : target & his_knights ? 2 : target & his_pawns ? 1 : 0,
            source_val = source & his_kings ? 6 : source & his_queens ? 5 : source & his_rooks ? 4 : source & his_bishops ? 3 : source & his_knights ? 2 : source & his_pawns ? 1 : 0;
        move_scores[move_i] = (target_val<<3 | 6-source_val)  // bitmasks here mb +0 Elo
          + (target_val >= source_val ? 60 : 0);  // @0.06s 185371.5:185177.5 mb +0 Elo p=0.630, pourquoi..
      }
    }
    if (is_helper_thread && is_outer)
      for (int move_i = 0; move_i < cnt_possible_moves; ++move_i)
        move_scores[move_i] += gen() & 31;
  }

  move best_move_guess;
  if (tt_interaction) {
    //auto [entry, found] = ttable.find(b);
    bool found;
    int entry_depth, entry_lb, entry_ub;
    move entry_bestmove;
    if constexpr (true)
      found = qttable.find(b, entry_depth, entry_lb, entry_ub, entry_bestmove);
    else {
      auto [entry, found_] = qttable.find(b);
      if (found = found_)
        entry_depth = entry.depth, entry_lb = entry.lb, entry_ub = entry.ub, entry_bestmove = entry.bestmove;
    }
    if (found)
      best_move_guess = entry_bestmove;
  }
  if (best_move_guess == move() && depth >= 3) {
    int eval = compute_alphabeta_2<use_new, false, 0, is_helper_thread>(b, best_move_guess, depth - 2, -inf, +inf, gen, hash_history, ttable, qttable, killer_moves, nnue_state, stop_search);
    if (eval == unk) {
      if (maintain_hashes) hash_history.pop_back();
      return unk;
    }
  }

  if (!move_scores_active) {
    int promoted_count = 0;
    if (best_move_guess != move()) {
      auto it = std::ranges::find(possible_moves, best_move_guess);
      if (it != possible_moves.end()) {
        std::iter_swap(it, possible_moves.begin());
        promoted_count = 1;
      }
    }
    if (!is_outer && depth > 0 && promoted_count < possible_moves.size()) {  // @3s 2554:2337 +15 Elo
      move killer = killer_moves.moves[depth];
      if (killer != move()) {
        for (int i = promoted_count; i < possible_moves.size(); ++i) {
          if (possible_moves[i] == killer && !b.is_capture(possible_moves[i])) {
            std::swap(possible_moves[i], possible_moves[promoted_count++]);
            break;
          }
        }
      }
    }
  } else {
    move killer = killer_moves.moves[depth];
    int indices[cnt_possible_moves];
    std::pair<int, move> moves_[cnt_possible_moves];
    for (int move_i = 0; move_i < cnt_possible_moves; ++move_i) {
      move m = possible_moves[move_i];
      if (m == best_move_guess)
        move_scores[move_i] = 200;
      else if (m == killer)
        move_scores[move_i] = 100;
      indices[move_i] = move_i;
      moves_[move_i] = {-move_scores[move_i], m};
    }
    //std::ranges::sort(indices, indices + cnt_possible_moves, [&move_scores](int move_i1, int move_i2) { return move_scores[move_i1] > move_scores[move_i2]; });
    std::ranges::sort(moves_, moves_ + cnt_possible_moves);
    //std::vector<move> new_order;
    //for (int i : indices)
    //  new_order.push_back(possible_moves[i]);
    //std::ranges::copy(new_order, possible_moves.begin());
    for (int move_i = 0; move_i < cnt_possible_moves; ++move_i)
      possible_moves[move_i ] = moves_[move_i].second;
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
      nnue_state.update(b, tmp), value = -compute_alphabeta_2<use_new, false, split_depth, is_helper_thread>(tmp, useless, depth - 1, -beta, -alpha, gen, hash_history, ttable, qttable, killer_moves, nnue_state, stop_search), nnue_state.update(tmp, b);
    if (value == -unk) {
      if (maintain_hashes) hash_history.pop_back();
      return unk;
    }

    best_move_found = m;
    if (value >= beta) {
      if (maintain_hashes)
        hash_history.pop_back();
      if (tt_interaction)
        ttable.inform(b, depth, value, +TT<use_new>::tt_inf, best_move_found);
      if (!is_outer && depth >= 1 && !b.is_capture(best_move_found))
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

  auto search_child_node = [&b, &ttable, &qttable, mate, maintain_hashes, &hash_history, &nnue_state, &stop_search](const move &m, int depth, int alpha, int beta, int move_i, std::mt19937 &local_gen, std::vector<uint64_t> &local_hash_history, KillerMoves &killer_moves) -> int {
    nnue_state_t local_nnue_state(nnue_state);
    alpha -= is_outer;  // in case of outer, we really care if it's exactly α or less
    constexpr bool use_pvs = use_new;  // idk why but PVS is -10 Elo. I was unable to find the bug. maybe the move ordering is not ripe yet -- let's focus on implementing other important things for now; much later: @.4s 40k:40k mb +0 Elo; even later: @.1s 77457:77632 mb -0 Elo p=0.331
    const bool use_lmr = move_i >= 2; // @.3s 273:34 +362 Elo, @3s 68:34 +120 Elo, @30s 80:45 +100 Elo, 4 vs 2 @30s 79:123 -77 Elo; much later 3 vs 2 @.3s 23553:23826 mb -2 Elo, @3s 897:905 mb -2 Elo, 1 vs 2 @.3s 6314:6668 -9 Elo, @3s 2647:2878 -15 Elo
    // don't apply lmr to captures @.1s 8333:8738 -8 Elo @1s 12544:13039 -7 Elo @10s 4164:4332 mb -7 Elo p=0.034
    board tmp(b);
    move useless;
    int value = tmp.make_move(m) ? mate : 0;
    if (value == 0) {
      local_nnue_state.update(b, tmp);
      constexpr int new_split_depth = max(0, split_depth - 1);
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
        value = -compute_alphabeta_2<use_new, false, new_split_depth, is_helper_thread>(tmp, useless, depth - 1 - reduction, -beta, -alpha, local_gen, local_hash_history, ttable, qttable, killer_moves, local_nnue_state, stop_search);
        if (value == -unk) {  // bugfix: don't popback here, our parent will do it
          return unk;
        }
        if (value > alpha)
          value = -compute_alphabeta_2<use_new, false, new_split_depth, is_helper_thread>(tmp, useless, depth - 1, -beta, -alpha, local_gen, local_hash_history, ttable, qttable, killer_moves, local_nnue_state, stop_search);
      } else {
        value = -compute_alphabeta_2<use_new, false, new_split_depth, is_helper_thread>(tmp, useless, depth - 1, -beta, -alpha, local_gen, local_hash_history, ttable, qttable, killer_moves, local_nnue_state, stop_search);
      }
      if (value == -unk) {  // bugfix: don't popback here, our parent will do it
        return unk;
      }
      //local_nnue_state.update(tmp, b);  // not needed
    }
    return value;
  };

  // process remaining moves
  constexpr bool run_parallel = !is_outer && split_depth > 0;
    bool singular_hope = false;//use_new && !is_outer && depth >= 6;  // d≥6 sing_margin=40 @20s 13180:13050 mb +2 Elo p=0.788
    const int singular_margin = 40;

  if constexpr (run_parallel) {
    std::vector<int> results(possible_moves.size());
    std::mutex alpha_mutex;
    std::vector<std::jthread> workers;
    for (size_t i = 1; i < possible_moves.size(); ++i) {
      workers.emplace_back([&results, &alpha_mutex, depth, alpha, beta, &hash_history, &gen, &search_child_node, i, m=possible_moves[i], &killer_moves]() {
        std::vector<uint64_t> search_history_copy = hash_history;
        std::mt19937 thread_gen = gen;
        KillerMoves thread_killer_moves(killer_moves);
        results[i] = search_child_node(m, depth, alpha, beta, i, thread_gen, search_history_copy, thread_killer_moves);
      });
    }
    workers.clear();
    for (size_t i = 1; i < possible_moves.size(); ++i) {
      int value = results[i];
      if (value == unk) {
        if (maintain_hashes) hash_history.pop_back();
        return unk;
      }
      const auto& m = possible_moves[i];
      if (value > alpha)
        alpha = value, best_move_found = m, cnt_best = 1;
      if (alpha >= beta)
        break;
      if (is_outer && value == alpha && gen() % ++cnt_best == 0)
        best_move_found = m;
    }
    if (alpha >= beta) {
       if (maintain_hashes)
         hash_history.pop_back();
       if (tt_interaction)
         ttable.inform(b, depth, alpha, +TT<use_new>::tt_inf, best_move_found);
       if (!is_outer && depth > 0 && !b.is_capture(best_move_found))
         killer_moves.moves[depth] = best_move_found;
       put_bestmove = best_move_found;
       return beta;
    }
  } else {  // sequential
    for (size_t i = 1; i < possible_moves.size(); ++i) {
      const auto& m = possible_moves[i];
      int value;
      if (!singular_hope || search_child_node(m, depth - 2, alpha - singular_margin, alpha - singular_margin + 1, i, gen, hash_history, killer_moves) > alpha - singular_margin)  // mystery: even with +singular_margin in both places, sing ext h is @2s 436:420 mb +6 Elo p=0.696
        singular_hope = false;
      value = search_child_node(m, depth, alpha, beta, i, gen, hash_history, killer_moves);
      if (value == unk) {
        if (maintain_hashes) hash_history.pop_back();
        return unk;
      }
      if (value > alpha) {
        alpha = value, best_move_found = m, cnt_best = 1;
        if (is_outer)
          cnt_best = 1;
      } else if (is_outer && value == alpha && gen() % ++cnt_best == 0) {
        best_move_found = m;
      }
      if (alpha >= beta) {
        if (maintain_hashes)
          hash_history.pop_back();
        if (tt_interaction)
          ttable.inform(b, depth, alpha, +TT<use_new>::tt_inf, best_move_found);
        if (!is_outer && depth > 0 && !b.is_capture(best_move_found))
          killer_moves.moves[depth] = best_move_found;
        put_bestmove = best_move_found;
        return beta;
      }
    }
  }

  put_bestmove = best_move_found;
  if (singular_hope) {  // ..has not died
    board tmp(b);
    move useless;
    int value;
    if (tmp.make_move(put_bestmove))
      value = mate;
    else
      nnue_state.update(b,tmp), value = -compute_alphabeta_2<use_new,false,split_depth,is_helper_thread>(tmp, useless, depth, -beta, -alpha, gen, hash_history, ttable, qttable, killer_moves, nnue_state, stop_search), nnue_state.update(tmp,b);
    if (maintain_hashes)
      hash_history.pop_back();
    return value;
  }
  if (maintain_hashes)
    hash_history.pop_back();
  if (tt_interaction) {
    int lower_bound = alpha > alpha0 ? alpha : -TT<use_new>::tt_inf;
    ttable.inform(b, depth, lower_bound, alpha, best_move_found);
  }
  return alpha;
}

#endif
