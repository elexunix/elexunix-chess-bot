#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <sys/time.h>
#include "board.h"
#include "heuristic_eval.h"
#include "nn_eval.h"
#include "search.h"


constexpr bool use_new = true;


int main() {
  const int seed = 57;
  board::precalc(seed);
  precalc_search();
  precalc_heuristic_eval();
  precalc_nn_eval();
  std::mt19937 host_gen(seed);
  board b(57);
  std::vector<board> history {b};
  if (true) {
    std::cout << "\n\n";
    std::cout << "=====================================================================================\n";
    std::cout << "|                                                                                   |\n";
    std::cout << "|    ELEXUNIX CHESS ENGINE                vs        FEDRO FEDRATION                 |\n";
    std::cout << "|    (Anti-Fedro upgrade)                                                           |\n";
    //std::cout << "|                                 October 2025 Finals                               |\n";
    std::cout << "|                               September 2026 Finals                               |\n";
    std::cout << "=====================================================================================\n";
    std::cout << "| Expected results: Complete Defeat of the Fedro Connected Component                |\n";
    std::cout << "\n\n\n\n";
  }
reenter_color:
  std::cout << "choose your number (first/second): ";
  std::string s;
  std::cin >> s;
  if (s != "first" && s != "second") {
    goto reenter_color;
  }
  if (s == "first") {
    b.show();
    move m;
    b.read_move(m, false);
    b.make_move(m);
    b.show();
    history.push_back(b);
  }
  double ttl_wc_time = 0, ttl_cpu_time = 0;
  double remaining_time = 600;
  std::vector<uint64_t> hash_history;
  int move_i = 1;
  TT<use_new> ttable(1 << 27), qttable(1 << 23);
  KillerMoves killer_moves;
  nnue_state_t<use_new> nnue_state(b);
  for (int i = 0; i < 10; ++i)
    hash_history.push_back(b.get_hash());
  while (true) {
    move m;
    double t0 = clock(), T0 = get_wall_time();
    for (int d = 1; get_wall_time() - T0 <= 2 && d <= 50; d += 1) {  // (0 + remaining_time / 50) / 4
      int eval = compute_alphabeta_2<use_new, true, 1>(b, m, d, -inf, +inf, host_gen, hash_history, ttable, qttable, killer_moves, nnue_state);
      std::cout << "depth " << d << " time " << get_wall_time() - T0 << " eval " << eval << " bestmove " << m << std::endl;
    }
    double wc_time = get_wall_time() - T0, cpu_time = (clock() - t0) / double(CLOCKS_PER_SEC);
    std::cout << "(wallclock time " << wc_time << "s, CPU time " << cpu_time << "s)\n";
    std::cout << "(total wallclock time " << (ttl_wc_time += wc_time) << "s, total CPU time " << (ttl_cpu_time += cpu_time) << "s)\n";
    //std::cout << "(remaining time " << (remaining_time -= wc_time) << ")" << std::endl;
    std::cout << "move to be made: " << move(m.i0, m.j0, m.i1, m.j1) << std::endl;
    hash_history.push_back(b.get_hash());
    b.make_move(m);
    b.show();
    history.push_back(b);
    std::cout << "hash_history = {";
    for (uint64_t x : hash_history) {
      std::cout << x << ", ";
    }
    std::cout << "};\n";
read_move:
    b.read_move(m, false);
    if (m.i0 == -2) {  // requested undo
      history.pop_back();
      history.pop_back();
      ttable.clear();  // otherwise we might have 0s for positions that are not repeating anymore
      b = history.back();
      b.show();
      goto read_move;
    }
    //m = move_i == 1 ? move({1, 3, 3, 3}) : move({0, 3, 2, 3});
    ++move_i;
    hash_history.push_back(b.get_hash());
    b.make_move(m);
    b.show();
    history.push_back(b);
  }
}
