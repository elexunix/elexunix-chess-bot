#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <thread>
#include <sys/time.h>
#include "board.h"
#include "heuristic_eval.h"
#include "search.h"

constexpr bool use_new = false;

std::string move_to_uci(const move& m, const board& b) {
  std::string uci = "";
  uci += char('a' + m.j0);
  uci += char('1' + m.i0);
  uci += char('a' + m.j1);
  uci += char('1' + m.i1);

  uint64_t from_mask = 1ull << (m.i0 * 8 + m.j0);
  if (b.side2move == WHITE && (from_mask & b.white_pawns) && m.i1 == 7) {
    uci += "q";
  } else if (b.side2move == BLACK && (from_mask & b.black_pawns) && m.i1 == 0) {
    uci += "q";
  }

  return uci;
}

int main() {
  const int seed = 57;
  board::precalc(seed);
  precalc_search();
  precalc_heuristic_eval();
  precalc_nn_eval();
  std::mt19937 gen(seed);
  board b(seed);

  TT<use_new> ttable(67108864), qttable(16777216);
  KillerMoves killer_moves;
  std::vector<uint64_t> hash_history;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "uci") {
      std::cout << "id name elexunix_chess_bot" << std::endl;
      std::cout << "id author elexunix" << std::endl;

      std::cout << "option name Move Overhead type spin default 100 min 0 max 5000" << std::endl;
      std::cout << "option name Threads type spin default 1 min 1 max 128" << std::endl;
      std::cout << "option name Hash type spin default 16 min 1 max 32000" << std::endl;
      std::cout << "option name SyzygyPath type string default <empty>" << std::endl;
      std::cout << "option name Ponder type check default false" << std::endl;
      std::cout << "option name MultiPV type spin default 1 min 1 max 500" << std::endl;
      std::cout << "option name UCI_ShowWDL type check default false" << std::endl;
      std::cout << "option name UCI_Chess960 type check default false\n";
      std::cout << "option name UCI_Variant type combo default chess var chess var chess960 var 3check\n";

      std::cout << "uciok" << std::endl;
    }
    else if (token == "setoption") {
      std::string dummy;
      std::getline(iss, dummy);
    }
    else if (token == "isready") {
      std::cout << "readyok" << std::endl;
    }
    else if (token == "ucinewgame") {
      ttable.clear();
      hash_history.clear();
      b = board(seed);
    }
    else if (token == "position") {
      size_t moves_pos = line.find(" moves ");
      std::string pos_part = (moves_pos != std::string::npos) ? line.substr(0, moves_pos) : line;

      if (pos_part.find("startpos") != std::string::npos) {
        b = board(seed);
      } else {
        size_t fen_pos = pos_part.find("fen ");
        if (fen_pos != std::string::npos) {
          std::string fen_str = pos_part.substr(fen_pos + 4);
          b = board(fen_str);
        }
      }

      hash_history.clear();
      ttable.clear();
      /*for (int i = 0; i < 10; ++i) {
        hash_history.push_back(b.get_hash());
      }*/

      if (moves_pos != std::string::npos) {
        std::string moves_str = line.substr(moves_pos + 7);
        std::istringstream moves_iss(moves_str);
        std::string move_token;

        while (moves_iss >> move_token) {
          int j0 = move_token[0] - 'a';
          int i0 = move_token[1] - '1';
          int j1 = move_token[2] - 'a';
          int i1 = move_token[3] - '1';

          move m(i0, j0, i1, j1);
          hash_history.push_back(b.get_hash());
          (void)b.make_move(m);
        }
      }
    }
    else if (token == "go") {
      double allocated_time = 2.0;
      std::string param;

      while (iss >> param) {
        if (param == "wtime" && b.side2move == WHITE) {
          int ms; iss >> ms;
          allocated_time = (ms / 1000.0) / 80.0;
        } else if (param == "btime" && b.side2move == BLACK) {
          int ms; iss >> ms;
          allocated_time = (ms / 1000.0) / 80.0;
        } else if (param == "movetime") {
          int ms; iss >> ms;
          allocated_time = (ms / 1000.0);
        }
      }

      if (allocated_time < 0.05)
        allocated_time = 0.05;  // given our poor connection, this is fine
      /*std::cout << "info hash_history = {";
      for (uint64_t h : hash_history) {
        std::cout << h << ", ";
      }
      std::cout << "};" << std::endl;*/

      move best_m, prev_best_m;
      double T0 = get_wall_time(), instability = .5;
      const double max_instability = 2.0;

      int prev_eval = -inf;
      nnue_state_t<use_new> nnue_state(b);
      std::vector<std::jthread> helper_threads;
      std::atomic<bool> stop_search(false);
      const int cnt_threads = 8;
      for (int thread_i = 0; thread_i < cnt_threads; ++thread_i)  // shadows thread_i but this is fine
        helper_threads.emplace_back([&b,hash_history,&ttable,&qttable,&stop_search](int thread_i, int seed) {
          std::mt19937 local_gen(seed);
          std::vector<uint64_t> local_hash_history(hash_history);
          KillerMoves local_killer_moves;
          nnue_state_t<use_new> local_nnue_state(b);
          move useless;
          for (int d = 0; d < max_depth; ++d)
            compute_alphabeta_2<use_new,true,0,true>(b, useless, d, -inf, +inf, local_gen, local_hash_history, ttable, qttable, local_killer_moves, local_nnue_state, stop_search);
        }, thread_i, gen());
      for (int d = 1; d < max_depth; d++) {
        move current_m;
//        std::atomic<int> nodes_left(3 * 3e5 * time_budget * instability);
        nnue_state_t<use_new> nnue_state(b);
        // single-threaded!!
        //int eval = compute_alphabeta_2<use_new, true, 0>(b, current_m, d, -inf, +inf, gen, hash_history, ttable, qttable, killer_moves, nnue_state);

        // this lazy smp
        int eval = compute_alphabeta_2<use_new,true,0,false>(b, current_m, d, -inf, +inf, gen, hash_history, ttable, qttable, killer_moves, nnue_state, stop_search);
//        if (eval == unk) {
//          eval = prev_eval, current_m = prev_best_m;
//        }
        double elapsed = get_wall_time() - T0;

        if (current_m.i0 != -1 && current_m.j0 != -1) {
          best_m = current_m;
          std::cout << "info depth " << d << " score cp " << eval << " time " << int(elapsed * 1000) << " pv " << move_to_uci(best_m, b) << std::endl;
        }
        if (best_m != prev_best_m) {
          if (d >= 2) instability += .25;
          prev_best_m = best_m;
        }
        if (eval < prev_eval - 50) {
          instability += .5;
        }
        prev_eval = eval;
        instability = min(instability, max_instability);

        if (elapsed >= allocated_time * instability) {
          stop_search = true;
          helper_threads.clear();
          break;
        }
      }

      std::cout << "bestmove " << move_to_uci(best_m, b) << std::endl;
    }
    else if (token == "quit") {
      break;
    }
  }

  return 0;
}
