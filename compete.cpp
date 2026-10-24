#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <random>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <sys/time.h>
#include "board.h"
#include "heuristic_eval.h"
#include "search.h"

extern const int enough;


template<bool test_multithreaded_games = false>
void compete(const auto& f_new, const auto& f_old, double scores[2]/*new:old*/, bool& finished, std::mutex& mutex, std::condition_variable& condvar, std::mt19937& host_gen, int cnt_threads, double time_control) {
  std::vector<std::jthread> threads;
  std::mutex score_mutex;
  for (int thread_i = 0; thread_i < (test_multithreaded_games?1:cnt_threads); ++thread_i) {
    threads.emplace_back([&](int thread_i, int seed) {
      constexpr int tt_size = 1 << 23;
      TT<false> tt_old(tt_size), qtt_old(tt_size/4);
      TT<true> tt_new(tt_size), qtt_new(tt_size/4);  // qttx2 @1s 2108:2142 mb -3 Elo p=0.295, @10s 4048.5:3962.5 mb +4 Elo p=0.831
      std::mt19937 gen(seed);
      for (int game_i = 1;; ++game_i) {
        const bool verbose = thread_i == 0 && game_i == 1;
        const bool is_new_white = thread_i + game_i & 1;
        if (game_i > 1) {
          condvar.notify_one();
        }
        std::unordered_map<board, int> history_cnts;
        std::string history_pgn;
        board b(true);
        double ttl_wc_time = 0, ttl_cpu_time = 0, clock_time_remaining[2] = {time_control, time_control};  // white:black
        std::vector<uint64_t> game_hash_history;
        const int n_init_halfmoves = 6;
        for (int init_m = 0; init_m < n_init_halfmoves; ++init_m) {
          moves_vec_t possible_moves = b.list_possible_moves<board::desire_type_t::vector_of_moves>();
          move m = possible_moves[gen() % possible_moves.size()];
          history_pgn += (init_m % 2 == 0 ? std::to_string(init_m / 2 + 1) + ". " : "") + b.get_pgn(m) + " ";
          (bool)b.make_move(m);
        }
        for (int move_i = n_init_halfmoves; move_i < 400; ++move_i) {
          const bool is_white_turn = move_i % 2 == 0, is_news_turn = is_new_white == is_white_turn;
          if (!b.has_legal_moves()) {
            std::scoped_lock sl(score_mutex);
            if (b.is_check())
              scores[is_news_turn]++;
            else
              scores[0] += .5, scores[1] += .5;
            break;
          }
          move m;
          double t0 = clock(), T0 = get_wall_time();
          KillerMoves killer_moves_new;
          KillerMoves killer_moves_old;
//          if (is_news_turn && (m = b.get_unique_legal_move()) != move()) {  // @0.1s 223720.5:224104.5 mb -0 Elo, mb due to cold TT
//            if (verbose) {
//              std::cout << "game " << game_i << " move " << move_i << " time " << get_wall_time() - T0 << " onlymove " << m << std::endl;
//            }
//          } else
          // early instability->think efforts: 30 .7 @.2s 1711.5:1533.5 +19 Elo, 40 .8 @2s 1029:1045 mb -3 Elo, 30 .8 @2s 5845.5:5515.5 +10 Elo, @.2s 1922.5:1732.5 +18 Elo, @30s being tested but sucks

          double time_budget = clock_time_remaining[!is_white_turn] / 80/*make 60 if +1ms/turn*/;  // trying eval instability heuristic with params x1.2 1.0 -50 +.5: @.3s 3845.5:3576.5 +13 Elo, @3s 448:552 -36 Elo
          double instability = .5;  // the very first intro of instability based on bestmove swinging was very successful: @0.2s 3162.5:2919.5 +14 Elo, @2s 589.5:487.5 +33 Elo, @40s 564:436 +45 Elo
          const double max_instability = 2.0;
          double prev_eval = -inf;
          std::vector<move> bestmoves;
          std::vector<std::jthread> helper_threads;
          std::atomic<bool> stop_search(false);
          /*if (is_news_turn) {
            nnue_state_t<true> nnue_state(b);
            move suggested_moves[cnt_threads];
            int vals[cnt_threads], max_cnt = 0;
            std::vector<std::jthread> threads;
            const int cnt_threads = 8;
            for (int thread_i = 0; thread_i < cnt_threads; ++thread_i)  // shadows thread_i but this is fine
              helper_threads.emplace_back([&](int thread_i, int seed) {
                std::mt19937 local_gen(seed);
                std::vector<uint64_t> local_hash_history(game_hash_history);
                KillerMoves local_killer_moves;
                nnue_state_t<true> local_nnue_state(b);
                move useless;
                for (int d = 0; d < max_depth; ++d)
                  compute_alphabeta_2<true,true,0,true>(b, useless, d, -inf, +inf, local_gen, local_hash_history, tt_new, qtt_new, local_killer_moves, local_nnue_state, stop_search);
              }, thread_i, gen());
          }*/
          for (int d = 1; d == 1 || d < max_depth && get_wall_time() - T0 <= time_budget * instability; d += 1) {  // /80:/160 @0.1s 645.5:373.5 +95 Elo, @.3s 662.5:358.5 +107 Elo, /40:/80 @0.1s 576.5:450.5 +43 Elo, @.3s 913.5:782.5 +27 Elo, @1s 3238:3337 mb -5 Elo p_>=.108, /60:/80 @0.1s 741:625 +30 Elo, @.3s 3141:2806 +20 Elo, @1s 598.5:495.5 +33 Elo, @3s 3197:2955 +14 Elo, @20s 2168.5:1969.5 +17 Elo; /40:/60 @.3s 10321:9879 +8 Elo, @1s 2018:2221 -17 Elo, @3s 507.5:611.5 -32 Elo
            //double nps = ttl_nodes / (ttl_wc_time_ours+1e-6);
            //if (verbose) std::cout << "nps " << nps << std::endl;
            int eval;//, init_nodes_left = !is_news_turn || ttl_nodes < 100 ? 2e9 : min(2e9, 10 * time_budget * instability * nps);
            //if (verbose) std::cout << "init_nodes_left " << init_nodes_left << std::endl;
            //std::atomic<int> nodes_left(init_nodes_left);
            if (is_news_turn) {
              eval = f_new(b, m, d, gen, game_hash_history, tt_new, qtt_new, killer_moves_new);
//              ttl_nodes += init_nodes_left - nodes_left;
//              if (verbose) std::cout << "nodes left " << nodes_left << " of " << init_nodes_left << std::endl;
//              if (eval == unk) {
//                if (verbose) std::cout << "CUT SHORT" << std::endl;
//                eval = prev_eval, m = bestmoves.back();
//                break;
//              }
            } else {
              eval = f_old(b, m, d, gen, game_hash_history, tt_old, qtt_old, killer_moves_old);
            }
            if (eval < prev_eval - 50)
              instability += .5;
            prev_eval = eval;
            bestmoves.push_back(m);
            assert(bestmoves.size() == d);
            //if (d >= 2 && bestmoves[d - 2] == bestmoves[d - 1])
            //  time_budget *= .8;
            if (d >= 2 && bestmoves[d - 1] != bestmoves[d - 2])
              instability += .25;
            //if (is_news_turn)
            //  instability = min(instability, max_instability);  // @.3s 15981:15425 +6 Elo, @3s 19923:19310 +5 Elo, @30s 6437:6091 +10 Elo
            if (verbose) std::cout << "game " << game_i << " move " << move_i << " depth " << d << " time " << get_wall_time() - T0 << " eval " << eval << " bestmove " << m << std::endl;
          }
          /*if (is_news_turn) {
            stop_search = true;
            helper_threads.clear();
            if (verbose) std::cout << "game " << game_i << " move " << move_i << " final time " << get_wall_time() - T0 << std::endl;
          }*/
          //if (is_news_turn) ttl_wc_time_ours += get_wall_time() - T0;
          double wc_time = get_wall_time() - T0, cpu_time = (clock() - t0) / double(CLOCKS_PER_SEC);
          clock_time_remaining[!is_white_turn] -= wc_time - .001;
          if (verbose) {
            std::cout << "(wallclock time " << wc_time << "s, CPU time " << cpu_time << "s, on the clock " << clock_time_remaining[0] << "s:" << clock_time_remaining[1] << "s)\n";
            std::cout << "(total wallclock time " << (ttl_wc_time += wc_time) << "s, total CPU time " << (ttl_cpu_time += cpu_time) << "s)\n";
            std::cout << "move to be made: " << move(m.i0, m.j0, m.i1, m.j1) << std::endl;
          }
          if (clock_time_remaining[!is_white_turn] < 0) {
            if (verbose) std::cout << (is_white_turn ? "WHITE" : "BLACK") << " LOST BY TIME" << std::endl;
            std::scoped_lock sl(score_mutex);
            scores[is_news_turn]++;
            break;
          }
          //history_pgn += (is_white_turn ? std::to_string(move_i / 2 + 1) + ". " : "") + b.get_pgn(m, is_white_turn) + " ";
          history_pgn += (is_white_turn ? std::to_string(move_i / 2 + 1) + ". " : "") + b.get_pgn(m) + " ";
          game_hash_history.push_back(b.get_hash());
          if (b.is_stalemate()) {
            std::scoped_lock sl(score_mutex);
            scores[0] += .5, scores[1] += .5;
            break;
          } else if (b.make_move(m)) {
            if (verbose) std::cout << (is_white_turn ? "WHITE" : "BLACK") << " WON" << std::endl;
            std::scoped_lock sl(score_mutex);
            scores[!is_news_turn]++;
            break;
          } else if (++history_cnts[b] == 3) {
            if (verbose) std::cout << "DRAW BY REPETITION" << std::endl;
            std::scoped_lock sl(score_mutex);
            scores[0] += .5, scores[1] += .5;
            break;
          } else if (b.white_knights == 0 && b.white_pawns == 0 && b.white_bishops == 0 && b.white_rooks == 0 && b.white_queens == 0
                  && b.black_knights == 0 && b.black_pawns == 0 && b.black_bishops == 0 && b.black_rooks == 0 && b.black_queens == 0) {
            if (verbose) {
              std::cout << "DRAW BY 2 KINGS [shortcut]" << std::endl;
            }
            std::scoped_lock sl(score_mutex);
            scores[0] += .5, scores[1] += .5;
            break;
          } else if (verbose) {
            b.show();
            std::cout << "\n\n";
            if (move_i == 399) {
              std::cout << "DRAW BECAUSE TOO LONG" << std::endl;
              std::scoped_lock sl(score_mutex);
              scores[0] += .5, scores[1] += .5;
              break;
            }
          }
          if (finished) {
            std::cout << "Thread " << thread_i << " left the party because it was told it's over" << std::endl;
            return;
          }
        }
        if (verbose) {
          std::cout << "PGN: " << history_pgn << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "\n";
          }
        }
      }
    }, thread_i, host_gen());
  }
}


template<typename T>
T logsumexp(const std::vector<T> v) {
  T max = *std::max_element(v.begin(), v.end()), sum = 0;
  for (const T x : v) sum += expl(x - max);
  return logl(sum) + max;
}

int main(int argc, const char **argv) {
  // what we were GIVING AWAY with pst+datasets: @0.1s 952:817 +27 Elo, @1s 305.5:232.5 +47 Elo, @3s 186.5:113.5 +86 Elo, @10s 179:121 +68 Elo; we more than recovered since
  //const std::string update_desc = "9f9:9f9 7 /2 : 15 /4";
  const std::string update_desc = "pvs";
  const float time_control = argc > 1 ? std::stof(argv[1]) : 20;
  // current mt8 version:single th @10s 75:47 mb +81 Elo p_>=.994 @100s 72.5:37.5 +114 Elo


  auto report_str = [&](double scores[2], bool final_say) {
    double deltaElo = scores[1] == 0 ? 1000 : (scores[0] == 0 ? -1000 : 400 * log10(scores[0] / scores[1]));
    const std::string tc_str = "@" + (time_control > 2 ? std::to_string((int)time_control) : time_control == (float).3 ? ".3" : time_control == (float).2 ? ".2" : std::to_string(time_control)) + "s";
    return tc_str + " " + std::to_string(scores[0])+":"+std::to_string(scores[1]) + (scores[0]+scores[1]>=300||final_say ? std::string(final_say?" ":" mb ") + (deltaElo>=0?"+":"-")+std::to_string(std::abs(std::round(deltaElo)))+" Elo" : "");
  };
  std::mt19937 host_gen(time(0));
  board::precalc(57);
  precalc_search();
  precalc_heuristic_eval();
  precalc_nn_eval();
  std::cout << "Competition started!" << std::endl;

  constexpr bool test_multithreaded_games = true;
  const int cnt_threads = 8;
  auto old_strategy = [](const board& b, move& m, int d, std::mt19937& gen, std::vector<uint64_t> game_hash_history, TT<false>& local_ttable, TT<false>& local_qttable, KillerMoves& killer_moves) {
    nnue_state_t<false> nnue_state(b);
    std::atomic<bool> stop_search(false);
    return compute_alphabeta_2<false, true, 0,false>(b, m, d, -inf, +inf, gen, game_hash_history, local_ttable, local_qttable, killer_moves, nnue_state, stop_search);
    // this lazy smp
    /*nnue_state_t<false> nnue_state(b);
    move suggested_moves[cnt_threads];
    int vals[cnt_threads], max_cnt = 0;
    std::vector<std::jthread> threads;
    std::atomic<bool> stop_search(false);
    for (int thread_i = 1; thread_i < cnt_threads; ++thread_i)
      threads.emplace_back([&](int thread_i) {
        std::mt19937 local_gen(gen());
        std::vector<uint64_t> local_hash_history(game_hash_history);
        KillerMoves local_killer_moves(killer_moves);
        nnue_state_t local_nnue_state(b);
        vals[thread_i] = compute_alphabeta_2<false,true,0,false>(b, suggested_moves[thread_i], d + thread_i, -inf, +inf, local_gen, local_hash_history, local_ttable, local_qttable, local_killer_moves, local_nnue_state, stop_search);
        if (thread_i == 0)
          stop_search = true;
      }, thread_i);
    threads.clear();
    m = suggested_moves[0];
    return vals[0];*/
  };
  auto new_strategy = [](const board& b, move& m, int d, std::mt19937& gen, std::vector<uint64_t> game_hash_history, TT<true>& local_ttable, TT<true>& local_qttable, KillerMoves& killer_moves) {
      nnue_state_t<true> nnue_state(b);
      std::atomic<bool> stop_search(false);
      return compute_alphabeta_2<true,true,0,false>(b, m, d, -inf, +inf, gen, game_hash_history, local_ttable, local_qttable, killer_moves, nnue_state, stop_search);
    // this lazy smp
    /*nnue_state_t<true> nnue_state(b);
    move suggested_moves[cnt_threads];
    int vals[cnt_threads], max_cnt = 0;
    std::vector<std::jthread> threads;
    std::atomic<bool> stop_search(false);
    for (int thread_i = 0; thread_i < cnt_threads; ++thread_i)
      threads.emplace_back([&](int thread_i) {
        std::mt19937 local_gen(gen());
        std::vector<uint64_t> local_hash_history(game_hash_history);
        KillerMoves local_killer_moves(killer_moves);
        nnue_state_t local_nnue_state(b);
        vals[thread_i] = (thread_i==0?compute_alphabeta_2<true,true,0,false>:compute_alphabeta_2<true,true,0,true>)(b, suggested_moves[thread_i], d, -inf, +inf, local_gen, local_hash_history, local_ttable, local_qttable, local_killer_moves, local_nnue_state, stop_search);
        if (thread_i == 0)
          stop_search = true;
      }, thread_i);
    threads.clear();*/
    //m = suggested_moves[0];
    //return vals[0];
    //return compute_alphabeta_2<true, true, 0>(b, m, d, -inf, +inf, gen, game_hash_history, local_ttable, local_qttable, killer_moves, nnue_state);
  };
  double scores[2] = {};
  bool finished = false;
  std::mutex mutex;
  std::condition_variable condvar;
  std::thread competition_host([&]() {
    compete<false>(new_strategy, old_strategy, scores, finished, mutex, condvar, host_gen, 24, time_control);
  });
  int last_total = 0;
  std::unique_lock lock(mutex);
  condvar.wait(lock, [&]() {
    int new_total = scores[0] + scores[1];
    if (new_total == 0) return false;
    std::vector<long double> log_summands_num, log_summands_den;
    for (auto [x, dx] = std::pair{0.5e-6, 1e-6}; x < 1; x += dx) {
      long double log_term = scores[0] * std::log(x) + scores[1] * log(1 - x) + std::log(dx);
      log_summands_den.push_back(log_term);
      if (x >= .5) log_summands_num.push_back(log_term);
    }
    long double log_integral = logsumexp(log_summands_num), log_evidence = logsumexp(log_summands_den),
                p_better = std::exp(log_integral - log_evidence);
    if (new_total < last_total * 1.02 && .001 < p_better && p_better < .999) return false;
    last_total = new_total;
    std::string s = report_str(scores, false);
    std::cout << update_desc << "\n";
    std::cout << s;
    //for (int i = s.length()+1; i < 40; ++i) std::cout << ' ';
    if (log_integral == std::numeric_limits<long double>::lowest() || log_evidence == std::numeric_limits<long double>::lowest()) {
      std::cout << "wtf!!" << std::endl;
      return true;
    }
    if (new_total >= 50) std::cout << /*"P(better) = " <<*/ " p=" << p_better;
    std::cout << ",\n" << std::endl;
    return new_total >= 300 && (p_better < .001 || p_better > .999) && (finished = true);
  });
  competition_host.join();
  std::cout << "\n\n ====================\n";
  std::cout << update_desc << std::endl;
  std::cout << report_str(scores, true) << std::endl;
}
