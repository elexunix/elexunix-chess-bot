#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <string>
#include "board.h"
#include "heuristic_eval.h"
#include "search.h"

constexpr bool use_new = false;

std::mutex file_mutex;

const int depth = 4;
const int games_total = 24000000;
const std::string filename = "/media/elexunix/hugedrive/datasets/elexunix-chess-bot/dataset11_d4_0.csv";


const int cnt_threads = 24;//std::thread::hardware_concurrency();
const int games_per_thread = games_total / 24;

std::string get_fen(const board& b) {
  return b.get_fen();
}

struct PositionRecord {
  std::string fen;
  int evals[11];
};

void worker_thread(int thread_id, int games_to_play, std::ofstream& out_csv) {
  constexpr int tt_size = 1 << 23;
  TT<use_new> local_tt(tt_size), local_qtt(tt_size/8);
  std::mt19937 gen(57 + thread_id);
  const std::atomic<bool> stop_search(false);

  for (int g = 0; g < games_to_play; ++g) {
  replay:  // thankfully, goto'ing here DOES clear vectors below automatically
    board b(57 + thread_id); // Startpos
    std::vector<uint64_t> hash_history;
    std::vector<PositionRecord> game_history;
    KillerMoves local_killers;

    int random_ply = 6 + gen() % 6;
    for (int i = 0; i < random_ply; ++i) {
      auto moves = b.list_possible_moves<board::desire_type_t::vector_of_moves>();
      if (moves.empty())
        break;
      if (b.make_move(moves[gen() % moves.size()]))
        goto replay;
      hash_history.push_back(b.get_hash());
    }
    if (b.can_eat_king())
      goto replay;
    //{
      //move useless;
      //nnue_state_t<use_new> nnue_state(b);
      //int eval = compute_alphabeta_2<use_new,true,0,false>(b, useless, depth, -inf, +inf, gen, hash_history, local_tt, local_qtt, local_killers, nnue_state, stop_search);
      //if (!(-500 <= eval && eval <= +500))
      //  goto replay;
    //}

    int outcome = 0; // 0 patta, 1 vince bianco, -1 vince nero
    int ply_count = 0;

    while (true) {
      PositionRecord rec;
      bool flip = b.side2move == BLACK;
      rec.fen = get_fen(b);

      int pov = b.side2move == WHITE ? 1 : -1;
      nnue_state_t<use_new> nnue_state(b);
      rec.evals[0] = b.hc_heuristic_score<use_new>(nnue_state) * pov;  // il motore valuta per chi deve muovere. Moltiplichiamo per avere il pov del Bianco.

      move best_m;
      for(int d = /*1*/depth; d <= depth; ++d) {
        move current_m;
        int score = compute_alphabeta_2<use_new,true,0,false>(b, current_m, d, -inf, +inf, gen, hash_history, local_tt, local_qtt, local_killers, nnue_state, stop_search);
        rec.evals[d] = score * pov;
        if (current_m == move()) {
          // game ended, we are checkmated/stalemated
          // keep best_m nullmove, the if after loop will end the game
        }
        else
          best_m = current_m;
      }
      //game_history.push_back(rec);

      if (best_m == move()) {
        if (b.is_check())
          outcome = -pov;
        break;
      }

      int standing_pat = b.hc_heuristic_score(nnue_state) * pov;  // for white
      if (!b.is_check() && !b.is_capture<1>(best_m))
        game_history.push_back(rec);

      assert(!b.is_stalemate());  // altrimenti avrebbe dovuto restituire
      if (ply_count++ > 250)
        break;
      bool ate_king = b.make_move(best_m);
      assert(!ate_king);

      if (std::find(hash_history.begin(), hash_history.end(), b.get_hash()) != hash_history.end())
        break;
      hash_history.push_back(b.get_hash());
    }

    std::lock_guard<std::mutex> lock(file_mutex);
    for (size_t i = 0; i < game_history.size(); ++i) {
      out_csv << game_history[i].fen;
      for (int d = 0; d <= depth; d += depth)
        out_csv << "," << game_history[i].evals[d];
      // le valutazioni a d=0 tra 5, 10, 20 mezzi turni
      int offsets[3] = {5, 10, 20};
      for (int off : offsets) {
        if (i + off < game_history.size())
          out_csv << "," << game_history[i + off].evals[0];
        else
          out_csv << "," << outcome * 10000;
      }
      out_csv << "," << outcome << "\n";
    }

    if (g % 500 == 0)
      std::cout << "Thread " << thread_id << " finished " << g << " games.\n";
  }
}

int main(int argc, const char **argv) {
  board::precalc(57);
  precalc_search();
  precalc_heuristic_eval();
  precalc_nn_eval();

  bool existed = std::filesystem::exists(filename);
  std::cout << (existed ? "APPENDING TO EXISTING DATASET" : "CREATING A NEW DATASET") << std::endl;
  std::ofstream out_csv(filename, std::ios::app);
  if (!existed) {
    out_csv << "fen";
    for (int d = 0; d <= depth; d += depth)
      out_csv << ",eval_d" << d;
    out_csv << ",eval_5,eval_10,eval_20,outcome\n";
  }

  std::vector<std::jthread> threads;
  for (int i = 0; i < cnt_threads; ++i)
    threads.emplace_back(worker_thread, i, games_per_thread, std::ref(out_csv));

  std::cout << "Avviati " << cnt_threads << " thread. Raccolta dati in corso..." << std::endl;
  return 0;
}
