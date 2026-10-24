#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <string>
#include "board.cuh"
#include "heuristic_eval.cuh"
#include "search.cuh"

constexpr bool use_new = false;

std::mutex file_mutex;

constexpr int depth = 4;
constexpr int k1 = 512, k2 = 128, k12 = k1 * k2, games_per_thread = 1;
const std::string filename = "dataset_cuda.csv";

__host__ __device__ void get_fen(const board& b, string& put_fen) {
  put_fen = b.get_fen();
}

struct PositionRecord {
  bool nonempty = false;
  string fen;
  int8_t game_outcome;
  int evals[11], offset_vals[3];
  __host__ __device__ PositionRecord(bool nonempty = false) : nonempty(nonempty) {}
};

__global__ void worker_thread(int games_to_play, PositionRecord put_data[k12][max_game_length]) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t curand_state;
  curand_init(id, id, id, &curand_state);
  constexpr int tt_size = 1 << 10;
  TT<use_new> very_local_tt(tt_size);
  //std::mt19937 gen(57 + thread_id);
  int seed = 57 + id;
  curand_init(seed, seed, seed, &curand_state);

  for (int g = 0; g < games_to_play; ++g) {
    board b(57); // Startpos
    hashes_vec_t hash_history;
    PositionRecord game_history[max_game_length];
    KillerMoves local_killers;
    int outcome = 0; // 0 patta, 1 vince bianco, -1 vince nero
    int game_length = 0;

    // Per avere varietà nel dataset, giochiamo edit:molte mosse casuali in apertura
    for(int i = 0; i < 8 + curand(&curand_state) % 8; ++i) {
      moves_vec_t moves = b.list_possible_moves<board::desire_type_t::vector_of_moves>();
      int cnt_possible_moves = moves.size();
      if (!cnt_possible_moves)
        break;
      b.make_move(moves[curand(&curand_state) % cnt_possible_moves]);
      hash_history.push_back(b.get_hash());
      game_length++;
    }

    while (true) {
      PositionRecord rec(true);
      bool flip = b.side2move == BLACK;
      if (flip)
        b.toggle_identity();
      get_fen(b, rec.fen);
      if (flip)
        b.toggle_identity();

      // Il motore valuta per chi deve muovere. Moltiplichiamo per avere il pov del Bianco.
      int pov = b.side2move == WHITE ? 1 : -1;
      nnue_state_t<use_new> nnue_state(b);
      rec.evals[0] = b.hc_heuristic_score<use_new>(nnue_state) * pov;

      move best_m;
      for(int d = 1; d <= depth; ++d) {
        move current_m;
        // Uso split_depth=0 perché ogni thread è già 100% occupato dalla sua partita
        int score = compute_alphabeta_2<use_new, true, 0>(b, current_m, d, -inf, +inf, curand_state, hash_history, very_local_tt, local_killers, nnue_state);
        rec.evals[d] = score * pov;
        if (current_m.i0 != -1)
          best_m = current_m;
      }

      game_history[game_length] = rec;

      // Termine partita (Matto o stallo)
      if (best_m.i0 == -1) {
        if (b.is_check())
          outcome = b.side2move == WHITE ? -1 : 1;
        break;
      }

      // Limitiamo le partite troppo lunghe
      if (game_length == max_game_length - 1) {
        outcome = 0;
        break;
      }

      b.make_move(best_m);
      hash_history.push_back(b.get_hash());
      game_length++;
    }

    for (int i = 0; i < game_length; ++i) {
      game_history[i].game_outcome = outcome;
      for (int j = 0; j < 3; ++j) {
        int off = j == 0 ? 5 : j == 1 ? 10 : 20;
        game_history[i].offset_vals[j] = i + off < game_length ? game_history[i + off].evals[0] : outcome * 10000;
      }
    }

    for (int i = 0; i <= game_length; ++i) {
      put_data[id][i] = i < game_length ? game_history[i] : PositionRecord();
    }
//    // Scrittura Thread-Safe nel CSV
//    std::lock_guard<std::mutex> lock(file_mutex);
//    for (size_t i = 0; i < game_history.size(); ++i) {
//      out_csv << game_history[i].fen;
//      for (int d = 0; d <= depth; ++d) {
//        out_csv << "," << game_history[i].evals[d];
//      }
//
//      // Lookahead: le valutazioni a d=0 tra 5, 10, 20 mezzi turni (ply)
//      int offsets[3] = {5, 10, 20};
//      for (int off : offsets) {
//        if (i + off < game_history.size()) {
//          out_csv << "," << game_history[i + off].evals[0];
//        } else {
//          out_csv << "," << (outcome * 10000); // +100.00 se il Bianco ha già vinto, -100.00 Nero
//        }
//      }
//      out_csv << "," << outcome << "\n";
//    }

    // Piccolo log per farti capire a che punto è
    //if (g % 5 == 0)
    //  std::cout << "Thread " << thread_id << " finished " << g << " games.\n";
  }
}

int main(int argc, const char **argv) {
  std::cout << "STARTING" << std::endl;
  board::precalc(57);

  bool existed = std::filesystem::exists(filename);
  std::cout << (existed ? "APPENDING TO EXISTING DATASET" : "CREATING A NEW DATASET") << std::endl;
  std::ofstream out_csv(filename, std::ios::app);
  if (!existed) {
    out_csv << "fen";
    for (int d = 0; d <= depth; ++d) out_csv << ",eval_d" << d;
    out_csv << ",eval_5,eval_10,eval_20,outcome\n";
  }

  /*std::vector<std::jthread> threads;
  for (int i = 0; i < cnt_threads; ++i) {
    threads.emplace_back(worker_thread, i, games_per_thread, std::ref(out_csv));
  }*/
  PositionRecord (*rollouts_data)[max_game_length];
  gpuErrchk(cudaMallocManaged(&rollouts_data, k12 * max_game_length * sizeof(PositionRecord)));
  worker_thread<<<k1,k2>>>(games_per_thread, rollouts_data);
  std::cout << "Avviati " << k12 << " thread. Raccolta dati in corso..." << std::endl;
  gpuErrchk(cudaDeviceSynchronize());
  return 0;
}
