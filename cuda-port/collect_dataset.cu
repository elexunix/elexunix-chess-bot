#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <string>
//#include <cuda/std/algorithm>
#include "board.cuh"
#include "heuristic_eval.cuh"
#include "search.cuh"

constexpr bool use_new = false;

std::mutex file_mutex;

constexpr int depth = 3;
constexpr int k1 = 256, k2 = 128, k12 = k1 * k2, games_per_thread = 1;
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

__global__ void worker_thread(int games_to_play, PositionRecord put_data[k12][max_game_length], int seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t curand_state;
  curand_init(seed + id, seed + id, seed + id, &curand_state);
  constexpr int tt_size = 1 << 9;
  TT<use_new> very_local_tt(tt_size);
  //std::mt19937 gen(57 + thread_id);
  int local_seed = seed + id;
  curand_init(local_seed, local_seed, local_seed, &curand_state);
    //if (id <= 3)
    //  printf("%d\n",curand(&curand_state));
  for (int g = 0; g < games_to_play; ++g) {
    assert(games_to_play == 1);
  replay:
    board b(local_seed);  // startpos
    hashes_vec_t hash_history;
    PositionRecord game_history[max_game_length];
    KillerMoves local_killers;
    int outcome = 0;  // 0 patta, 1 vince bianco, -1 vince nero
    int game_length = 0;
    for (int i = 0; i < 4 + curand(&curand_state) % 4; ++i) {  // per avere varietà nel dataset, giochiamo edit:molte mosse casuali in apertura
      moves_vec_t moves = b.list_possible_moves<board::desire_type_t::vector_of_moves>();
      int cnt_possible_moves = moves.size();
      if (!cnt_possible_moves)
        break;
      if (b.make_move(moves[curand(&curand_state) % cnt_possible_moves])) {
        hash_history.clear();
        goto replay;
      }
      hash_history.push_back(b.get_hash());
    }
    if (b.can_eat_king()) {
      hash_history.clear();
      goto replay;
    }

    while (true) {
      PositionRecord rec(true);
      bool flip = b.side2move == BLACK;
      get_fen(b, rec.fen);
      int pov = b.side2move == WHITE ? 1 : -1;  // il motore valuta per chi deve muovere. Moltiplichiamo per avere il pov del Bianco.
      nnue_state_t<use_new> nnue_state(b);
      rec.evals[0] = b.hc_heuristic_score<use_new>(nnue_state) * pov;
      move best_m;
      for(int d = 1; d <= depth; ++d) {
        move current_m;
        int score = compute_alphabeta_2<use_new, true, 0>(b, current_m, d, -inf, +inf, curand_state, hash_history, very_local_tt, local_killers, nnue_state);
        rec.evals[d] = score * pov;
        if (current_m.i0 != -1)
          best_m = current_m;
      }
      game_history[game_length] = rec;
      if (best_m.i0 == -1) {
        if (b.is_check())
          outcome = b.side2move == WHITE ? -1 : 1;
        break;
      }
      if (game_length == max_game_length - 2) {
        outcome = 0;
        break;
      }
      b.make_move(best_m);
      /*if (cuda::std::find(hash_history.begin(), hash_history.end(), b.get_hash()) != hash_history.end()) {
        outcome = 0;
        break;
      }*/
      bool found = false;
      for (uint64_t h : hash_history) {
        if (b.get_hash() == h) {
          found = true;
          break;
        }
      }
      if (found) {
        outcome = 0;
        break;
      }
      hash_history.push_back(b.get_hash());
      game_length++;
      if (id < 256)
        printf("thread %d game_length %d\n", id, game_length);
    }
    assert(game_length <= max_game_length - 1);  // we need nullpos in the end as our "\0"

    for (int i = 0; i < game_length; ++i) {
      game_history[i].game_outcome = outcome;
      for (int j = 0; j < 3; ++j) {
        int off = j == 0 ? 5 : j == 1 ? 10 : 20;
        game_history[i].offset_vals[j] = i + off < game_length ? game_history[i + off].evals[0] : outcome * 10000;
      }
    }
    if (id <= 3)
      printf("%d\n",game_length);
    for (int i = 0; i <= game_length; ++i)
      put_data[id][i] = i < game_length ? game_history[i] : PositionRecord();
  }
}

int main(int argc, const char **argv) {
  std::cout << "STARTING" << std::endl;
  gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 120000/*120000*/));
  std::cout << "SUCCESSFULLY INCREASED STACK SIZE" << std::endl;
  board::precalc(57);
  precalc_heuristic_eval();
  precalc_nn_eval();

  bool existed = std::filesystem::exists(filename);
  std::cout << (existed ? "APPENDING TO EXISTING DATASET" : "CREATING A NEW DATASET") << std::endl;
  std::ofstream out_csv(filename, std::ios::app);
  if (!existed) {
    out_csv << "fen";
    for (int d = 0; d <= depth; ++d)
      out_csv << ",eval_d" << d;
    out_csv << ",eval_5,eval_10,eval_20,outcome\n";
  }

  PositionRecord (*rollouts_data)[max_game_length];
  gpuErrchk(cudaMallocManaged(&rollouts_data, k12 * max_game_length * sizeof(PositionRecord)));
  std::mt19937 gen(time(0));
  std::cout << "Avviati " << k12 << " thread. Raccolta dati in corso..." << std::endl;
  worker_thread<<<k1,k2>>>(games_per_thread, rollouts_data, gen());
  double t0 = get_wall_time();
  gpuErrchk(cudaDeviceSynchronize());
  std::cout << "took " << get_wall_time()-t0 << " seconds" << std::endl;
  std::cout << "Now writing to file" << std::endl;
  t0 = get_wall_time();
  for (int game_i = 0; game_i < k12; ++game_i) {
    for (const PositionRecord& rec : rollouts_data[game_i]) {
      if (!rec.nonempty)
        break;
      std::string fen;
      for (char c : rec.fen) {
        if (!c)
          break;
        fen += c;
      }
      out_csv << fen;
      for (int d = 0; d <= depth; ++d)
        out_csv << "," << rec.evals[d];
      int offsets[3] = {5, 10, 20};
      for (int val : rec.offset_vals)
        out_csv << "," << val;
      out_csv << "," << (int)rec.game_outcome << "\n";
    }
  }
  out_csv.flush();
  std::cout << "Done writing to file (" << get_wall_time()-t0 << "s)" << std::endl;
  return 0;
}
