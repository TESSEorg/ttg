#include <stdlib.h>  // std::atoi, std::rand()
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../blockmatrix.h"

#define TABLE_SIZE 524288
#define CHUNK 1024
#define CHUNKBIG (32 * CHUNK)

#include "ttg.h"
using namespace ttg;

typedef struct params {
  long LocalTableSize; /* local size of the table may be rounded up >= MinLocalTableSize */
  long ProcNumUpdates; /* usually 4 times the local size except for time-bound runs */

  unsigned long logTableSize;      /* it is an unsigned 64-bit value to type-promote expressions */
  unsigned long TableSize;         /* always power of 2 */
  unsigned long MinLocalTableSize; /* TableSize/NumProcs */
  unsigned long GlobalStartMyProc; /* first global index of the global table stored locally */
  unsigned long Top;               /* global indices below 'Top' are asigned in MinLocalTableSize+1 blocks;
                                      above 'Top' -- in MinLocalTableSize blocks */

  int logNumProcs, NumProcs, MyProc;

  int Remainder; /* TableSize % NumProcs */
} params_t;

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#define ZERO64B 0L

class RandomData {
 private:
  int size;                             // size of the array
  int n;                                // Number of non-zero elements
  std::shared_ptr<unsigned long> data;  // Enables shallow copy for shared memory settings.
                                        // std::vector<unsigned long> data;

 public:
  RandomData() = default;
  RandomData(int size) : size(size) {
    data = std::shared_ptr<unsigned long>(new unsigned long[size], [](unsigned long *p) { delete[] p; });
  }

  ~RandomData() {}

  // Total size of the data array
  int getsize() const { return size; }
  // Count of non-zero elements
  int getn() const { return n; }
  // Set the count of non-zero elements
  void setn(int i) { n = i; }
  const unsigned long *get() const { return data.get(); }
  unsigned long *get() { return data.get(); }

  unsigned long &operator()(int i) { return data.get()[i]; }

  /*template<typename Archive>
    void serialize(Archive& ar) {}*/
};

namespace madness {
  namespace archive {
    template <class Archive>
    struct ArchiveStoreImpl<Archive, RandomData> {
      static inline void store(const Archive &ar, const RandomData &d) {
        ar << d.getsize() << d.getn();
        ;
        ar << wrap(d.get(), d.getsize());
      }
    };

    template <class Archive>
    struct ArchiveLoadImpl<Archive, RandomData> {
      static inline void load(const Archive &ar, RandomData &d) {
        int size, n;
        ar >> size >> n;
        d = RandomData(size);
        d.setn(n);
        ar >> wrap(d.get(), d.getsize());
      }
    };
  }  // namespace archive
}  // namespace madness

using Key = std::pair<std::pair<int, int>, int>;  // Proc#, #iterations, #logNumProcs to exchange data

/* Utility routine to start random number generator at Nth step */
unsigned long HPCC_starts(long n) {
  int i, j;
  unsigned long m2[64];
  unsigned long temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((long)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((long)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1) break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1) temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) ran = (ran << 1) ^ ((long)ran < 0 ? POLY : 0);
  }

  return ran;
}

// Kick start the computation by generating a random number and instantiating the data arrays.
auto make_start(params_t &tparams, Edge<Key, unsigned long> rand_input_edge, Edge<Key, RandomData> &main_iter_data_edge,
                Edge<Key, RandomData> &input_send_edge) {
  auto f = [tparams](const Key &key,
                     std::tuple<Out<Key, unsigned long>, Out<Key, RandomData>, Out<Key, RandomData>> &out) {
    RandomData data(CHUNKBIG);
    RandomData tosend(CHUNKBIG);

    unsigned long ran = HPCC_starts(4 * tparams.GlobalStartMyProc);
    send<0>(key, ran, out);
    send<1>(key, data, out);
    send<2>(key, tosend, out);
  };

  return make_tt<Key>(f, edges(), edges(rand_input_edge, main_iter_data_edge, input_send_edge), "Start", {},
                      {"random input edge", "main iteration data edge", "input send edge"});
}

// Generates random data for exchanging with other processes.
auto make_randomgen_op(params_t &tparams, Edge<Key, unsigned long> rand_input_edge,
                       Edge<Key, RandomData> &main_iter_data_edge, Edge<Key, RandomData> &input_send_edge,
                       Edge<Key, RandomData> &process_data_edge, Edge<Key, RandomData> &process_send_edge) {
  auto f = [tparams](const Key &key, unsigned long ran, RandomData &&data, RandomData &&tosend,
                     std::tuple<Out<Key, unsigned long>, Out<Key, RandomData>, Out<Key, RandomData>> &out) {
    int niterate = tparams.ProcNumUpdates / CHUNK;
    auto [myprocnum, iter] = key.first;
    // std::cout << "randomgen total iterations : " << niterate << " iteration : " << iter << ", rank :" <<
    // ttg::default_execution_context().rank() << std::endl;

    if (iter < niterate) {
      int i, j, logTableLocal, ipartner;
      int ndata, nkeep, nsend, nrecv;
      unsigned long niterate, datum, procmask, nlocalm1, index;

      logTableLocal = tparams.logTableSize - tparams.logNumProcs;
      nlocalm1 = (unsigned long)(tparams.LocalTableSize - 1);
      // std::cout << "GlobalStartMyProc : " << tparams.GlobalStartMyProc << ", ran : " << ran << ", logTableLocal : "
      // << logTableLocal << ", nlocalm1 : " << nlocalm1 << std::endl;

      for (i = 0; i < CHUNK; i++) {
        ran = (ran << 1) ^ ((long)ran < ZERO64B ? POLY : ZERO64B);
        data(i) = ran;
      }
      // std::cout << "ran : " << ran << std::endl;
      data.setn(CHUNK);

      // std::cout << "generated data array with " << data.getn() << " elements\n";
      send<1>(std::make_pair(std::make_pair(myprocnum, iter), 0), data, out);
      send<2>(std::make_pair(std::make_pair(myprocnum, iter), 0), tosend, out);
      if (iter < tparams.ProcNumUpdates / CHUNK - 1)
        send<0>(std::make_pair(std::make_pair(myprocnum, iter + 1), 0), ran, out);
    }
  };

  return make_tt(f, edges(rand_input_edge, main_iter_data_edge, input_send_edge),
                 edges(rand_input_edge, process_data_edge, process_send_edge), "generator",
                 {"rand input edge", "main_iterator_data_edge", "input_send_edge"},
                 {"rand recur edge", "process_data_edge", "process_send_edge"});
}

// Exchanges data with other processses (Send part of MPI_SendRecv)
auto make_processdata_op(params_t &tparams, Edge<Key, RandomData> &process_data_edge,
                         Edge<Key, RandomData> &process_send_edge, Edge<Key, RandomData> &keep_data_edge,
                         Edge<Key, RandomData> &send_data_edge, Edge<Key, RandomData> &send_to_other_data_edge,
                         Edge<Key, RandomData> &direct_update_data_edge, Edge<Key, RandomData> &forward_send_edge) {
  auto f = [tparams](const Key &key, RandomData &&data, RandomData &&tosend,
                     std::tuple<Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, RandomData>,
                                Out<Key, RandomData>> &out) {
    auto [first, second] = key;
    auto [myprocnum, iter] = first;
    auto j = second;

    // std::cout << "processdata iteration : " << j << ", logNumProcs : " << tparams.logNumProcs << ", rank :" <<
    // ttg::default_execution_context().rank() << std::endl;
    if (j < tparams.logNumProcs) {
      int logTableLocal, ipartner;
      int ndata, nkeep, nsend, nrecv;
      unsigned long datum, procmask, nlocalm1, index;

      logTableLocal = tparams.logTableSize - tparams.logNumProcs;
      nlocalm1 = (unsigned long)(tparams.LocalTableSize - 1);
      ndata = data.getn();
      // j:value, myproc->ipartner:values
      // j:0, 0->1,1->1,2->1,3->1
      // j:1, 0->1,1->2,2->4,3->8
      // j:2, 0->1,1->4,2->8,3->64
      // j:3, 0->1,1->8,2->64,3->512

      //    for (j = 0; j < tparams.logNumProcs; j++) {
      nkeep = nsend = 0;
      ipartner = (1 << j) ^ tparams.MyProc;
      procmask = ((unsigned long)1) << (logTableLocal + j);
      if (ipartner > tparams.MyProc) {
        for (int i = 0; i < ndata; i++) {
          if (data(i) & procmask)
            tosend(nsend++) = data(i);
          else
            data(nkeep++) = data(i);
        }
      } else {
        for (int i = 0; i < ndata; i++) {
          if (data(i) & procmask)
            data(nkeep++) = data(i);
          else
            tosend(nsend++) = data(i);
        }
      }

      // std::cout << "Sending processed data" << " with " << nsend << " elements from " <<
      // ttg::default_execution_context().rank()
      //<< " to pids : " << ipartner << " for j = " << j << std::endl;
      tosend.setn(nsend);
      send<1>(std::make_pair(key.first, j), tosend, out);
      send<2>(std::make_pair(std::make_pair(ipartner, key.first.second), j), tosend, out);
      // std::cout << "Forward my data with " << nkeep << " elements from P#" << ttg::default_execution_context().rank()
      // << "\n";
      data.setn(nkeep);
      send<0>(std::make_pair(key.first, j), data, out);
    } else {
      // std::cout << "no processing needed, sending data arrays with " << data.getn() << " and " << tosend.getn() << "
      // elements\n";
      send<3>(std::make_pair(key.first, j), data, out);
      send<4>(std::make_pair(key.first, j), tosend, out);
    }
  };

  return make_tt(
      f, edges(process_data_edge, process_send_edge),
      edges(keep_data_edge, send_data_edge, send_to_other_data_edge, direct_update_data_edge, forward_send_edge),
      "Process Data", {"process data edge", "process send edge"},
      {"keep data edge", "send data edge", "send to other Ps data edge", "direct update data edge",
       "direct update send edge"});
}

// Exchanges data with other processes (Recv part of MPI_SendRecv)
auto make_receivedata_op(params_t &tparams, Edge<Key, RandomData> &keep_data_edge,
                         Edge<Key, RandomData> &send_data_edge, Edge<Key, RandomData> &send_to_other_data_edge,
                         Edge<Key, RandomData> &process_data_edge, Edge<Key, RandomData> &update_data_edge,
                         Edge<Key, RandomData> &forward_send_edge, Edge<Key, RandomData> &process_send_edge) {
  auto f =
      [tparams](
          const Key &key, RandomData &&keep_data, RandomData &&my_send_data, RandomData &&other_data,
          std::tuple<Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, RandomData>> &out) {
        auto [first, second] = key;
        auto [myprocnum, iter] = first;
        auto j = second;

        int nkeep = keep_data.getn();
        int nrecv = other_data.getn();
        int ndata = nkeep + nrecv;

        // std::cout << "receivedata j : " << j << ", rank :" << ttg::default_execution_context().rank() << " with " <<
        // nrecv << " elements" << " and my_data with " << nkeep << " elements" << std::endl; Copy other_data into
        // my_data array
        for (int count = 0; count < nrecv; count++) keep_data(nkeep++) = other_data(count);

        keep_data.setn(ndata);
        send<0>(std::make_pair(first, j + 1), keep_data, out);

        if (j == tparams.logNumProcs) {
          send<2>(key, keep_data, out);
          send<3>(key, my_send_data, out);
        }

        send<1>(std::make_pair(first, j + 1), my_send_data, out);
      };

  return make_tt(f, edges(keep_data_edge, send_data_edge, send_to_other_data_edge),
                 edges(process_data_edge, process_send_edge, update_data_edge, forward_send_edge),
                 "Receive Data from other Ps", {"keep_data", "my_send_data", "recv_data"},
                 {"process_data_edge", "process_send_edge", "update_data_edge", "update_send_edge"});
}

// Update the data array and continue to next iteration
auto make_randomupdate_op(std::vector<unsigned long> &table, params_t &tparams,
                          Edge<Key, RandomData> &direct_update_data_edge, Edge<Key, RandomData> &update_data_edge,
                          Edge<Key, RandomData> &forward_send_edge, Edge<Key, RandomData> &main_iter_data_edge,
                          Edge<Key, RandomData> &input_send_edge,
                          Edge<Key, std::vector<unsigned long>> &result_data_edge) {
  auto f = [&table, tparams](
               const Key &key, RandomData &&my_data, RandomData &&my_send_data,
               std::tuple<Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, std::vector<unsigned long>>> &out) {
    auto [myprocnum, iter] = key.first;
    int ndata = my_data.getn();
    unsigned long datum, nlocalm1;
    int index;
    // std::cout << "randomupdate iteration : " << iter << ", rank :" << ttg::default_execution_context().rank() <<
    // std::endl;
    nlocalm1 = (unsigned long)(tparams.LocalTableSize - 1);
    // std::cout << "Data length : " << ndata << " nlocalm1 : " << nlocalm1 << std::endl;
    for (int i = 0; i < ndata; i++) {
      datum = my_data(i);
      index = datum & nlocalm1;
      table[index] ^= datum;
    }

    if (iter == tparams.ProcNumUpdates / CHUNK - 1)
      send<2>(Key(std::make_pair(myprocnum, iter), 0), table, out);
    else {
      // std::cout << "Iteration:" << iter << " done for process " << ttg::default_execution_context().rank() << "\n";
      send<0>(Key(std::make_pair(myprocnum, iter + 1), 0), my_data, out);
      send<1>(Key(std::make_pair(myprocnum, iter + 1), 0), my_send_data, out);
    }
  };

  return make_tt(f, edges(fuse(direct_update_data_edge, update_data_edge), forward_send_edge),
                 edges(main_iter_data_edge, input_send_edge, result_data_edge), "Random Update TT",
                 {"update_edge", "forward_send_edge"}, {"main_iter_edge", "input_send_edge", "result_edge"});
}

#define NUPDATE (4 * TABLE_SIZE)

// TODO: Original MPI code needs to be run in order to validate the result.
auto make_verifyresult(params_t &tparams,  // std::vector<unsigned long> &verify_table,
                       Edge<Key, std::vector<unsigned long>> &result_edge) {
  auto f = [tparams](const Key &key, std::vector<unsigned long> &&result_table, std::tuple<> &out) {
    /* Verification of results (in serial or "safe" mode; optional) */

    // std::cout << "Verifying results...\n";
    /*unsigned long i, temp;
      temp = 0;
      for (i=0; i<tparams.LocalTableSize; i++)
      if (result_table[i] != verify_table[i])
      temp++;

      std::cout << temp << " errors found in process " << ttg::default_execution_context().rank()
      << " : " << ((temp <= 0.01*TABLE_SIZE) ? "PASSED!" : "FAILED!") << std::endl;*/
  };

  return make_tt<Key>(f, edges(result_edge), edges(), "Verify TT", {"result_edge"}, {});
}

// Original GUPS Random Access benchmark code
void Power2NodesMPIRandomAccessUpdateVerfy(std::vector<unsigned long> verify_table, params_t tparams) {
  int i, j, logTableLocal, ipartner;
  int ndata, nkeep, nsend, nrecv;
  long iterate, niterate;
  unsigned long ran, datum, procmask, nlocalm1, index;
  unsigned long *data, *send;
  MPI_Status status;

  /* setup: should not really be part of this timed routine */

  data = (unsigned long *)malloc(CHUNKBIG * sizeof(unsigned long));
  send = (unsigned long *)malloc(CHUNKBIG * sizeof(unsigned long));

  ran = HPCC_starts(4 * tparams.GlobalStartMyProc);

  niterate = tparams.ProcNumUpdates / CHUNK;
  logTableLocal = tparams.logTableSize - tparams.logNumProcs;
  nlocalm1 = (unsigned long)(tparams.LocalTableSize - 1);

  /* actual update loop: this is only section that should be timed */

  for (iterate = 0; iterate < niterate; iterate++) {
    for (i = 0; i < CHUNK; i++) {
      ran = (ran << 1) ^ ((long)ran < ZERO64B ? POLY : ZERO64B);
      data[i] = ran;
    }
    ndata = CHUNK;
    // j:value, myproc->ipartner:values
    // j:0, 0->1,1->1,2->1,3->1
    // j:1, 0->1,1->2,2->4,3->8
    // j:2, 0->1,1->4,2->8,3->64
    // j:3, 0->1,1->8,2->64,3->512
    for (j = 0; j < tparams.logNumProcs; j++) {
      nkeep = nsend = 0;
      ipartner = (1 << j) ^ tparams.MyProc;
      procmask = ((unsigned long)1) << (logTableLocal + j);
      if (ipartner > tparams.MyProc) {
        for (i = 0; i < ndata; i++) {
          if (data[i] & procmask)
            send[nsend++] = data[i];
          else
            data[nkeep++] = data[i];
        }
      } else {
        for (i = 0; i < ndata; i++) {
          if (data[i] & procmask)
            data[nkeep++] = data[i];
          else
            send[nsend++] = data[i];
        }
      }

      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      MPI_Sendrecv(send, nsend, MPI_UNSIGNED_LONG, ipartner, 0, &data[nkeep], CHUNKBIG, MPI_UNSIGNED_LONG, ipartner, 0,
                   MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nrecv);
      ndata = nkeep + nrecv;
    }

    for (i = 0; i < ndata; i++) {
      datum = data[i];
      index = datum & nlocalm1;
      verify_table[index] ^= datum;
    }
  }

  /* clean up: should not really be part of this timed routine */

  free(data);
  free(send);
}

int main(int argc, char *argv[]) {
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  // std::vector<unsigned long> table;
  params_t tparams;

  initialize(argc, argv, -1);

  tparams.NumProcs = ttg::default_execution_context().size();
  ;
  tparams.logTableSize = 19;
  tparams.TableSize = TABLE_SIZE;
  tparams.logNumProcs = log2(tparams.NumProcs);
  // tparams.ProcNumUpdates = 4 * tparams.TableSize;
  tparams.MinLocalTableSize = (tparams.TableSize / tparams.NumProcs);
  tparams.LocalTableSize = tparams.MinLocalTableSize;
  tparams.ProcNumUpdates = 4 * tparams.LocalTableSize;
  tparams.MyProc = ttg::default_execution_context().rank();
  tparams.GlobalStartMyProc = (tparams.MinLocalTableSize * ttg::default_execution_context().rank());

  std::vector<unsigned long> table(tparams.LocalTableSize);
  // std::vector<unsigned long> verify_table(tparams.LocalTableSize);

  Edge<Key, unsigned long> rand_input_edge;
  Edge<Key, RandomData> main_iter_data_edge, input_send_edge;
  Edge<Key, RandomData> process_data_edge, process_send_edge;
  Edge<Key, RandomData> keep_data_edge, send_data_edge, direct_update_edge;
  Edge<Key, RandomData> send_to_other_data_edge, forward_send_edge, update_data_edge;
  Edge<Key, std::vector<unsigned long>> result_data_edge;

  // unsigned long ran = HPCC_starts(4 * tparams.GlobalStartMyProc);
  // std::cout << "ran : " << ran << ", GlobalStartMyProc : " << tparams.GlobalStartMyProc << std::endl;
  // table = (unsigned long*)malloc(sizeof(unsigned long) * tparams.LocalTableSize );
  for (unsigned long i = 0; i < tparams.LocalTableSize; i++) {
    table[i] = i + tparams.GlobalStartMyProc;
    // verify_table[i] = tparams.GlobalStartMyProc;
  }

  // For verification
  // Power2NodesMPIRandomAccessUpdateVerfy(verify_table, tparams);

  auto r0 = make_start(tparams, rand_input_edge, main_iter_data_edge, input_send_edge);
  auto r1 = make_randomgen_op(tparams, rand_input_edge, main_iter_data_edge, input_send_edge, process_data_edge,
                              process_send_edge);
  auto r2 = make_processdata_op(tparams, process_data_edge, process_send_edge, keep_data_edge, send_data_edge,
                                send_to_other_data_edge, direct_update_edge, forward_send_edge);
  auto r3 = make_receivedata_op(tparams, keep_data_edge, send_data_edge, send_to_other_data_edge, process_data_edge,
                                update_data_edge, forward_send_edge, process_send_edge);
  auto r4 = make_randomupdate_op(table, tparams, direct_update_edge, update_data_edge, forward_send_edge,
                                 main_iter_data_edge, input_send_edge, result_data_edge);
  auto r5 = make_verifyresult(tparams, result_data_edge);

  auto keymap = [](const Key &key) { return key.first.first; };
  r0->set_keymap(keymap);
  r1->set_keymap(keymap);
  r2->set_keymap(keymap);
  r3->set_keymap(keymap);
  r4->set_keymap(keymap);

  auto connected = make_graph_executable(r0.get());
  assert(connected);
  TTGUNUSED(connected);
  // std::cout << "Graph is connected.\n";

  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "==== begin dot ====\n";
    std::cout << Dot()(r0.get()) << std::endl;
    std::cout << "==== end dot ====\n";

    std::cout << "#Procs : " << tparams.NumProcs << " logNumProcs : " << tparams.logNumProcs
              << " LocalTableSize : " << tparams.LocalTableSize << std::endl;
    beg = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < tparams.NumProcs; p++) {
      std::cout << "Invoking for process " << p << std::endl;
      r0->invoke(Key(std::make_pair(p, 0), 0));
    }
  }

  execute();
  fence();
  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Total Main table size = 2^" << tparams.logTableSize << " = " << tparams.TableSize << " words\n";
    std::cout << "PE Main table size = 2^" << (tparams.logTableSize - tparams.logNumProcs) << " = "
              << tparams.TableSize / tparams.NumProcs << " words/PE ---- " << tparams.logNumProcs << "\n";

    std::cout << "Number of updates EXECUTED = " << 4 * tparams.TableSize << "\n";
    auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000000;
    std::cout << "TTG Execution Time (seconds) : " << elapsed << std::endl;
    double GUPs = 1e-9 * 4 * tparams.TableSize / elapsed;
    std::cout << GUPs << " Billion(10^9) Updates    per second [GUP/s]\n";
    std::cout << (GUPs / tparams.NumProcs) << " Billion(10^9) Updates/PE per second [GUP/s]\n";
  }

  ttg_finalize();
}
