#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdlib.h> // std::atoi, std::rand()
#include <iomanip>
#include <string>
#include <memory>
#include "blockmatrix.h"

#define CHUNK    1024
#define CHUNKBIG (32*CHUNK)

typedef struct params {
  int64_t LocalTableSize; /* local size of the table may be rounded up >= MinLocalTableSize */
  int64_t ProcNumUpdates; /* usually 4 times the local size except for time-bound runs */

  uint64_t logTableSize;   /* it is an unsigned 64-bit value to type-promote expressions */
  uint64_t TableSize;      /* always power of 2 */
  uint64_t MinLocalTableSize; /* TableSize/NumProcs */
  uint64_t GlobalStartMyProc; /* first global index of the global table stored locally */
  uint64_t Top; /* global indices below 'Top' are asigned in MinLocalTableSize+1 blocks;
                 above 'Top' -- in MinLocalTableSize blocks */

  int logNumProcs, NumProcs, MyProc;

  int Remainder; /* TableSize % NumProcs */
} params_t;

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#define ZERO64B 0L

class RandomData {
 private:
  int size; //size of the array
  int n; //Number of non-zero elements
  std::shared_ptr<uint64_t> data;

 public:
  RandomData() = default;
  RandomData(int size) : size(size) {
    data = std::shared_ptr<uint64_t>(new uint64_t[size], [](uint64_t* p) { delete[] p; });
  }

  ~RandomData() {}

  int getsize() const { return size; }
  int getn() const { return n; }
  void setn(int i) { n = i; }
  const uint64_t* get() const { return data.get(); }
  uint64_t* get() { return data.get(); }

  uint64_t& operator() (int i) { return data.get()[i]; }
};

namespace madness {
  namespace archive {
    template <class Archive>
    struct ArchiveStoreImpl<Archive, RandomData> {
      static inline void store(const Archive& ar, const RandomData& d) {
        ar << d.getsize() << d.getn();;
        ar << wrap(d.get(), d.getsize()); //BlockMatrix<T>(bm.rows(), bm.cols());
      }
    };

    template <class Archive>
    struct ArchiveLoadImpl<Archive, RandomData> {
      static inline void load(const Archive& ar, RandomData& d) {
        int size, n;
        ar >> size >> n;
        d = RandomData(size);
        ar >> wrap(d.get(), d.getsize());
      }
    };
  }
}

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

using Key = std::pair<std::pair<int, int>, int>; //Proc#, #iterations, #logNumProcs to exchange data

/* Utility routine to start random number generator at Nth step */
uint64_t
HPCC_starts(int64_t n)
{
  int i, j;
  uint64_t m2[64];
  uint64_t temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0);
  }

  return ran;
}

struct Control {
  template <typename Archive>
  void serialize(Archive& ar) {}
};

auto make_randomgen_op(params_t &tparams, Edge<Key, Control> main_iter, 
		       Edge<Key, RandomData> process_edge) {
  auto f = [tparams](const Key& key, Control& c, std::tuple<Out<Key, RandomData>> &out)
  {
    int niterate = tparams.ProcNumUpdates / CHUNK;
    auto [myprocnum, iter] = key.first;
    //std::cout << "randomgen total iterations : " << niterate << " iteration : " << iter << ", rank :" << ttg_default_execution_context().rank() << std::endl;

    if (iter < niterate) {
      int i, j, logTableLocal, ipartner;
      int ndata, nkeep, nsend, nrecv;
      uint64_t ran, niterate, datum, procmask, nlocalm1, index;
   
      RandomData data(CHUNKBIG);

      ran = HPCC_starts(4 * tparams.GlobalStartMyProc);
    
      logTableLocal = tparams.logTableSize - tparams.logNumProcs;
      nlocalm1 = (uint64_t)(tparams.LocalTableSize - 1);

      for (i = 0; i < CHUNK; i++) {
	      ran = (ran << 1) ^ ((int64_t) ran < ZERO64B ? POLY : ZERO64B);
	      data(i) = ran;
      }

      data.setn(CHUNK);
      
      //std::cout << "generated data array with " << data.getn() << " elements\n";
      send<0>(std::make_pair(std::make_pair(myprocnum, iter), 0), data, out);
    }
  };

  return wrap(f, edges(main_iter), edges(process_edge), "generator", {"main_iterator"}, 
                   {"process_edge"});
}

auto make_processdata_op(params_t &tparams, Edge<Key, RandomData> &process_edge, 
		       Edge<Key, RandomData> &keep_data, 
		       Edge<Key, RandomData> &send_data,
           Edge<Key, RandomData> &direct_edge)
{
  auto f = [tparams](const Key &key, RandomData &data, std::tuple<Out<Key, RandomData>, Out<Key, RandomData>, Out<Key, RandomData>> &out) {
    auto [first, second] = key;
    auto [myprocnum, iter] = first;
    auto j = second;

    //std::cout << "processdata iteration : " << j << ", logNumProcs : " << tparams.logNumProcs << ", rank :" << ttg_default_execution_context().rank() << std::endl;
    if (j < tparams.logNumProcs) {
      int logTableLocal, ipartner;
      int ndata, nkeep, nsend, nrecv;
      uint64_t datum, procmask, nlocalm1, index;
   
      RandomData tosend(CHUNKBIG);
    
      logTableLocal = tparams.logTableSize - tparams.logNumProcs;
      nlocalm1 = (uint64_t)(tparams.LocalTableSize - 1);

      //j:value, myproc->ipartner:values
      //j:0, 0->1,1->1,2->1,3->1
      //j:1, 0->1,1->2,2->4,3->8
      //j:2, 0->1,1->4,2->8,3->64
      //j:3, 0->1,1->8,2->64,3->512
      
      //    for (j = 0; j < tparams.logNumProcs; j++) {
      nkeep = nsend = 0;
      ipartner = (1 << j) ^ tparams.MyProc;
      procmask = ((uint64_t) 1) << (logTableLocal + j);
      if (ipartner > tparams.MyProc) {
	      for (int i = 0; i < ndata; i++) {
	        if (data(i) & procmask) tosend(nsend++) = data(i);
	        else data(nkeep++) = data(i);
	      }
      } else {
        for (int i = 0; i < ndata; i++) {
          if (data(i) & procmask) data(nkeep++) = data(i);
          else tosend(nsend++) = data(i);
	      }
      }

      //std::cout << "Sending processed data from " << ttg_default_execution_context().rank() << " to pids : " << ipartner << " " << j << std::endl;
      tosend.setn(nsend);
      send<1>(std::make_pair(std::make_pair(ipartner, key.first.second), j), tosend, out);

      data.setn(nkeep);
      send<0>(std::make_pair(key.first, j), data, out);
    } else {
      //std::cout << "no processing needed, sending data array with " << data.getn() << " elements\n";
      send<2>(std::make_pair(key.first, j), data, out);
    }
  }; 

  return wrap(f, edges(process_edge), edges(keep_data, send_data, direct_edge), "Process Data", {
              "process edge"}, {"keep data", "send data", "direct edge"}); 
}

auto make_receivedata_op(params_t &tparams, 
			  Edge<Key, RandomData> &keep_data,
			  Edge<Key, RandomData> &send_data,
			  Edge<Key, RandomData> &update_edge,
			  Edge<Key, RandomData> process_edge) {
  auto f = [tparams](const Key &key, 
	       RandomData &keep_data, RandomData &other_data, std::tuple<Out<Key, RandomData>, Out<Key, RandomData>> &out) {
    auto [first, second] = key;
    auto [myprocnum, iter] = first;
    auto j = second;

    int nkeep = keep_data.getn();
    int nrecv = other_data.getn();
    int ndata = nkeep + nrecv;
 
    //std::cout << "receivedata j : " << j << ", rank :" << ttg_default_execution_context().rank() << std::endl;   
    //Copy other_data into my_data array
    for (int count = 0; count < nrecv; count++)
      keep_data(nkeep++) = other_data(count);
    
    keep_data.setn(ndata);
    send<1>(std::make_pair(first, j + 1), keep_data, out); 
    
    if (j == tparams.logNumProcs)
      send<0>(key, keep_data, out);
  };
  
  return wrap(f, edges(keep_data, send_data), edges(update_edge, process_edge), "Receive Data from other Ps", 
              {"keep_data", "recv_data"}, {"update_edge", "process_edge"});
}

auto make_randomupdate_op(std::vector<uint64_t> &table, params_t &tparams, Edge<Key, RandomData> &direct_edge, 
        Edge<Key, RandomData> &update_edge,
			  Edge<Key, Control> main_iter) {
  auto f = [&table, tparams](const Key& key, RandomData &my_data, std::tuple<Out<Key, Control>> &out) {
    auto [myprocnum, iter] = key.first;
    int ndata = my_data.getn();
    uint64_t datum, nlocalm1;
    int index;
    //std::cout << "randomupdate iteration : " << iter << ", rank :" << ttg_default_execution_context().rank() << std::endl;
    nlocalm1 = (uint64_t)(tparams.LocalTableSize - 1);
    //std::cout << "Data length : " << ndata << " nlocalm1 : " << nlocalm1 << std::endl; 
    for (int i = 0; i < ndata; i++) {
      datum = my_data(i);
      index = datum & nlocalm1;
      //std::cout << "trying to update\n";
      table[index] ^= datum;
      //std::cout << "updated..." << index << "\n";
    }
    //std::cout << "Iteration:" << iter << " done for process " << ttg_default_execution_context().rank() << "\n";
    send<0>(Key(std::make_pair(myprocnum, iter+1), 0), Control(), out);
  };
  
  return wrap<Key>(f, edges(fuse(direct_edge,update_edge)), edges(main_iter), "Random Update Op", {"update_edge"}, {"main_iter_edge"});
}
 

int main(int argc, char* argv[]) {
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  //std::vector<uint64_t> table;
  params_t tparams;

  ttg_initialize(argc, argv, -1);

  tparams.NumProcs = ttg_default_execution_context().size();;
  tparams.logTableSize = 19;
  tparams.TableSize = 524288;
  tparams.logNumProcs = log2(tparams.NumProcs);
  tparams.ProcNumUpdates = 4 * tparams.TableSize;
  tparams.MinLocalTableSize = (tparams.TableSize / tparams.NumProcs);
  tparams.LocalTableSize = tparams.MinLocalTableSize;
  tparams.MyProc = ttg_default_execution_context().rank();
  tparams.GlobalStartMyProc = (tparams.MinLocalTableSize * ttg_default_execution_context().rank());  

  std::vector<uint64_t> table(tparams.LocalTableSize);

  Edge<Key, Control> main_iter; 
  Edge<Key, RandomData> process_edge;
  Edge<Key, RandomData> keep_data, send_data, direct_edge;
  Edge<Key, RandomData> update_edge;
		
  //table = (uint64_t*)malloc(sizeof(uint64_t) * tparams.LocalTableSize );
  for (int i = 0; i < tparams.LocalTableSize; i++)
  //for(const auto& t: table)
    table[i] = i + tparams.GlobalStartMyProc;
  
  auto r1 = make_randomgen_op(tparams, main_iter, process_edge);
  auto r2 = make_processdata_op(tparams, process_edge, keep_data, send_data, direct_edge);
  auto r3 = make_receivedata_op(tparams, keep_data, send_data, update_edge, process_edge);
  auto r4 = make_randomupdate_op(table, tparams, direct_edge, update_edge, main_iter);

  auto keymap = [](const Key& key) { return key.first.first; }; 
  r1->set_keymap(keymap);
  r2->set_keymap(keymap);
  r3->set_keymap(keymap);
  r4->set_keymap(keymap);

  auto connected = make_graph_executable(r1.get());
  assert(connected);
  TTGUNUSED(connected);
  //std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    std::cout << "==== begin dot ====\n";
    std::cout << Dot()(r1.get()) << std::endl;
    std::cout << "==== end dot ====\n";

    std::cout << "#Procs : " << tparams.NumProcs << " logNumProcs : " << tparams.logNumProcs << " LocalTableSize : " << tparams.LocalTableSize << std::endl;
    beg = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < tparams.NumProcs; p++) {
      std::cout << "Invoking for process " << p << std::endl;
      r1->invoke(Key(std::make_pair(p,0), 0), Control());
    }
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());
  if (ttg_default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Total Main table size = 2^" << tparams.logTableSize << " = " << tparams.TableSize << " words\n";
    std::cout << "PE Main table size = 2^" << (tparams.logTableSize - tparams.logNumProcs) << " = " 
              << tparams.TableSize/tparams.NumProcs << " words/PE ---- " << tparams.logNumProcs << "\n";

    std::cout << "Number of updates EXECUTED = " << tparams.ProcNumUpdates << "\n";
    auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000000;
    std::cout << "TTG Execution Time (seconds) : " << elapsed << std::endl;
    double GUPs = 1e-9 * tparams.ProcNumUpdates / elapsed;
    std::cout << GUPs << " Billion(10^9) Updates    per second [GUP/s]\n";
    std::cout << (GUPs / tparams.NumProcs) << " Billion(10^9) Updates/PE per second [GUP/s]\n";
  }

  ttg_finalize();
}
