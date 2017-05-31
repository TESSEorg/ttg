all: mxm_simple_suma t9

CXXFLAGS = -g -O0 -Wall $(subst c1x,c++14,$(shell pkg-config --cflags parsec))
LDFLAGS = $(shell pkg-config --libs parsec)

MADTOP = /home/rjh/Devel/madness

edge3: edge3.cc edge.h
	mpicxx -o edge3 $(CXXFLAGS) edge3.cc $(LDFLAGS)

edge3mad: edge3mad.cc edgemad.h edge.h
	mpicxx -DHAVE_CONFIG_H   -I$(MADTOP)/src -I$(MADTOP)/src -I$(MADTOP)/src/apps -D"MADNESS_GITREVISION=\"`git --git-dir=$(MADTOP)/.git rev-parse HEAD`\" " -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -DTBB_USE_DEBUG=1  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs -MT edge3mad.o -MD -MP  -c -o edge3mad.o edge3mad.cc &&\
	/bin/bash $(MADTOP)/libtool  --tag=CXX   --mode=link mpicxx  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs  -lmpfr -o edge3mad edge3mad.o $(MADTOP)/src/madness/mra/libMADmra.la $(MADTOP)/src/madness/tensor/libMADlinalg.la $(MADTOP)/src/madness/tensor/libMADtensor.la $(MADTOP)/src/madness/misc/libMADmisc.la $(MADTOP)/src/madness/external/muParser/libMADmuparser.la $(MADTOP)/src/madness/external/tinyxml/libMADtinyxml.la $(MADTOP)/src/madness/world/libMADworld.la  -ltbb_debug -lprofiler -ltcmalloc  -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm -ldl  

ttg: ttg.cc ttg-mad.h ttgbase.h
	mpicxx -DHAVE_CONFIG_H   -I$(MADTOP)/src -I$(MADTOP)/src -I$(MADTOP)/src/apps -D"MADNESS_GITREVISION=\"`git --git-dir=$(MADTOP)/.git rev-parse HEAD`\" " -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -DTBB_USE_DEBUG=1  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs -MT ttg.o -MD -MP  -c -o ttg.o ttg.cc &&\
	/bin/bash $(MADTOP)/libtool  --tag=CXX   --mode=link mpicxx  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs  -lmpfr -o ttg ttg.o $(MADTOP)/src/madness/mra/libMADmra.la $(MADTOP)/src/madness/tensor/libMADlinalg.la $(MADTOP)/src/madness/tensor/libMADtensor.la $(MADTOP)/src/madness/misc/libMADmisc.la $(MADTOP)/src/madness/external/muParser/libMADmuparser.la $(MADTOP)/src/madness/external/tinyxml/libMADtinyxml.la $(MADTOP)/src/madness/world/libMADworld.la  -ltbb_debug -lprofiler -ltcmalloc  -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm -ldl

t9-wrap-ttg: t9-wrap-ttg.cc ttg-mad.h ttgbase.h
	mpicxx -DHAVE_CONFIG_H   -I$(MADTOP)/src -I$(MADTOP)/src -I$(MADTOP)/src/apps -D"MADNESS_GITREVISION=\"`git --git-dir=$(MADTOP)/.git rev-parse HEAD`\" " -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -DTBB_USE_DEBUG=1  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs -MT ttg.o -MD -MP  -c -o t9-wrap-ttg.o t9-wrap-ttg.cc &&\
	/bin/bash $(MADTOP)/libtool  --tag=CXX   --mode=link mpicxx  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs  -lmpfr -o t9-wrap-ttg t9-wrap-ttg.o $(MADTOP)/src/madness/mra/libMADmra.la $(MADTOP)/src/madness/tensor/libMADlinalg.la $(MADTOP)/src/madness/tensor/libMADtensor.la $(MADTOP)/src/madness/misc/libMADmisc.la $(MADTOP)/src/madness/external/muParser/libMADmuparser.la $(MADTOP)/src/madness/external/tinyxml/libMADtinyxml.la $(MADTOP)/src/madness/world/libMADworld.la  -ltbb_debug -lprofiler -ltcmalloc  -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm -ldl

mxm_simple_suma_ttg: mxm_simple_suma_ttg.cc ttg-mad.h ttgbase.h
	mpicxx -DHAVE_CONFIG_H   -I$(MADTOP)/src -I$(MADTOP)/src -I$(MADTOP)/src/apps -D"MADNESS_GITREVISION=\"`git --git-dir=$(MADTOP)/.git rev-parse HEAD`\" " -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -DTBB_USE_DEBUG=1  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs -MT ttg.o -MD -MP  -c -o mxm_simple_suma_ttg.o mxm_simple_suma_ttg.cc &&\
	/bin/bash $(MADTOP)/libtool  --tag=CXX   --mode=link mpicxx  -g -O0 -g -Wall -Wno-strict-aliasing -Wno-deprecated -Wno-unused-local-typedefs  -lmpfr -o mxm_simple_suma_ttg mxm_simple_suma_ttg.o $(MADTOP)/src/madness/mra/libMADmra.la $(MADTOP)/src/madness/tensor/libMADlinalg.la $(MADTOP)/src/madness/tensor/libMADtensor.la $(MADTOP)/src/madness/misc/libMADmisc.la $(MADTOP)/src/madness/external/muParser/libMADmuparser.la $(MADTOP)/src/madness/external/tinyxml/libMADtinyxml.la $(MADTOP)/src/madness/world/libMADworld.la  -ltbb_debug -lprofiler -ltcmalloc  -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm -ldl

clean:
	rm -rf *.o mxm_simple_suma t9 edge3mad ttg t9-wrap-ttg mxm_simple_suma_ttg
