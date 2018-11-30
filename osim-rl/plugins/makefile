cc=clang++
LINK=ln
BUILDDIR = .
SRCDIR = .
INCDIR = .
INSTALLDIR = /bin
CPPFLAGS = -I$(INCDIR) -std=c++14  -O2  $(PROFILE) $(ADDCPPFLAGS) -fPIC -I~/anaconda3/envs/opensim-rl_modded/include -I~/anaconda3/envs/opensim-rl_modded/include/simbody
LINKFLAGS = -L`pip3 show opensim |grep Location | cut -d' ' -f2`/opensim -l:_simulation.so

all:
	$(cc) $(LINKFLAGS) $(CPPFLAGS) -undefined dynamic_lookup `python3 -m pybind11 --includes`  -shared $(SRCDIR)/my_manager_factory.cpp -o $(BUILDDIR)/my_manager_factory`python3-config --extension-suffix` 
