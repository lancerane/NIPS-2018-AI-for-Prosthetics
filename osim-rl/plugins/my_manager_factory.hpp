#include <OpenSim/Simulation/Manager/Manager.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct my_manager_factory{
    enum integrator{
        CPodes = 0,
        ExplicitEuler,
        RungeKutta2,
        RungeKutta3,
        RungeKuttaFeldberg,
        RungeKuttaMerson,
        SemiExplicitEuler2,
        SemiExplicitEuler,
        Verlet
    };
    static void create(py::object *model,py::object *manager, integrator type, double accuracy);
};
