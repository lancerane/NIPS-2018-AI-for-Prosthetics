#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "my_manager_factory.hpp"

struct PySwigObject {
    PyObject_HEAD
    void * ptr;
    const char * desc;
};
void* extract_swig_wrapped_pointer(PyObject* obj)
{
    char thisStr[] = "this";
    //first we need to get the this attribute from the Python Object
    if (!PyObject_HasAttrString(obj, thisStr))
        return NULL;

    PyObject* thisAttr = PyObject_GetAttrString(obj, thisStr);
    if (thisAttr == NULL)
        return NULL;
    //This Python Object is a SWIG Wrapper and contains our pointer
    void* pointer = ((PySwigObject*)thisAttr)->ptr;
    Py_DECREF(thisAttr);
    return pointer;
}
void replace_swig_wrapped_pointer(PyObject* obj, void *ptr)
{
    char thisStr[] = "this";
    //first we need to get the this attribute from the Python Object
    if (!PyObject_HasAttrString(obj, thisStr))
        return ;

    PyObject* thisAttr = PyObject_GetAttrString(obj, thisStr);
    if (thisAttr == NULL)
        return ;
    //This Python Object is a SWIG Wrapper and contains our pointer
    ((PySwigObject*)thisAttr)->ptr = ptr;
    Py_DECREF(thisAttr);
}

void my_manager_factory::create(py::object *model,py::object *manager, integrator type, double accuracy){
    OpenSim::Model* m = (OpenSim::Model*)extract_swig_wrapped_pointer(model->ptr());
    //py::object p = py::reinterpret_borrow<py::object>(model->ptr());
    SimTK::Integrator *integ;
    switch (type){
        case CPodes:
            integ = new SimTK::CPodesIntegrator (m->getMultibodySystem());
            break;
        case ExplicitEuler:
            integ = new SimTK::ExplicitEulerIntegrator (m->getMultibodySystem());
            break;
        case RungeKutta2:
            integ = new SimTK::RungeKutta2Integrator (m->getMultibodySystem());
            break;
        case RungeKutta3:
            integ = new SimTK::RungeKutta3Integrator (m->getMultibodySystem());
            break;
        case RungeKuttaFeldberg:
            integ = new SimTK::RungeKuttaFeldbergIntegrator (m->getMultibodySystem());
            break;
        case RungeKuttaMerson:
            integ = new SimTK::RungeKuttaMersonIntegrator (m->getMultibodySystem());
            break;
        case SemiExplicitEuler2:
            integ = new SimTK::SemiExplicitEuler2Integrator (m->getMultibodySystem());
            break;
        case SemiExplicitEuler:
            integ = new SimTK::SemiExplicitEulerIntegrator (m->getMultibodySystem(),accuracy);
            break;
        case Verlet:
            integ = new SimTK::VerletIntegrator (m->getMultibodySystem());
            break;
        default:
            integ = new SimTK::RungeKuttaMersonIntegrator (m->getMultibodySystem());
            break;
    }
    //integ->setAllowInterpolation(true);
    integ->setAccuracy(accuracy);
    OpenSim::Manager* manptr = new OpenSim::Manager(*m,*integ);
    replace_swig_wrapped_pointer(manager->ptr(),(void*)manptr);
}



PYBIND11_MODULE(my_manager_factory, m) {
    py::enum_<my_manager_factory::integrator>(m,"integrator")
        .value("CPodes",my_manager_factory::integrator::CPodes)
        .value("ExplicitEuler",my_manager_factory::integrator::ExplicitEuler)
        .value("RungeKutta2",my_manager_factory::integrator::RungeKutta2)
        .value("RungeKutta3",my_manager_factory::integrator::RungeKutta3)
        .value("RungeKuttaFeldberg",my_manager_factory::integrator::RungeKuttaFeldberg)
        .value("RungeKuttaMerson",my_manager_factory::integrator::RungeKuttaMerson)
        .value("SemiExplicitEuler2",my_manager_factory::integrator::SemiExplicitEuler2)
        .value("SemiExplicitEuler",my_manager_factory::integrator::SemiExplicitEuler)
        .value("Verlet",my_manager_factory::integrator::Verlet)
        .export_values();
    py::class_<my_manager_factory>(m, "my_manager_factory")
        .def(py::init<>())
        .def_static("create", &my_manager_factory::create);
}

