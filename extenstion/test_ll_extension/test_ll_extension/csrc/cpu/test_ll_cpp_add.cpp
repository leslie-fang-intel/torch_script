// #include <pybind11/pybind11.h>

// int test_ll_cpp_add(int a, int b) {
//     return a*2 + b;
// }

// PYBIND11_MODULE(test_ll_extension, m) {
//     m.def("test_ll_cpp_add", &test_ll_cpp_add, "A function that adds two numbers");
// }

#include <Python.h>

static PyObject* say_hello(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", "Hello from C extension!");
}

static PyMethodDef ExampleMethods[] = {
    {"say_hello", say_hello, METH_NOARGS, "Say hello"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    "example",   /* name of module */
    "Example C extension", /* module documentation */
    -1,
    ExampleMethods
};

PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&examplemodule);
}