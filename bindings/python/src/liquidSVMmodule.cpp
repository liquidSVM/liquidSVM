// Copyright 2015-2017 Philipp Thomann
//
// This file is part of liquidSVM.
//
// liquidSVM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// liquidSVM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.

#undef COMPILE_FOR_COMMAND_LINE__

#define COMPILE_FOR_R__

#ifdef __MINGW32__
#define MS_WIN64
#endif

#include <Python.h>

#define VPRINTF(message_format, ...) \
{ \
    char buffer[1001]; \
    va_list args; \
    va_start (args, message_format); \
    vsnprintf (buffer, 1001, message_format, args); \
    va_end (args); \
    PyGILState_STATE gstate; \
    gstate = PyGILState_Ensure(); \
    PySys_WriteStdout("%s", buffer); \
    PyGILState_Release(gstate); \
}

bool doInterrupt = false;
void CheckUserInterrupt();

#define CLEAR_INTERRUPT doInterrupt = false;


#include "common/liquidSVM.h"

void CheckUserInterrupt(){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
	int err = PyErr_CheckSignals();
    PyGILState_Release(gstate);
	if(err < 0)
	    doInterrupt = true;
	if(doInterrupt)
		throw string("Interrupted");
}


/*
// only for reference - maybe we never will use this approach...
static PyObject*
say_hello(PyObject* self, PyObject* args)
{
    const char* name;

    if (!PyArg_ParseTuple(args, "s", &name))
        return NULL;

    printf("Hello %s! %s\n", name, liquid_svm_default_params(-1,1));

    Py_RETURN_NONE;
}
*/

/*
static PyMethodDef liquidSVMmethods[] =
{
     //{"say_hello", say_hello, METH_VARARGS, "Greet somebody."},
     {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initliquidSVM(void)
{
	printf("Loading C++ Module of liquidSVM-python\n");
    (void) Py_InitModule("liquidSVM", liquidSVMmethods);
}
*/

// the following helps in eclipse to switch between python 2 and 3
#ifndef PY_MAJOR_VERSION
#define PY_MAJOR_VERSION 3
#endif


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif




static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}




static PyMethodDef liquidSVM_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};




#if PY_MAJOR_VERSION >= 3

static int liquidSVM_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int liquidSVM_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "liquidSVM",
        NULL,
        sizeof(struct module_state),
        liquidSVM_methods,
        NULL,
        liquidSVM_traverse,
        liquidSVM_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_liquidSVM(void)

#else
#define INITERROR return

void
initliquidSVM(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("liquidSVM", liquidSVM_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("liquidSVM.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

