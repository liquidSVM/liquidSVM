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

#define VPRINTF(message_format, ...) va_list arguments; \
              va_start(arguments, message_format); \
              vprintf(message_format, arguments); \
              va_end(arguments);
//              PySys_FormatStdout(message_format, arguments); \ // gave SIGSEGV
//              PySys_WriteStdout(message_format, arguments); \ // gave SIGSEGV and we would need to restrict to 1000bytes

#include <Python.h>

void CheckUserInterrupt();

#include "common/liquidSVM.h"

// Interrupt handling does not work at the moment, since we cannot save thread state
// and the following needs to be executed "on the GIL" or so.
// This will only possible once we implement real bindings and all the functions
// will then be able to save the thread state
// still this will be needed to be done only say on the master thread??
#ifndef BLABLA____
void CheckUserInterrupt(){}
#else
#define MY_PY_BEGIN if (PyEval_ThreadsInitialized()) py_thread_state = PyEval_SaveThread();
#define MY_PY_END if (py_thread_state) PyEval_RestoreThread(py_thread_state);
PyThreadState *py_thread_state = NULL;
void CheckUserInterrupt(){
	MY_PY_END
	PyErr_CheckSignals();
	if(PyErr_Occurred())
	{
		MY_PY_BEGIN
		throw string("Interrupted");
	}else{
		MY_PY_BEGIN
	}
}
#endif


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

