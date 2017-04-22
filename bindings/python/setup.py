from setuptools import setup
from setuptools.extension import Extension
#from setuptools.command.install import install
import shlex

import sys, os

# class liquidSvmInstall(install):
# 	user_options = install.user_options + [
#           ('liquid-svm-target=', None, 'liquid svm target: native to optimize for current machine or generic for generic architectures'),
#           ('liquid-svm-compile-args=', None, 'further arguments to pass to compiler'),
#     ]
# 	def initialize_options(self):
# 		self.liquid_svm_target = None
# 	def finalize_options(self):
# 		assert self.liquid_svm_target in (None, 'native', 'generic'), 'Invalid liquid-svm-target!'
# 	def run(self):
# 		print "Target: ", self.liquid_svm_target
# 		install.run(self)
# 		print "Target: ", self.liquid_svm_target

target = 'native'
extra_compile_args = []
if 'LIQUIDSVM_CONFIGURE_ARGS' in os.environ:
	args = shlex.split(os.environ['LIQUIDSVM_CONFIGURE_ARGS'])
	if args[0] in ['native','generic','debug','empty']:
		target = args[0]
		args = args[1:]
		extra_compile_args = extra_compile_args + args
print("Using target: ",target)
print("Using further args: ",extra_compile_args)

if sys.platform.startswith("win"):
	#extra_compile_args = []
	#extra_link_args = ['/EXPORT:liquid_svm_set_param','/EXPORT:liquid_svm_get_param','/EXPORT:liquid_svm_get_config_line',
	#'/EXPORT:liquid_svm_init','/EXPORT:liquid_svm_train','/EXPORT:liquid_svm_select','/EXPORT:liquid_svm_test','/EXPORT:liquid_svm_clean']
	extra_link_args = []
	export_symbols = ['liquid_svm_set_param','liquid_svm_get_param','liquid_svm_get_config_line',
	'liquid_svm_init','liquid_svm_train','liquid_svm_select','liquid_svm_test','liquid_svm_clean']
else:
	export_symbols = []
	extra_compile_args += ['-O3','-std=c++0x']
	if target=="native":
		extra_compile_args += ['-march=native']
	elif target=="generic":
		extra_compile_args += ['-mtune=generic','-msse2']
	elif target=="empty":
		extra_compile_args += ['-mtune=generic','-msse2']
	elif target=="debug":
		extra_compile_args += ['-g2','-mtune=generic','-msse2']
	extra_link_args = []
	if sys.platform == 'darwin':
		extra_compile_args += ['-stdlib=libc++','-mmacosx-version-min=10.7','-DSYSTEM_WITH_64BIT']
		extra_link_args += ['-mmacosx-version-min=10.7']

module = Extension('liquidSVM', sources = ['src/liquidSVMmodule.cpp'],
                   include_dirs=["../..","../../bindings","src"],
                   extra_compile_args=extra_compile_args, extra_link_args=extra_link_args,
				   export_symbols=export_symbols)

setup (name = 'liquidSVM',
		#cmdclass={ 'install': liquidSvmInstall},
        version = '1.0.1', ## needs also to be set in Makefile
        description = '''Support vector machines (SVMs) and related kernel-based learning
		algorithms are a well-known class of machine learning algorithms, for
		non-parametric classification and regression.
		liquidSVM is an implementation of SVMs whose key features are:
		fully integrated hyper-parameter selection,
		extreme speed on both small and large data sets,
		full flexibility for experts, and
		inclusion of a variety of different learning scenarios:
		multi-class classification, ROC, and Neyman-Pearson learning, and
		least-squares, quantile, and expectile regression.''',
        url='http://www.isa.uni-stuttgart.de/software/',
        author='Ingo Steinwart, Philipp Thomann',
        author_email='philipp.thomann@mathematik.uni-stuttgart.de',
        license='AGPL v3',
	    long_description=open(os.path.join(os.path.dirname(__file__), "README.rst")).read(),
        packages=['liquidSVM'],
        package_data={'liquidSVM': ['data/*.csv*','demo.html','demo.ipynb']},
        ext_modules = [module],
        install_requires = ['numpy'] )
