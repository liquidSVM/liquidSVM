# Copyright 2015-2017 Philipp Thomann
#
# This file is part of liquidSVM.
#
# liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function
import sys
import liquidSVM
from liquidSVM import mcSVM, lsSVM, qtSVM, exSVM, nplSVM, rocSVM
from liquidSVM import iris, iris_labs


def demo():
    # print  "Compiled " +_default_params(-1,1)

    print(iris.shape)

    model = mcSVM(iris, iris_labs, display=1)
#     errT = model.train()
#     errT = model.train(solver=2, L=1, f=[4,5], d=1)
#     model.select()
    result, err = model.test(iris, iris_labs)
    print(result[:4, :], err[:4, :])
    # print model.test(iris, np.zeros(iris.shape[0]))
#     print model.predict(iris)
    print("================== Least squares Regression =====================")
    model = lsSVM(iris, iris_labs, display=1)
    print(
        "======================= Quantile Regression =======================")
    model = qtSVM(iris, iris_labs, display=1)
    print("==================== Expectile Regression =======================")
    model = exSVM(iris, iris_labs, display=1)
    print("======================= NPL =======================")
    model = nplSVM(iris, iris_labs, display=1)
    print("======================= ROC =======================")
    model = rocSVM(iris, iris_labs, display=1)

    print(model.get("COMPILE_INFO"))
    model.set("DISPLAY", "2")
    print(model.get("DISPLAY"))
    print(model.configLine(3))
    print("cleaning: ")
    model.clean()


def help():
    print("""Usage: python -m liquidSVM <data-name> [<scenario>] [<config-args>*]
    E.g. python -m liquidSVM covtype.1000 mc --threads=1 --voronoi='5 400'
    """)


def main(*argv):
    argv = list(argv[1:])
    if len(argv) == 0 or argv[0] == '-h' or argv[0] == '--help':
        help()
        return

    if argv[0] == '-v' or argv[0] == '--version':
        print('TODO')
        return

    data = argv.pop(0)
    scenario = 'ls' if len(argv) < 2 or argv[:2] == '--' else argv.pop(0)
    config = {}

    for arg in argv:
        if arg[:2] == '--':
            s = arg[2:].split("=")
            if len(s) != 2:
                print('Did not understand: ' + arg)
                continue
            config[s[0]] = s[1]
        else:
            print('Did not understand: ' + arg)

    f = getattr(liquidSVM, scenario + 'SVM')
    # f = globals()[scenario+'SVM']
    model = f(data, **config)
    print("Test errors:", *model.last_result[1][:, 0])


if __name__ == '__main__':
    print("Hello to liquidSVM (python)")
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] == '--demo':
        demo()
    elif sys.argv[1] == '--doctest':
        import doctest
        print("Performing doctest")
        doctest.testmod(liquidSVM)
    else:
        main(*sys.argv)
