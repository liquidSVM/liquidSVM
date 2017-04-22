#!/bin/bash
 
# Copyright 2015, 2016, 2017 Ingo Steinwart
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

# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.



DISPLAY_PART=$1

echo "The meaning of the parameters is as follows:"
echo


echo "<base_filename>             is the name of the dataset (without extensions),"
echo "                            which is supposed to be in the folder: ./data ."
echo "                            It is assumed that there are two files"
echo "                                <base_filename>.train.<extension>"
echo "                                <base_filename>.test.<extension>"
echo "                            Here <extension> can be 'csv' for comma-separated"
echo "                            data files (label first) or 'lsv' for LIBSVM's"
echo "                            format."

echo
echo "<display>                   This parameter determines the amount of output of"
echo "                            you see at the screen: The larger its value is,"
echo "                            the more you see."

echo
echo "<threads>                   This parameter determines the number of cores"
echo "                            used for computing the kernel matrices, the"
echo "                            validation error, and the test error."
echo "                            The default <threads> = 0 means that all physical"
echo "                            cores of your CPU run one thread."

if [[ -z $DISPLAY_PART ]]
then
	echo
	echo "<partition>                 This parameter determines the way the input space"
	echo "                            is partitioned. The default value <partition> = 0"
	echo "                            disables partitioning. For large data sets the" 
	echo "                            highest speed can usually be achieved by"
	echo "                            <partition> = 4, whereas the best test error is"
	echo "                            typically obtained by <partition> = 5."
fi

echo
echo "<grid_size>                 This parameter determines the size of the hyper-"
echo "                            parameter grid used during the training phase."
echo "                            Larger values correspond to larger grids. By"
echo "                            default, a 10x10 grid is used. Exact descriptions"
echo "                            can be found in the file:"
echo "                                ./scripts/auxiliary/get-grid.sh"

if [[ -z $DISPLAY_PART ]]
then
	echo
	echo "<adaptive_grid_search>      This parameter determines, whether an adaptive"
	echo "                            grid search heuristic is employed. Larger values"
	echo "                            lead to more aggressive strategies. The default"
	echo "                            <adaptive_grid_search> = 0 disables the heuristic."
fi

echo
echo "<random_seed>               This parameter determines the seed for the random"
echo "                            generator. <random_seed> = -1 uses the internal" 
echo "                            timer create the seed. All other values lead to"
echo "                            repeatable behavior of the svm."
