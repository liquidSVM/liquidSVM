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


 


#####################################################
#
# This script prepares all global options for svm-XXX
#
#####################################################


# Make sure only options are used for which the corresponding variable is set


if ! [[ -z $THREADS ]]
then
	THREADS_OPT="-T "$THREADS" "$THREAD_OFFSET
fi

if ! [[ -z $GPUS ]]
then
	GPUS_OPT="-GPU "$GPUS" "$GPU_OFFSET
fi

if ! [[ -z $DISPLAY ]]
then
	DISPLAY_OPT="-d "$DISPLAY
fi


SVM_GLOBAL_OPTS="$THREADS_OPT $GPUS_OPT $DISPLAY_OPT"
