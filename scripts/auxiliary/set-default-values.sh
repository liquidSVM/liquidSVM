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


 


##################################################################################
#
# This script sets default values for the standard arguments from the command line
# as well as some other variables.
#
##################################################################################


# Fold names

BLOCKS=1
ALTERNATING=2
RANDOM_FOLDS=3
STRATIFIED=4
RANDOM_SUBSET=5


# Solver names 

KERNEL_RULE=0
SVM_LS_2D=1
SVM_HINGE_2D=2
SVM_QUANTILE=3
SVM_EXPECTILE_2D=4
SVM_TEMPLATE=5

SVM_LS_PAR=6
SVM_HINGE_PAR=7


# Command line parameters and more

DISPLAY=1
: ${THREADS:=0}
: ${GPUS:=0}
: ${GRID_CHOICE:=0}
: ${RANDOM_SEED:=1}

: ${RETRAIN_METHOD:=1}
: ${VOTE_METHOD:=1}
: ${PARTITION_CHOICE:=0}
: ${VORONOI:="0"}

: ${ADAPTIVITY_CONTROL:=0}

: ${DISPLAY_ROC_STYLE:=0}

: ${CREATE_SOL_TRAIN:=}
: ${DELETE_OLD_LOGS_ETC:=}


: ${NUM_FOLDS:=5}
: ${THREAD_OFFSET:=0}
: ${GPU_OFFSET:=0}

: ${LABEL_POS:=1}
: ${WEIGHT_POS:=0}
: ${ID_POS:=0}
: ${GROUP_ID_POS:=0}


