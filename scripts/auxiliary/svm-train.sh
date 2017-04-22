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


 


##############################################################################
#
# This script runs all svm-train with the help of some variables set elsewhere
#
##############################################################################


# Make sure only options are used for which the corresponding variable is set

# Global options

source $SML_SCRIPTS_DIR/global-svm-options.sh


#----------------------------- Options for svm-train ------------------------------------------------------

if ! [[ -z $RANDOM_SEED ]]
then
	RANDOM_SEED_OPT="-r "$RANDOM_SEED
fi

if ! [[ -z $CLIPPING ]]
then
	CLIPPING_OPT="-s "$CLIPPING" 0.001"
fi

if ! [[ -z "$WS_TYPE" ]]
then
	WS_TYPE_OPT="-W "$WS_TYPE
fi

if ! [[ -z "$SVM_TYPE" ]]
then
	SVM_TYPE_OPT="-S "$SVM_TYPE
fi

if ! [[ -z "$INIT" ]]
then
	INIT_OPT="-i "$INIT
fi

if ! [[ -z "$BOOT_STRAP" ]]
then
	BOOT_STRAP_OPT="-W 3 "$BOOT_STRAP
fi

if ! [[ -z "$RANDOM_CHUNK_SIZE" ]]
then
	RANDOM_CHUNK_OPT="-P 1 "$RANDOM_CHUNK_SIZE
fi

if ! [[ -z "$RANDOM_CHUNK_NUMBER" ]]
then
	RANDOM_CHUNK_OPT="-P 2 "$RANDOM_CHUNK_NUMBER
fi

if ! [[ -z "$VORONOI" ]]
then
	VORONOI_OPT="-P "$VORONOI
fi

if ! [[ -z "$FOLDS" ]]
then
	FOLDS_OPT="-f "$FOLDS
fi

if ! [[ -z "$WEIGHTS" ]]
then
	WEIGHT_OPT=$(echo -w "[" $WEIGHTS "]")
elif ! [[ -z "$MIN_WEIGHT" ]]
then
	WEIGHT_OPT=$(echo -w $MIN_WEIGHT $MAX_WEIGHT $WEIGHT_STEPS)
	if ! [[ -z "$GEO_WEIGHTS" ]]
	then
		WEIGHT_OPT=$WEIGHT_OPT" "$GEO_WEIGHTS" "$NPL_SWAP
	fi
fi


LOSS_OPT="-L "$LOSS_TYPE" -1.0"


SVM_TRAIN_OPTS=$RANDOM_SEED_OPT"   "$CLIPPING_OPT"   "$WS_TYPE_OPT"   "$INIT_OPT"   "$SVM_TYPE_OPT"   "$BOOT_STRAP_OPT"   "$RANDOM_CHUNK_OPT"   "$VORONOI_OPT"   "$FOLDS_OPT"   "$GRID_OPTIONS"   "$WEIGHT_OPT"  "$WEIGHT_OPTIONS"  "$LOSS_OPT"   "$KERNEL_OPT


echo
echo -------------------------- svm-train -------------------------------------------


$SML_BIN_DIR/svm-train    $SVM_TRAIN_OPTS   $SVM_GLOBAL_OPTS   $SVM_TRAIN_FILENAMES
