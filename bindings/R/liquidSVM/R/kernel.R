# Copyright 2015-2017 Philipp Thomann
# 
# This file is part of liquidSVM.
# 
#  liquidSVM is free software: you can redistribute it and/or modify
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

#' Calculates the kernel matrix.
#' 
#' @param data the data set
#' @param gamma the gamma-parameter
#' @param type kernel type to use: one of "gaussian.rbf","poisson"
#' @param threads how many threads to be used
## #' @param GPUs how many GPUs to use (not implemented yet)
#' @return kernel matrix
#' @examples 
#' kern(trees)
#' image(kern(trees, 2, "pois"))
#' @export
kern <- function(data, gamma=1, type=c("gaussian.rbf","poisson"), threads=1){
  GPUs <- 0
  if(length(dim(data))==2)
    stopifnot(length(dim(data))==2 & all(sapply(data, is.numeric)))
  else if(is.null(dim(data)) & is.numeric(data))
    data <- data.frame(data)
  else
    stop("Could not use data for kernel calculation")
  aux_file <- ""
  if(missing(type)){
    type <- 0
  }else if(!length(type)%in%c(1,2)){
    warning("type argument not understood")
    type <- 0
  }else if(!is.numeric(type)){
    if(length(type)==2){
      aux_file <- type[2]
      stopifnot(file.exists('aux_file'))
      type <- type[1]
    }
    type <- match.arg(type, choices = c("gaussian.rbf","poisson", "hierarchical"))
    type <- switch(type, gaussian.rbf=0, poisson=1, hierarchical=2)
  }

  ret <- .Call('liquid_svm_R_kernel',
               as.numeric(t(data)), as.integer(ncol(data)), as.integer(type), as.character(aux_file), as.numeric(gamma), as.integer(threads), as.integer(GPUs) ,
               PACKAGE='liquidSVM')
  stopifnot( is.numeric(ret) & !is.null(ret))

  return(matrix(ret, nrow(data)))
}
  
