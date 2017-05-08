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

require(liquidSVM)

context("liquidSVM-kernel")

test_that("kernel",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  result <- kern(trees, threads=1)
  n <- nrow(trees)
  expect_equal(dim(result),c(n,n))
  expect_true(all(result >= 0))
})

test_that("kernel named type",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  result <- kern(trees, type='gaussian', threads=1)
  n <- nrow(trees)
  expect_equal(dim(result),c(n,n))
  expect_true(all(result >= 0))
})

test_that("kernel aux_file",{
  set.seed(123)
  
  # aux_file is actually not read for gaussian.rbf...
  expect_error(result <- kern(trees, type=c('gaussian','.'), threads=1))
})

test_that("kernel wrong type",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  expect_warning(result <- kern(trees, type=NULL, threads=1))
  n <- nrow(trees)
  expect_equal(dim(result),c(n,n))
  expect_true(all(result >= 0))
})

test_that("kernel 1-dim",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  result <- kern(trees$Girth, threads=1)
  n <- nrow(trees)
  expect_equal(dim(result),c(n,n))
  expect_true(all(result >= 0))
})

