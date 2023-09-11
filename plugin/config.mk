#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2022, Advanced Micro Devices, Inc.
#  Copyright (C) 2023, Southern Methodist University
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

ifndef CONFIG_MK_PLUGIN_INCLUDED
CONFIG_MK_PLUGIN_INCLUDED := yes

# The installation prefix, exec_prefix, libdir, includedir, and shareddir
# values from configure tell us where to install the libraries, header files,
# and public makefile fragments. We must first assign each substituted
# @anchor@ to its own variable. Why? Because the subsitutions may contain
# unevaluated variable expressions. For example, '${exec_prefix}/lib' may be replaced
# with '${exec_prefix}/lib'. By assigning the anchors to variables first, and
# then assigning them to their final INSTALL_* variables, we allow prefix and
# exec_prefix to be used in the definitions of exec_prefix, libdir,
# includedir, and sharedir.
prefix            := /Users/dnp/blis-plugins
exec_prefix       := ${prefix}
libdir            := ${exec_prefix}/lib
includedir        := ${prefix}/include
sharedir          := /Users/dnp/blis-plugins/share/blis/..

# Define the name of the config makefile.
REAL_CONFIG_MK_FILE := $(sharedir)/blis/config.mk

# Include the configuration file.
include $(REAL_CONFIG_MK_FILE)

# The name of the plugin.
PLUGIN_NAME       := fmm_blis

# This list contains some number of "kernel:config" pairs, where "config"
# specifies which configuration's compilation flags (CFLAGS) should be
# used to compile the source code for the kernel set named "kernel".
KCONFIG_MAP       := haswell:haswell zen:haswell

# The C compiler.
CC_VENDOR         := gcc
CC                := gcc-13

# Important C compiler ranges.
GCC_OT_4_9_0      := no
GCC_OT_6_1_0      := no
GCC_OT_9_1_0      := no
GCC_OT_10_3_0     := no
CLANG_OT_9_0_0    := no
CLANG_OT_12_0_0   := no
AOCC_OT_2_0_0     := no
AOCC_OT_3_0_0     := no

# The C++ compiler.
CXX               := g++

# The Fortran compiler.
FC                := gfortran

# Static library indexer.
RANLIB            := ranlib

# Archiver.
AR                := ar

# Preset (required) CFLAGS and LDFLAGS. These variables capture the value
# of the CFLAGS and LDFLAGS environment variables at configure-time (and/or
# the value of CFLAGS/LDFLAGS if either was specified on the command line).
# These flags are used in addition to the flags automatically determined
# by the build system.
CFLAGS_PRESET     := 
LDFLAGS_PRESET    := 

# The level of debugging info to generate.
DEBUG_TYPE        := off
ENABLE_DEBUG      := no

# Whether to compile and link the AddressSanitizer library.
MK_ENABLE_ASAN    := no

# Whether the compiler supports "#pragma omp simd" via the -fopenmp-simd option.
PRAGMA_OMP_SIMD   := no

# Whether to output verbose command-line feedback as the Makefile is
# processed.
ENABLE_VERBOSE    := no

# Whether we need to employ an alternate method for passing object files to
# ar and/or the linker to work around a small value of ARG_MAX.
ARG_MAX_HACK      := no

# Whether to build the static and shared libraries.
# NOTE: The "MK_" prefix, which helps differentiate these variables from
# their corresonding cpp macros that use the BLIS_ prefix.
MK_ENABLE_STATIC  := yes
MK_ENABLE_SHARED  := yes

# Whether to use an install_name based on @rpath.
MK_ENABLE_RPATH   := no

# Whether to export all symbols within the shared library, even those symbols
# that are considered to be for internal use only.
EXPORT_SHARED     := public

# end of ifndef CONFIG_MK_PLUGIN_INCLUDED conditional block
endif
