#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: language_level=3

from libc.stdint cimport uintptr_t


cdef extern from "cuvs/neighbors/prefilters.h" nogil:

    ctypedef enum cuvsPrefilterType:
        NO_FILTER
        BITSET
        BITMAP

    ctypedef struct cuvsPrefilter:
        uintptr_t addr
        cuvsPrefilterType type
