# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-learned "base" physics routines for JAX-CFD."""

import jax_cfd.sb.advection
import jax_cfd.sb.array_utils
import jax_cfd.sb.boundaries
import jax_cfd.sb.diffusion
import jax_cfd.sb.equations
import jax_cfd.sb.fast_diagonalization
import jax_cfd.sb.finite_differences
import jax_cfd.sb.forcings
import jax_cfd.sb.funcutils
import jax_cfd.sb.grids
import jax_cfd.sb.initial_conditions
import jax_cfd.sb.interpolation
import jax_cfd.sb.pressure
import jax_cfd.sb.resize
import jax_cfd.sb.subgrid_models
import jax_cfd.sb.time_stepping
import jax_cfd.sb.validation_problems
