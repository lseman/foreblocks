#pragma once

#include "foretree/core/dataset.hpp"
#include "foretree/core/histogram_primitives.hpp"
#include "foretree/core/binning_strategies.hpp"
#include "foretree/core/data_binner.hpp"
#include "foretree/core/gradient_hist_system.hpp"

#ifdef FORETREE_HAS_CUDA
#include "foretree/gpu/cuda_histogram.hpp"
#endif
