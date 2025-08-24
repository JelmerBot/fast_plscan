#include <nanobind/nanobind.h>
#include <omp.h>

namespace nb = nanobind;

NB_MODULE(_threads_ext, m) {
  m.doc() = "Module for adjusting how many threads PLSCAN uses.";

  m.def(
      "get_max_threads", &omp_get_max_threads,
      R"(Returns the default number of OpenMP threads used.)"
  );

  m.def(
      "set_num_threads", &omp_set_num_threads, nb::arg("num_threads"),
      R"(
          Sets the default number of OpenMP threads to use.

          Parameters
          ----------
          num_threads : int
                The number of threads to set.
        )"
  );
}