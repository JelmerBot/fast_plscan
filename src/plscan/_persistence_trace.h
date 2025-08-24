#ifndef PLSCAN_PERSISTENCE_TRACE_H
#define PLSCAN_PERSISTENCE_TRACE_H

#include <span>

#include "_array.h"

// Ownership for a PersistenceTraceView.
struct PersistenceTraceCapsule {
  nb::capsule min_size;
  nb::capsule persistence;
};

// Non-owning view of a leaf tree
struct PersistenceTraceWriteView {
  std::span<float> const min_size;
  std::span<float> const persistence;

  [[nodiscard]] size_t size() const {
    return min_size.size();
  }
};

// Non-owning view of a leaf tree
struct PersistenceTraceView {
  std::span<float const> const min_size;
  std::span<float const> const persistence;

  [[nodiscard]] size_t size() const {
    return min_size.size();
  }
};

struct PersistenceTrace {
  array_ref<float const> min_size;
  array_ref<float const> persistence;

  PersistenceTrace() = default;
  PersistenceTrace(PersistenceTrace &&) = default;
  PersistenceTrace(PersistenceTrace const &) = default;
  PersistenceTrace &operator=(PersistenceTrace &&) = default;
  PersistenceTrace &operator=(PersistenceTrace const &) = default;

  // Python side constructor.
  PersistenceTrace(
      array_ref<float const> const min_size,
      array_ref<float const> const persistence
  )
      : min_size(min_size), persistence(persistence) {}

  // C++ side constructor that converts buffers to potentially smaller arrays.
  PersistenceTrace(
      PersistenceTraceWriteView const view, PersistenceTraceCapsule cap,
      size_t const num_traces
  )
      : min_size(to_array(view.min_size, std::move(cap.min_size), num_traces)),
        persistence(
            to_array(view.persistence, std::move(cap.persistence), num_traces)
        ) {}

  // Allocate buffers to fill and resize later.
  static auto allocate(size_t const num_traces) {
    auto [sizes, size_cap] = new_buffer<float>(num_traces);
    auto [pers, pers_cap] = new_buffer<float>(num_traces);
    return std::make_pair(
        PersistenceTraceWriteView{sizes, pers},
        PersistenceTraceCapsule{std::move(size_cap), std::move(pers_cap)}
    );
  }

  [[nodiscard]] PersistenceTraceView view() const {
    return {to_view(min_size), to_view(persistence)};
  }

  [[nodiscard]] size_t size() const {
    return min_size.size();
  }
};

#endif  // PLSCAN_PERSISTENCE_TRACE_H
