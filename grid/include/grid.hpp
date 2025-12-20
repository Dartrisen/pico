#pragma once
#include <cstddef>
#include <cstdint>

struct Grid {
private:
    int nx;
    double dx;

public:
    int idx(int i) const noexcept {
        return i;
    }

    int size() const noexcept {
        return nx;
    }

    double cell_size() const noexcept {
        return dx;
    }
};
