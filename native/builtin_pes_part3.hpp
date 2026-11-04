    std::vector<std::pair<int64_t, int64_t>> pairs;
};

struct LJContext {
    double epsilon, sigma, cutoff, skin;
    double cutoff2, list_cutoff2, energy_shift;
    std::vector<LJFrameCache> frames;

    LJContext(double epsilon_, double sigma_, double cutoff_, double skin_)
        : epsilon(epsilon_), sigma(sigma_), cutoff(cutoff_), skin(skin_),
          cutoff2(cutoff_ * cutoff_),
          list_cutoff2((cutoff_ + skin_) * (cutoff_ + skin_)) {
        const double ratio = sigma / cutoff;
        const double r6 = ratio * ratio * ratio * ratio * ratio * ratio;
        energy_shift = 4. * epsilon * (r6 * r6 - r6);
    }
};

inline void message(pyrpmd_error_v2* error, const char* text) noexcept {
    if (!error) return;
    error->required_capacity = static_cast<int64_t>(std::strlen(text)) + 1;
    if (error->message && error->capacity > 0)
        std::snprintf(error->message, static_cast<size_t>(error->capacity), "%s", text);
}

inline pyrpmd_rc_v2 reject(pyrpmd_error_v2* error, const char* text,
                          pyrpmd_rc_v2 code = PYRPMD_INVALID_ARGUMENT_V2) noexcept {
    message(error, text);
    return code;
}

inline bool invert_cell(const double* cell, double* inverse) noexcept {
    double matrix[3][6]{};
    double norm = 0.;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const double value = cell[3 * row + column];
            if (!std::isfinite(value)) return false;
            matrix[row][column] = value;
            norm += value * value;
        }
        matrix[row][3 + row] = 1.;
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.) || !std::isfinite(norm)) return false;
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row)
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
        if (!(std::abs(matrix[pivot][column]) > 1e-12 * norm)) return false;
        if (pivot != column)
            for (int entry = 0; entry < 6; ++entry)
                std::swap(matrix[pivot][entry], matrix[column][entry]);
        const double divisor = matrix[column][column];
        for (int entry = 0; entry < 6; ++entry) matrix[column][entry] /= divisor;
        for (int row = 0; row < 3; ++row) if (row != column) {
            const double factor = matrix[row][column];
            for (int entry = 0; entry < 6; ++entry)
                matrix[row][entry] -= factor * matrix[column][entry];
        }
    }
    double inverse_norm = 0.;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            inverse[3 * row + column] = matrix[row][3 + column];
            inverse_norm += inverse[3 * row + column] * inverse[3 * row + column];
        }
    inverse_norm = std::sqrt(inverse_norm);
    return std::isfinite(inverse_norm) && norm * inverse_norm <= 1e12;
}

inline bool minimum_image(const double* displacement, const double* cell,
                          const double* inverse, double* result) noexcept {
    double fractional[3], reduced_fractional[3], nearest[3];
    for (int axis = 0; axis < 3; ++axis) {
        fractional[axis] = 0.;
        for (int cart = 0; cart < 3; ++cart)
            fractional[axis] += displacement[cart] * inverse[3 * cart + axis];
        if (!std::isfinite(fractional[axis]) ||
            std::abs(fractional[axis]) > static_cast<double>(INT64_MAX) / 4.) return false;
        nearest[axis] = std::nearbyint(fractional[axis]);
        reduced_fractional[axis] = fractional[axis] - nearest[axis];
    }
    for (int cart = 0; cart < 3; ++cart) {
        result[cart] = displacement[cart];
        for (int axis = 0; axis < 3; ++axis)
            result[cart] -= nearest[axis] * cell[3 * axis + cart];
    }
    double best2 = result[0] * result[0] + result[1] * result[1] + result[2] * result[2];
    double inverse_norm2 = 0.;
    for (int index = 0; index < 9; ++index) inverse_norm2 += inverse[index] * inverse[index];
    const double radius = std::sqrt(std::max(0., best2 * inverse_norm2)) + 32. * std::numeric_limits<double>::epsilon();
    int64_t low[3], high[3];
    uint64_t visits = 1;
    for (int axis = 0; axis < 3; ++axis) {
        low[axis] = static_cast<int64_t>(std::ceil(reduced_fractional[axis] - radius));
        high[axis] = static_cast<int64_t>(std::floor(reduced_fractional[axis] + radius));
        if (high[axis] < low[axis]) return false;
        const uint64_t width = static_cast<uint64_t>(high[axis] - low[axis] + 1);
        if (width > 100000 || visits > 100000 / width) return false;
        visits *= width;
    }
    double trial[3];
    for (int64_t a = low[0]; a <= high[0]; ++a)
        for (int64_t b = low[1]; b <= high[1]; ++b)
            for (int64_t c = low[2]; c <= high[2]; ++c) {
                const double lattice[3] = {static_cast<double>(a), static_cast<double>(b), static_cast<double>(c)};
                for (int cart = 0; cart < 3; ++cart) {
                    trial[cart] = result[cart];
                    for (int axis = 0; axis < 3; ++axis)
                        trial[cart] -= lattice[axis] * cell[3 * axis + cart];
                }
                const double norm2 = trial[0] * trial[0] + trial[1] * trial[1] + trial[2] * trial[2];
                if (norm2 < best2) {
                    best2 = norm2;
                    std::copy(trial, trial + 3, result);
                }
            }
    return true;
}

inline bool valid_cutoff_cell(const double* inverse, double cutoff) noexcept {
    double minimum_height = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        double norm2 = 0.;
        for (int cart = 0; cart < 3; ++cart)
            norm2 += inverse[3 * cart + axis] * inverse[3 * cart + axis];
        if (!(norm2 > 0.) || !std::isfinite(norm2)) return false;
        minimum_height = std::min(minimum_height, 1. / std::sqrt(norm2));
    }
    return cutoff <= .5 * minimum_height *
        (1. + 64. * std::numeric_limits<double>::epsilon());
}

inline bool same_cell(const LJFrameCache& cache, const double* cell) noexcept {
    return cache.valid && std::equal(cache.cell.begin(), cache.cell.end(), cell);
}

inline bool rebuild_needed(const LJFrameCache& cache, const double* q, int64_t natoms,
                           const double* cell, const double* inverse, double skin) noexcept {
    if (!same_cell(cache, cell) || cache.natoms != natoms ||
        cache.reference.size() != static_cast<size_t>(3 * natoms)) return true;
    const double limit2 = .25 * skin * skin;
    for (int64_t atom = 0; atom < natoms; ++atom) {
        double delta[3], image[3];
        for (int cart = 0; cart < 3; ++cart)
            delta[cart] = q[3 * atom + cart] - cache.reference[3 * atom + cart];
        if (!minimum_image(delta, cell, inverse, image)) return true;
        const double norm2 = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
        if (norm2 > limit2) return true;
    }
    return false;
}

inline bool rebuild(LJFrameCache& cache, const double* q, int64_t natoms,
                    const double* cell, const double* inverse, double list_cutoff2) {
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (int64_t i = 0; i < natoms - 1; ++i)
        for (int64_t j = i + 1; j < natoms; ++j) {
            double delta[3], image[3];
            for (int cart = 0; cart < 3; ++cart)
                delta[cart] = q[3 * j + cart] - q[3 * i + cart];
            if (!minimum_image(delta, cell, inverse, image)) return false;
            const double norm2 = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
            if (norm2 < list_cutoff2) pairs.emplace_back(i, j);
        }
    std::vector<double> reference(q, q + 3 * natoms);
    std::copy(cell, cell + 9, cache.cell.begin());
    cache.natoms = natoms;
    cache.reference.swap(reference);
    cache.pairs.swap(pairs);
    cache.valid = true;
    return true;
}

extern "C" inline pyrpmd_rc_v2 pyrpmd_builtin_lj_evaluate_v2(
    pyrpmd_pes_context_v2* opaque, const pyrpmd_pes_input_v2* input,
    pyrpmd_pes_output_v2* output, pyrpmd_error_v2* error) noexcept {
    try {
        if (error && (error->capacity < 0 || (error->capacity > 0 && !error->message)))
