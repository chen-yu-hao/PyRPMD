/* Philox4x32-10 and Box--Muller normals, independently implemented from
 * Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11,
 * doi:10.1145/2063384.2063405. No third-party implementation is copied.
 *
 * philox-v1: the seed is the 64-bit key. A first block with counter
 * (step.lo, step.hi, event.lo, event.hi) derives a full 128-bit event origin.
 * Normal draws XOR (dof/2, bead, purpose, component) into that origin and use
 * the original seed key. No event domain is truncated to a 64-bit subkey. Event counts distinguish
 * repeated baths at the same physical step; all indexing is independent of
 * thread scheduling. Both seed and event count belong in checkpoint state.
 * FP64 libm determines normal rounding; bitwise replay is within one build.
 */
#ifndef PYRPMD_PHILOX_H
#define PYRPMD_PHILOX_H
#include <array>
#include <cmath>
#include <cstdint>

namespace pyrpmd::philox {
using Counter = std::array<uint32_t,4>;
using Key = std::array<uint32_t,2>;

inline Counter generate(Counter counter, Key key) {
    for (int round=0; round<10; ++round) {
        const uint64_t p0=uint64_t(0xD2511F53U)*counter[0];
        const uint64_t p1=uint64_t(0xCD9E8D57U)*counter[2];
        counter={uint32_t(p1>>32)^counter[1]^key[0], uint32_t(p1),
                 uint32_t(p0>>32)^counter[3]^key[1], uint32_t(p0)};
        key[0]+=0x9E3779B9U;key[1]+=0xBB67AE85U;
    }
    return counter;
}

struct Stream {
    Key key{0,0};
    Counter origin{0,0,0,0};
    Stream() = default;
    Stream(uint64_t seed, uint64_t step, uint64_t event)
        : key{uint32_t(seed),uint32_t(seed>>32)},
          origin(generate({uint32_t(step),uint32_t(step>>32),
                           uint32_t(event),uint32_t(event>>32)},key)) {}

    std::array<double,2> normal_pair(uint32_t bead,uint32_t pair,uint32_t purpose,uint32_t component=0) const {
        const auto words=generate({origin[0]^pair,origin[1]^bead,
                                   origin[2]^purpose,origin[3]^component},key);
        // Two 52-bit uniforms strictly inside (0,1), exactly representable.
        const double u=(double((uint64_t(words[0])<<20)|(words[1]>>12))+.5)*0x1p-52;
        const double v=(double((uint64_t(words[2])<<20)|(words[3]>>12))+.5)*0x1p-52;
        const double radius=std::sqrt(-2.*std::log(u)),angle=0x1.921fb54442d18p+2*v;
        return {radius*std::cos(angle),radius*std::sin(angle)};
    }
    double normal(uint32_t bead,uint32_t dof,uint32_t purpose,uint32_t component=0) const {
        return normal_pair(bead,dof/2,purpose,component)[dof%2];
    }
};

inline double normal(uint64_t seed, uint64_t step, uint32_t bead, uint32_t dof,
                     uint32_t purpose, uint64_t event=0, uint32_t component=0) {
    return Stream(seed,step,event).normal(bead,dof,purpose,component);
}
} // namespace pyrpmd::philox
#endif
