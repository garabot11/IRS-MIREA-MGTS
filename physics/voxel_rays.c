/* First occupied voxel along each ray, on the NumPy reference's distance grid. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

void kontur_voxel_first_hit(const uint8_t *grid, const int64_t *shape,
                            const int64_t *minimum, double edge,
                            const double *origin, const double *directions,
                            double *ranges, const double *samples,
                            size_t ray_count, size_t sample_count) {
    for (size_t i = 0; i < ray_count; ++i) {
        const double *direction = &directions[3 * i];
        for (size_t j = 0; j < sample_count && samples[j] <= ranges[i]; ++j) {
            const double s = samples[j];
            const double x = floor((origin[0] + s * direction[0]) / edge) - minimum[0];
            const double y = floor((origin[1] + s * direction[1]) / edge) - minimum[1];
            const double z = floor((origin[2] + s * direction[2]) / edge) - minimum[2];
            if (x >= 0 && x < shape[0] && y >= 0 && y < shape[1] && z >= 0 && z < shape[2]) {
                const size_t cell = ((size_t)x * shape[1] + (size_t)y) * shape[2] + (size_t)z;
                if (grid[cell]) {
                    ranges[i] = s;
                    break;
                }
            }
        }
    }
}
