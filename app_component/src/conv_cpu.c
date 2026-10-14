#include "conv_cpu.h"

/* 40 位 RTL 累加器可以容纳全部 25 个 signed16 乘积和 signed32 bias。
 * C 使用 int64_t 实现同样的数值范围，避免 32 位累加溢出。
 */
static int16_t quantize(
    int64_t value, uint32_t shift, uint32_t round_enable, uint32_t mode16
) {
    if (shift != 0U) {
        int64_t magnitude = value < 0 ? -value : value;
        if (round_enable != 0U) {
            /* 就近舍入，中点远离零。 */
            magnitude = (magnitude + (INT64_C(1) << (shift - 1U))) >> shift;
        } else if (value < 0) {
            /* 用正数运算实现负数向负无穷取整，不依赖 C 的负数右移规则。 */
            magnitude = (magnitude + ((INT64_C(1) << shift) - 1)) >> shift;
        } else {
            magnitude >>= shift;
        }
        value = value < 0 ? -magnitude : magnitude;
    }

    int64_t minimum = mode16 != 0U ? -32768 : -128;
    int64_t maximum = mode16 != 0U ? 32767 : 127;
    if (value < minimum) {
        value = minimum;
    } else if (value > maximum) {
        value = maximum;
    }
    return (int16_t)value;
}

/* 分开写 8/16 位循环，避免在每一次乘法中判断元素位宽。 */
static void
convolve8(const conv_config_t *c, const int8_t *source, int8_t *target) {
    uint32_t out_width  = c->width - c->kernel_size + 1U;
    uint32_t out_height = c->height - c->kernel_size + 1U;
    for (uint32_t y = 0; y < out_height; ++y) {
        int8_t *output_row = target + y * c->dst_stride;
        for (uint32_t x = 0; x < out_width; ++x) {
            int64_t sum = c->bias;
            for (uint32_t r = 0; r < c->kernel_size; ++r) {
                const int8_t  *input_row = source + (y + r) * c->src_stride + x;
                const int16_t *weights   = c->weights + r * c->kernel_size;
                for (uint32_t col = 0; col < c->kernel_size; ++col) {
                    sum += (int64_t)input_row[col] * weights[col];
                }
            }
            output_row[x] =
                (int8_t)quantize(sum, c->shift, c->round_enable, 0U);
        }
    }
}

static void
convolve16(const conv_config_t *c, const int16_t *source, int16_t *target) {
    uint32_t out_width    = c->width - c->kernel_size + 1U;
    uint32_t out_height   = c->height - c->kernel_size + 1U;
    uint32_t src_elements = c->src_stride / sizeof(int16_t);
    uint32_t dst_elements = c->dst_stride / sizeof(int16_t);
    for (uint32_t y = 0; y < out_height; ++y) {
        int16_t *output_row = target + y * dst_elements;
        for (uint32_t x = 0; x < out_width; ++x) {
            int64_t sum = c->bias;
            for (uint32_t r = 0; r < c->kernel_size; ++r) {
                const int16_t *input_row = source + (y + r) * src_elements + x;
                const int16_t *weights   = c->weights + r * c->kernel_size;
                for (uint32_t col = 0; col < c->kernel_size; ++col) {
                    sum += (int64_t)input_row[col] * weights[col];
                }
            }
            output_row[x] = quantize(sum, c->shift, c->round_enable, 1U);
        }
    }
}

void conv_cpu_run(
    const conv_config_t *config, const void *source, void *target
) {
    if (config->mode16 != 0U) {
        convolve16(config, source, target);
    } else {
        convolve8(config, source, target);
    }
}
