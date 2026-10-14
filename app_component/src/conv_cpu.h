#ifndef CONV_CPU_H
#define CONV_CPU_H

#include <stdint.h>

/* 单通道 VALID 互相关配置。stride 单位是字节，权重按行排列。
 * 调用者保证 kernel_size 为 3/5，shift 为 0..31，宽高至少等于核大小。
 * mode16=0 使用 int8_t 输入/输出；mode16=1 使用 int16_t 输入/输出。
 * bias 的小数位数必须与输入和权重的乘积一致。
 */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t kernel_size;
    uint32_t mode16;
    uint32_t round_enable;
    uint32_t shift;
    int32_t  bias;
    int16_t  weights[25];
} conv_config_t;

/* 纯 C 标量实现：64 位累加、与 RTL 相同的舍入和饱和规则。
 * 只修改每行有效输出元素，保留行尾填充。输入和输出不可重叠。
 */
void conv_cpu_run(
    const conv_config_t *config, const void *source, void *target
);

#endif
