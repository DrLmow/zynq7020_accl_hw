/*
 * Zynq-7020 CPU / PL 卷积性能比较。
 *
 * CPU 与硬件使用相同输入、权重、bias、右移和舍入配置。
 * UART 打印、数据生成和结果检查均在计时区间之外。
 * HP0 不提供 CPU 缓存一致性：启动前 flush，完成后 invalidate。
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "conv_cpu.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xparameters.h"
#include "xpseudo_asm.h"
#include "xiltimer.h"

/* 可修改的测试参数。PL 时钟只用于显示周期对应的时间，不设置硬件时钟。 */
#define BENCH_MAX_WIDTH       512U
#define BENCH_MAX_HEIGHT      512U
#define BENCH_REPEATS         5U
#define BENCH_TIMEOUT_SECONDS 5U
#define BENCH_ROUND_ENABLE    1U
#define BENCH_PL_CLOCK_HZ     100000000U
#define ALIGN_UP(value, alignment) \
    (((value) + (alignment) - 1U) & ~((alignment) - 1U))
#define BUFFER_BYTES \
    ALIGN_UP(ALIGN_UP(BENCH_MAX_WIDTH * 2U, 8U) * BENCH_MAX_HEIGHT, 64U)

#define ACCEL_BASE ((UINTPTR)XPAR_CONV_ACCEL_BD_WRAPPER_0_BASEADDR)
enum {
    REG_ID          = 0x000,
    REG_VERSION     = 0x004,
    REG_CONTROL     = 0x008,
    REG_STATUS      = 0x00c,
    REG_IRQ_ENABLE  = 0x010,
    REG_ERROR_CODE  = 0x014,
    REG_CONFIG      = 0x018,
    REG_WIDTH       = 0x01c,
    REG_HEIGHT      = 0x020,
    REG_SRC_ADDR    = 0x024,
    REG_DST_ADDR    = 0x028,
    REG_SRC_STRIDE  = 0x02c,
    REG_DST_STRIDE  = 0x030,
    REG_OUT_SHIFT   = 0x034,
    REG_BIAS        = 0x038,
    REG_OUT_WIDTH   = 0x03c,
    REG_OUT_HEIGHT  = 0x040,
    REG_CYCLES_LO   = 0x044,
    REG_CYCLES_HI   = 0x048,
    REG_MAX_WIDTH   = 0x04c,
    REG_MAX_HEIGHT  = 0x050,
    REG_WEIGHT_BASE = 0x100
};
enum {
    STATUS_BUSY   = 1U,
    STATUS_DONE   = 2U,
    STATUS_ERROR  = 4U,
    STATUS_REJECT = 8U
};

/* 静态缓冲区由链接脚本放进 DDR，不占用小容量栈。
 * 64 字节对齐并保留足够的空间，使范围缓存操作不会涉及相邻缓冲区。
 * union 给 8/16 位访问提供各自正确的对象类型，避免强制转换数组的别名问题。
 */
typedef union
{
    int8_t  pixels8[BUFFER_BYTES];
    int16_t pixels16[BUFFER_BYTES / 2U];
} image_buffer_t;

static image_buffer_t    input_buffer __attribute__((aligned(64)));
static image_buffer_t    cpu_buffer __attribute__((aligned(64)));
static image_buffer_t    hw_buffer __attribute__((aligned(64)));
static volatile uint32_t checksum_sink;

typedef struct {
    uint64_t sum;
    uint64_t minimum;
    uint64_t maximum;
} timing_stats_t;

typedef struct {
    uint64_t execution_ticks; /* 写 START 前到软件观察到 DONE。含轮询开销。 */
    uint64_t total_ticks;     /* 配置、缓存维护、执行、输出失效的总时间。 */
    uint64_t pl_cycles;       /* RTL 的周期计数，包含校验和任务收尾。 */
} hardware_sample_t;

static uint32_t read_register(uint32_t offset) {
    return Xil_In32(ACCEL_BASE + offset);
}

static void write_register(uint32_t offset, uint32_t value) {
    Xil_Out32(ACCEL_BASE + offset, value);
}

static void *image_data(image_buffer_t *buffer, uint32_t mode16) {
    if (mode16 != 0U) {
        return buffer->pixels16;
    }
    return buffer->pixels8;
}

/* 防止编译器跨越计时点移动内存访问，并等待先前 CPU 访问完成。 */
static uint64_t timestamp(void) {
    XTime now;
    __asm__ volatile("" : : : "memory");
    dsb();
    isb();
    XTime_GetTime(&now);
    __asm__ volatile("" : : : "memory");
    return (uint64_t)now;
}

/* Cortex-A9 全局定时器由两个核共享；本例程运行于单核 standalone。
 * 当前 BSP 的 XTime_GetTime() 只读计数器，不保证启动。
 * 保留已有计数值和其他控制位，只清除预分频并设置 enable。
 * prescaler=0 时计数频率为 CPU/2，与 COUNTS_PER_SECOND 一致。
 */
static int initialize_timer(void) {
    const UINTPTR  control_address = (UINTPTR)XPAR_GLOBAL_TIMER_BASEADDR + 0x08U;
    const uint32_t enable_mask     = 1U;
    const uint32_t prescaler_mask  = 0x0000ff00U;
    uint32_t       before          = Xil_In32(control_address);
    uint32_t       configured      = (before & ~prescaler_mask) | enable_mask;
    Xil_Out32(control_address, configured);
    dsb();
    isb();

    /* 使用固定次数的 volatile 循环检查计数是否推进。
     * 不用 sleep() 或基于计时器的等待，避免计时器故障时自检也无限等待。
     * 这只检查计数器工作，不是时钟频率校准。
     */
    uint64_t start = timestamp();
    for (volatile uint32_t spin = 0U; spin < 4096U; ++spin) {
        /* volatile 使循环在 -O3 下也保留实际读写。 */
    }
    uint64_t end   = timestamp();
    uint32_t after = Xil_In32(control_address);
    if ((after & (prescaler_mask | enable_mask)) != enable_mask ||
        end <= start) {
        printf(
            "FAIL: global timer self-check: control=0x%08" PRIx32
            " -> 0x%08" PRIx32 ", counter=%" PRIu64 " -> %" PRIu64 "\r\n",
            before, after, start, end
        );
        return -1;
    }
    printf(
        "Global timer ready: control=0x%08" PRIx32 " -> 0x%08" PRIx32
        ", self-check delta=%" PRIu64 " ticks\r\n",
        before, after, end - start
    );
    return 0;
}

/* 这些卷积任务远长于一个 timer tick。
 * 若计数不前进或倒退，立即报告故障，不输出误导性的零时间/加速比。
 */
static int measure_elapsed(
    const char *name, uint64_t start, uint64_t end, uint64_t *elapsed
) {
    if (end <= start) {
        printf(
            "FAIL: %s timer did not advance: start=%" PRIu64 " end=%" PRIu64
            "\r\n",
            name, start, end
        );
        return -1;
    }
    *elapsed = end - start;
    return 0;
}

static uint64_t ticks_to_ns(uint64_t ticks, uint64_t frequency) {
    /* 先取商和余数，避免 ticks * 1e9 发生不必要的中间溢出。 */
    return (ticks / frequency) * UINT64_C(1000000000) +
           ((ticks % frequency) * UINT64_C(1000000000)) / frequency;
}

static void add_sample(timing_stats_t *stats, uint64_t value) {
    stats->sum += value;
    if (value < stats->minimum) {
        stats->minimum = value;
    }
    if (value > stats->maximum) {
        stats->maximum = value;
    }
}

static void print_time(uint64_t ticks, uint64_t frequency) {
    uint64_t ns = ticks_to_ns(ticks, frequency);
    printf("%" PRIu64 ".%03" PRIu64 " us", ns / 1000U, ns % 1000U);
}

static void
print_stats(const char *name, const timing_stats_t *stats, uint64_t frequency) {
    printf("  %-22s avg=", name);
    print_time(stats->sum / BENCH_REPEATS, frequency);
    printf("  min=");
    print_time(stats->minimum, frequency);
    printf("  max=");
    print_time(stats->maximum, frequency);
    printf("\r\n");
}

static uint32_t random_next(uint32_t *state) {
    /* 固定种子，便于复现；不依赖库函数 rand() 的实现。 */
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void prepare_input(conv_config_t *c) {
    uint32_t rng   = 0x31415926U ^ c->width ^ (c->height << 16) ^
                     (c->kernel_size << 8) ^ c->mode16;
    uint32_t bytes = c->mode16 != 0U ? 2U : 1U;
    memset(&input_buffer, 0x5a, c->src_stride * c->height);
    for (uint32_t y = 0; y < c->height; ++y) {
        for (uint32_t x = 0; x < c->width; ++x) {
            uint32_t index = (y * c->src_stride) / bytes + x;
            if (c->mode16 != 0U) {
                input_buffer.pixels16[index] =
                    (int16_t)((int32_t)(random_next(&rng) % 2047U) - 1023);
            } else {
                input_buffer.pixels8[index] =
                    (int8_t)((int32_t)(random_next(&rng) % 63U) - 31);
            }
        }
    }
    for (uint32_t i = 0; i < 25U; ++i) {
        uint32_t range = c->mode16 != 0U ? 33U : 9U;
        c->weights[i]  = (int16_t)((int32_t)(random_next(&rng) % range) -
                                   (int32_t)(range / 2U));
    }
    c->bias         = c->mode16 != 0U ? -12345 : -17;
    c->shift        = c->mode16 != 0U ? 10U : 4U;
    c->round_enable = BENCH_ROUND_ENABLE;
}

static int32_t output_pixel(
    const image_buffer_t *buffer, const conv_config_t *c, uint32_t x, uint32_t y
) {
    if (c->mode16 != 0U) {
        return buffer->pixels16[(y * c->dst_stride) / 2U + x];
    }
    return buffer->pixels8[y * c->dst_stride + x];
}

static uint32_t
output_checksum(const image_buffer_t *buffer, const conv_config_t *c) {
    uint32_t hash   = 2166136261U;
    uint32_t width  = c->width - c->kernel_size + 1U;
    uint32_t height = c->height - c->kernel_size + 1U;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            hash = (hash ^ (uint32_t)output_pixel(buffer, c, x, y)) * 16777619U;
        }
    }
    return hash;
}

/* 全部有效像素逐个核对，另检查行尾填充仍为哨兵值 0xa5。 */
static int check_output(const conv_config_t *c) {
    uint32_t             width     = c->width - c->kernel_size + 1U;
    uint32_t             height    = c->height - c->kernel_size + 1U;
    uint32_t             row_bytes = width * (c->mode16 != 0U ? 2U : 1U);
    uint32_t             errors    = 0;
    const unsigned char *hw_bytes  = (const unsigned char *)&hw_buffer;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            int32_t expected = output_pixel(&cpu_buffer, c, x, y);
            int32_t actual   = output_pixel(&hw_buffer, c, x, y);
            if (expected != actual) {
                if (errors < 8U) {
                    printf(
                        "  MISMATCH x=%" PRIu32 " y=%" PRIu32 " CPU=%" PRId32
                        " HW=%" PRId32 "\r\n",
                        x, y, expected, actual
                    );
                }
                ++errors;
            }
        }
        for (uint32_t b = row_bytes; b < c->dst_stride; ++b) {
            if (hw_bytes[y * c->dst_stride + b] != 0xa5U) {
                if (errors < 8U) {
                    printf(
                        "  PADDING MODIFIED row=%" PRIu32 " byte=%" PRIu32
                        "\r\n",
                        y, b
                    );
                }
                ++errors;
            }
        }
    }
    if (errors != 0U) {
        printf("  FAIL: %" PRIu32 " mismatches/padding errors\r\n", errors);
        return -1;
    }
    return 0;
}

static uint64_t read_pl_cycles(void) {
    uint32_t high_before;
    uint32_t low;
    uint32_t high_after;
    do {
        high_before = read_register(REG_CYCLES_HI);
        low         = read_register(REG_CYCLES_LO);
        high_after  = read_register(REG_CYCLES_HI);
    }
    while (high_before != high_after);
    return ((uint64_t)high_after << 32) | low;
}

/* 一次硬件任务。轮询用于保持例程简单，中断不参与测量。
 * total_ticks 从配置前开始，到完成输出缓存失效结束。
 */
static int run_hardware(const conv_config_t *c, hardware_sample_t *sample) {
    uint32_t out_height   = c->height - c->kernel_size + 1U;
    uint32_t input_bytes  = ALIGN_UP(c->src_stride * c->height, 64U);
    uint32_t output_bytes = ALIGN_UP(c->dst_stride * out_height, 64U);
    void    *source       = image_data(&input_buffer, c->mode16);
    void    *target       = image_data(&hw_buffer, c->mode16);
    if ((read_register(REG_STATUS) & STATUS_BUSY) != 0U) {
        printf("  FAIL: accelerator was already busy\r\n");
        return -1;
    }

    uint64_t total_start = timestamp();
    write_register(REG_IRQ_ENABLE, 0U);
    write_register(REG_STATUS, STATUS_DONE | STATUS_ERROR | STATUS_REJECT);
    write_register(
        REG_CONFIG, (c->kernel_size == 5U ? 1U : 0U) | (c->mode16 << 1) |
                        (c->round_enable << 2)
    );
    write_register(REG_WIDTH, c->width);
    write_register(REG_HEIGHT, c->height);
    write_register(REG_SRC_ADDR, (uint32_t)(UINTPTR)source);
    write_register(REG_DST_ADDR, (uint32_t)(UINTPTR)target);
    write_register(REG_SRC_STRIDE, c->src_stride);
    write_register(REG_DST_STRIDE, c->dst_stride);
    write_register(REG_OUT_SHIFT, c->shift);
    write_register(REG_BIAS, (uint32_t)c->bias);
    for (uint32_t i = 0; i < c->kernel_size * c->kernel_size; ++i) {
        write_register(
            REG_WEIGHT_BASE + 4U * i, (uint32_t)(uint16_t)c->weights[i]
        );
    }

    /* Flush 输入使 DDR 可见；flush 输出清除可能在 DMA 后回写的旧脏行。
     * 对齐范围完全位于本例程独占的缓冲区内。
     */
    Xil_DCacheFlushRange((INTPTR)source, input_bytes);
    Xil_DCacheFlushRange((INTPTR)target, output_bytes);
    uint64_t run_start = timestamp();
    write_register(REG_CONTROL, 1U);

    uint32_t polls = 0;
    uint32_t status;
    for (;;) {
        status = read_register(REG_STATUS);
        if ((status & (STATUS_DONE | STATUS_ERROR | STATUS_REJECT)) != 0U) {
            break;
        }
        /* 每 256 次轮询检查一次超时，减少读取计时器带来的干扰。 */
        if ((polls++ & 255U) == 0U &&
            timestamp() - run_start >
                (uint64_t)COUNTS_PER_SECOND * BENCH_TIMEOUT_SECONDS) {
            printf(
                "  TIMEOUT: status=0x%08" PRIx32
                "; stop benchmark and reset the hardware before retrying\r\n",
                status
            );
            /* 不复用缓冲区，也不伪造任务完成；AXI 可能仍有未完成事务。 */
            return -1;
        }
    }
    uint64_t run_end = timestamp();
    if ((status & (STATUS_ERROR | STATUS_REJECT | STATUS_BUSY)) != 0U) {
        printf(
            "  FAIL: status=0x%08" PRIx32 " error_code=%" PRIu32 "\r\n", status,
            read_register(REG_ERROR_CODE)
        );
        return -1;
    }

    /* 此时最后一个 AXI B 响应已返回，才能失效并由 CPU 读取输出。 */
    Xil_DCacheInvalidateRange((INTPTR)target, output_bytes);
    uint64_t total_end = timestamp();
    if (measure_elapsed(
            "HW START -> DONE", run_start, run_end, &sample->execution_ticks
        ) != 0 ||
        measure_elapsed(
            "HW total incl. cache", total_start, total_end, &sample->total_ticks
        ) != 0) {
        return -1;
    }
    sample->pl_cycles = read_pl_cycles();

    if (read_register(REG_OUT_WIDTH) != c->width - c->kernel_size + 1U ||
        read_register(REG_OUT_HEIGHT) != out_height) {
        printf("  FAIL: output dimensions disagree with configuration\r\n");
        return -1;
    }
    return 0;
}

static int benchmark_case(
    uint32_t width, uint32_t height, uint32_t kernel, uint32_t mode16
) {
    conv_config_t c       = {0};
    c.width               = width;
    c.height              = height;
    c.kernel_size         = kernel;
    c.mode16              = mode16;
    uint32_t bytes        = mode16 != 0U ? 2U : 1U;
    c.src_stride          = ALIGN_UP(width * bytes, 8U);
    c.dst_stride          = ALIGN_UP((width - kernel + 1U) * bytes, 8U);
    uint32_t output_bytes = c.dst_stride * (height - kernel + 1U);
    prepare_input(&c);
    memset(&cpu_buffer, 0xa5, output_bytes);

    void          *source       = image_data(&input_buffer, mode16);
    void          *cpu_target   = image_data(&cpu_buffer, mode16);
    timing_stats_t cpu          = {0, UINT64_MAX, 0};
    timing_stats_t hw_execution = {0, UINT64_MAX, 0};
    timing_stats_t hw_total     = {0, UINT64_MAX, 0};
    timing_stats_t pl_cycles    = {0, UINT64_MAX, 0};

    /* CPU 先预热一次，再测量同一输入的多次调用，缓存保持启用。 */
    conv_cpu_run(&c, source, cpu_target);
    for (uint32_t repeat = 0; repeat < BENCH_REPEATS; ++repeat) {
        uint64_t start = timestamp();
        conv_cpu_run(&c, source, cpu_target);
        uint64_t end = timestamp();
        uint64_t elapsed;
        if (measure_elapsed("CPU scalar C", start, end, &elapsed) != 0) {
            return -1;
        }
        add_sample(&cpu, elapsed);
        /* 计时外观察每次结果，避免优化器将计算视为无用。 */
        checksum_sink ^= output_checksum(&cpu_buffer, &c);
    }

    /* 硬件也预热一次。所有正式重复都做完整配置和缓存维护。
     * memset / 比对 / checksum / UART 输出不计入硬件时间。
     */
    for (uint32_t repeat = 0; repeat <= BENCH_REPEATS; ++repeat) {
        hardware_sample_t sample;
        memset(&hw_buffer, 0xa5, output_bytes);
        if (run_hardware(&c, &sample) != 0 || check_output(&c) != 0) {
            return -1;
        }
        checksum_sink ^= output_checksum(&hw_buffer, &c);
        if (repeat != 0U) {
            add_sample(&hw_execution, sample.execution_ticks);
            add_sample(&hw_total, sample.total_ticks);
            add_sample(&pl_cycles, sample.pl_cycles);
        }
    }

    uint32_t pixels = (width - kernel + 1U) * (height - kernel + 1U);
    printf(
        "  PASS: %" PRIu32 " pixels, checksum=0x%08" PRIx32 "\r\n", pixels,
        output_checksum(&cpu_buffer, &c)
    );
    print_stats("CPU scalar C", &cpu, COUNTS_PER_SECOND);
    print_stats("HW START -> DONE", &hw_execution, COUNTS_PER_SECOND);
    print_stats("HW total incl. cache", &hw_total, COUNTS_PER_SECOND);
    printf(
        "  PL cycles/frame        avg=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64
        "\r\n",
        pl_cycles.sum / BENCH_REPEATS, pl_cycles.minimum, pl_cycles.maximum
    );
    print_stats("PL cycles -> time", &pl_cycles, BENCH_PL_CLOCK_HZ);

    /* 每个样本已经检查为正，分母非零；不能把无效计时伪装成 0x。 */
    uint64_t speedup = cpu.sum * 1000U / hw_total.sum;
    printf(
        "  CPU / HW total speedup = %" PRIu64 ".%03" PRIu64 "x\r\n",
        speedup / 1000U, speedup % 1000U
    );
    uint64_t average_ns =
        ticks_to_ns(hw_total.sum / BENCH_REPEATS, COUNTS_PER_SECOND);
    if (average_ns != 0U) {
        uint64_t pixels_per_second =
            (uint64_t)pixels * UINT64_C(1000000000) / average_ns;
        printf(
            "  HW total throughput    = %" PRIu64 " output pixels/s\r\n",
            pixels_per_second
        );
    }
    return 0;
}

int main(void) {
    static const uint32_t sizes[][2] = {
        { 64U,  64U},
        {256U, 256U},
        {512U, 512U}
    };
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    printf("\r\n=== CPU C vs AXI convolution benchmark ===\r\n");
    if (initialize_timer() != 0) {
        return 1;
    }
    printf(
        "CPU clock=%" PRIu32 " Hz, timer=%" PRIu64 " Hz, PL assumed=%u Hz\r\n",
        (uint32_t)XPAR_CPU_CORE_CLOCK_FREQ_HZ, (uint64_t)COUNTS_PER_SECOND,
        BENCH_PL_CLOCK_HZ
    );
#if defined(__OPTIMIZE__)
    printf("CPU compiler optimization enabled (project default: -O3).\r\n");
#else
    printf(
        "WARNING: CPU optimization disabled; rebuild with -O3 for "
        "comparison.\r\n"
    );
#endif
    printf(
        "Single core, scalar C, signed int64 accumulator, caches enabled.\r\n"
    );
    printf(
        "Warm-up excluded; %u timed repetitions; reused input per case.\r\n",
        BENCH_REPEATS
    );
    printf(
        "Speedup uses HW total including configuration and cache "
        "maintenance.\r\n"
    );
    printf(
        "HW START->DONE includes polling; PL cycles include "
        "validation/drain.\r\n"
    );

    uint32_t id = read_register(REG_ID);
    if (id != 0x434f4e56U) {
        printf(
            "FAIL: CONV ID=0x%08" PRIx32 " at 0x%08" PRIx32 "\r\n", id,
            (uint32_t)ACCEL_BASE
        );
        return 1;
    }
    uint32_t max_width  = read_register(REG_MAX_WIDTH);
    uint32_t max_height = read_register(REG_MAX_HEIGHT);
    printf(
        "Accelerator base=0x%08" PRIx32 ", version=0x%08" PRIx32
        ", limits=%" PRIu32 "x%" PRIu32 "\r\n",
        (uint32_t)ACCEL_BASE, read_register(REG_VERSION), max_width, max_height
    );

    for (size_t size = 0; size < sizeof(sizes) / sizeof(sizes[0]); ++size) {
        uint32_t width  = sizes[size][0];
        uint32_t height = sizes[size][1];
        if (width > BENCH_MAX_WIDTH || height > BENCH_MAX_HEIGHT ||
            width > max_width || height > max_height) {
            printf(
                "SKIP %" PRIu32 "x%" PRIu32
                ": exceeds buffer/hardware limits\r\n",
                width, height
            );
            continue;
        }
        for (uint32_t mode16 = 0; mode16 <= 1U; ++mode16) {
            for (uint32_t kernel = 3U; kernel <= 5U; kernel += 2U) {
                printf(
                    "\r\nInput=%" PRIu32 "x%" PRIu32 ", kernel=%" PRIu32
                    "x%" PRIu32 ", signed %u-bit\r\n",
                    width, height, kernel, kernel, mode16 != 0U ? 16U : 8U
                );
                if (benchmark_case(width, height, kernel, mode16) != 0) {
                    printf(
                        "Benchmark aborted; no performance claim for a failed "
                        "case.\r\n"
                    );
                    return 1;
                }
            }
        }
    }
    printf(
        "\r\nAll selected cases passed. Sink=0x%08" PRIx32 "\r\n", checksum_sink
    );
    return 0;
}
