// Shared register offsets and error codes. Addresses are byte offsets.
package conv_accel_pkg;
    localparam logic [11:0] REG_ID = 12'h000;
    localparam logic [11:0] REG_VERSION = 12'h004;
    localparam logic [11:0] REG_CONTROL = 12'h008;
    localparam logic [11:0] REG_STATUS = 12'h00c;
    localparam logic [11:0] REG_IRQ_ENABLE = 12'h010;
    localparam logic [11:0] REG_ERROR_CODE = 12'h014;
    localparam logic [11:0] REG_CONFIG = 12'h018;
    localparam logic [11:0] REG_WIDTH = 12'h01c;
    localparam logic [11:0] REG_HEIGHT = 12'h020;
    localparam logic [11:0] REG_SRC_ADDR = 12'h024;
    localparam logic [11:0] REG_DST_ADDR = 12'h028;
    localparam logic [11:0] REG_SRC_STRIDE = 12'h02c;
    localparam logic [11:0] REG_DST_STRIDE = 12'h030;
    localparam logic [11:0] REG_OUT_SHIFT = 12'h034;
    localparam logic [11:0] REG_BIAS = 12'h038;
    localparam logic [11:0] REG_OUT_WIDTH = 12'h03c;
    localparam logic [11:0] REG_OUT_HEIGHT = 12'h040;
    localparam logic [11:0] REG_CYCLES_LO = 12'h044;
    localparam logic [11:0] REG_CYCLES_HI = 12'h048;
    localparam logic [11:0] REG_MAX_WIDTH = 12'h04c;
    localparam logic [11:0] REG_MAX_HEIGHT = 12'h050;
    localparam logic [11:0] REG_CAPABILITIES = 12'h054;
    localparam logic [11:0] REG_WEIGHT_BASE = 12'h100;

    typedef enum logic [31:0] {
        ERR_NONE = 0,
        ERR_DIMENSION = 1,
        ERR_ALIGNMENT = 2,
        ERR_STRIDE = 3,
        ERR_ADDRESS = 4,
        ERR_OVERLAP = 5,
        ERR_READ = 6,
        ERR_WRITE = 7,
        ERR_RLAST = 8
    } error_code_t;
endpackage
