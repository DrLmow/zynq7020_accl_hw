// AXI4-Lite register bank. AW and W are captured independently: neither
// channel waits for the other channel's VALID before asserting READY.
module axi_lite_regs #(
    parameter int MAX_WIDTH  = 2048,
    parameter int MAX_HEIGHT = 8192
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [ 3:0] s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [ 1:0] s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [31:0] s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [ 1:0] s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    input  logic               busy,
    input  logic               done_event,
    input  logic               error_event,
    input  logic               reject_event,
    input  logic        [31:0] error_code,
    input  logic        [15:0] out_width,
    input  logic        [15:0] out_height,
    input  logic        [63:0] cycle_count,
    output logic               start_request,
    output logic               irq,
    output logic        [ 2:0] cfg,
    output logic        [15:0] width,
    output logic        [15:0] height,
    output logic        [31:0] src_addr,
    output logic        [31:0] dst_addr,
    output logic        [31:0] src_stride,
    output logic        [31:0] dst_stride,
    output logic        [ 4:0] out_shift,
    output logic signed [31:0] bias,
    output logic        [15:0] weights      [0:24]
);
    import conv_accel_pkg::*;
    logic        aw_pending;
    logic        w_pending;
    logic [11:0] write_addr;
    logic [31:0] write_data;
    logic [ 3:0] write_strobe;
    logic        done_sticky;
    logic        error_sticky;
    logic        reject_sticky;
    logic [ 1:0] irq_enable;
    logic [31:0] saved_error;
    logic [31:0] read_value;
    logic        read_mapped;
    logic        unused_address_upper;
    assign unused_address_upper = ^{s_axi_awaddr[31:12], s_axi_araddr[31:12]};

    // A full byte merge is used before masking each register's valid bits.
    function automatic logic [31:0] merge_bytes(
        input logic [31:0] old_value,  // Register contents before the write.
        input logic [31:0] new_value,  // AXI-Lite write data.
        input logic [3:0] strobe      // Byte lanes enabled by WSTRB.
    );
        logic [31:0] result;
        result = old_value;
        for (int byte_id = 0; byte_id < 4; byte_id++) begin
            if (strobe[byte_id]) begin
                result[byte_id*8+:8] = new_value[byte_id*8+:8];
            end
        end
        return result;
    endfunction

    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready  = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    assign irq           = (done_sticky && irq_enable[0]) || (error_sticky && irq_enable[1]);

    // Only the low 12 address bits are decoded; the interconnect selects
    // this slave's 4 KiB aperture using the upper system address bits.
    always_comb begin
        read_value  = 0;
        read_mapped = 1;
        case (s_axi_araddr[11:0])
            REG_ID: begin
                read_value = 32'h434f4e56;
            end
            REG_VERSION: begin
                read_value = 32'h00010000;
            end
            REG_CONTROL: begin
                read_value = 0;
            end
            REG_STATUS: begin
                read_value = {28'b0, reject_sticky, error_sticky, done_sticky, busy};
            end
            REG_IRQ_ENABLE: begin
                read_value = {30'b0, irq_enable};
            end
            REG_ERROR_CODE: begin
                read_value = saved_error;
            end
            REG_CONFIG: begin
                read_value = {29'b0, cfg};
            end
            REG_WIDTH: begin
                read_value = {16'b0, width};
            end
            REG_HEIGHT: begin
                read_value = {16'b0, height};
            end
            REG_SRC_ADDR: begin
                read_value = src_addr;
            end
            REG_DST_ADDR: begin
                read_value = dst_addr;
            end
            REG_SRC_STRIDE: begin
                read_value = src_stride;
            end
            REG_DST_STRIDE: begin
                read_value = dst_stride;
            end
            REG_OUT_SHIFT: begin
                read_value = {27'b0, out_shift};
            end
            REG_BIAS: begin
                read_value = bias;
            end
            REG_OUT_WIDTH: begin
                read_value = {16'b0, out_width};
            end
            REG_OUT_HEIGHT: begin
                read_value = {16'b0, out_height};
            end
            REG_CYCLES_LO: begin
                read_value = cycle_count[31:0];
            end
            REG_CYCLES_HI: begin
                read_value = cycle_count[63:32];
            end
            REG_MAX_WIDTH: begin
                read_value = 32'(MAX_WIDTH);
            end
            REG_MAX_HEIGHT: begin
                read_value = 32'(MAX_HEIGHT);
            end
            REG_CAPABILITIES: begin
                read_value = 32'h1f;
            end
            default: begin
                read_mapped = 0;
                for (int i = 0; i < 25; i++) begin
                    if (s_axi_araddr[11:0] == REG_WEIGHT_BASE + 12'(4 * i)) begin
                        read_value  = {16'b0, weights[i]};
                        read_mapped = 1;
                    end
                end
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            aw_pending    <= 0;
            w_pending     <= 0;
            write_addr    <= 0;
            write_data    <= 0;
            write_strobe  <= 0;
            s_axi_bvalid  <= 0;
            s_axi_bresp   <= 0;
            s_axi_rvalid  <= 0;
            s_axi_rdata   <= 0;
            s_axi_rresp   <= 0;
            start_request <= 0;
            done_sticky   <= 0;
            error_sticky  <= 0;
            reject_sticky <= 0;
            saved_error   <= 0;
            irq_enable    <= 0;
            cfg           <= 3'b100;
            width         <= 0;
            height        <= 0;
            src_addr      <= 0;
            dst_addr      <= 0;
            src_stride    <= 0;
            dst_stride    <= 0;
            out_shift     <= 0;
            bias          <= 0;
            for (int i = 0; i < 25; i++) weights[i] <= 0;
        end else begin
            start_request <= 0;
            if (s_axi_awvalid && s_axi_awready) begin
                write_addr <= s_axi_awaddr[11:0];
                aw_pending <= 1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                write_data   <= s_axi_wdata;
                write_strobe <= s_axi_wstrb;
                w_pending    <= 1;
            end
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 0;
            end
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 0;
            end
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rdata  <= read_value;
                s_axi_rresp  <= read_mapped ? 2'b00 : 2'b10;
            end

            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending   <= 0;
                w_pending    <= 0;
                s_axi_bvalid <= 1;
                s_axi_bresp  <= 0;
                case (write_addr)
                    REG_CONTROL: begin
                        if (write_strobe[0] && write_data[0]) begin
                            start_request <= 1;
                            // Make rejection visible when the write response
                            // is issued, rather than after controller latency.
                            if (busy) begin
                                reject_sticky <= 1;
                            end else begin
                                done_sticky   <= 0;
                                error_sticky  <= 0;
                                reject_sticky <= 0;
                                saved_error   <= 0;
                            end
                        end
                    end
                    REG_STATUS: begin
                        if (write_strobe[0]) begin
                            if (write_data[1]) begin
                                done_sticky <= 0;
                            end
                            if (write_data[2]) begin
                                error_sticky <= 0;
                                saved_error  <= 0;
                            end
                            if (write_data[3]) begin
                                reject_sticky <= 0;
                            end
                        end
                    end
                    REG_IRQ_ENABLE: begin
                        irq_enable <=
                            2'(merge_bytes({30'b0, irq_enable}, write_data, write_strobe));
                    end
                    default: begin
                        // Configuration is immutable while a task is active.
                        if (busy) begin
                            s_axi_bresp <= 2'b10;
                        end else begin
                            case (write_addr)
                                REG_CONFIG: begin
                                    cfg <= 3'(merge_bytes({29'b0, cfg}, write_data, write_strobe));
                                end
                                REG_WIDTH: begin
                                    width <=
                                        16'(merge_bytes({16'b0, width}, write_data, write_strobe));
                                end
                                REG_HEIGHT: begin
                                    height <=
                                        16'(merge_bytes({16'b0, height}, write_data, write_strobe));
                                end
                                REG_SRC_ADDR: begin
                                    src_addr <= merge_bytes(src_addr, write_data, write_strobe);
                                end
                                REG_DST_ADDR: begin
                                    dst_addr <= merge_bytes(dst_addr, write_data, write_strobe);
                                end
                                REG_SRC_STRIDE: begin
                                    src_stride <= merge_bytes(src_stride, write_data, write_strobe);
                                end
                                REG_DST_STRIDE: begin
                                    dst_stride <= merge_bytes(dst_stride, write_data, write_strobe);
                                end
                                REG_OUT_SHIFT: begin
                                    out_shift <= 5'(
                                        merge_bytes({27'b0, out_shift}, write_data, write_strobe));
                                end
                                REG_BIAS: begin
                                    bias <= $signed(merge_bytes(bias, write_data, write_strobe));
                                end
                                default: begin
                                    s_axi_bresp <= 2'b10;
                                    for (int i = 0; i < 25; i++) begin
                                        if (write_addr == REG_WEIGHT_BASE + 12'(4 * i)) begin
                                            weights[i] <= 16'(merge_bytes(
                                                {16'b0, weights[i]}, write_data, write_strobe
                                            ));
                                            s_axi_bresp <= 0;
                                        end
                                    end
                                end
                            endcase
                        end
                    end
                endcase
            end
            // Hardware events win over a simultaneous software W1C.
            if (done_event) begin
                done_sticky <= 1;
            end
            if (reject_event) begin
                reject_sticky <= 1;
            end
            if (error_event) begin
                error_sticky <= 1;
                saved_error  <= error_code;
            end
        end
    end
endmodule
