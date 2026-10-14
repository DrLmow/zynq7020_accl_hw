// Single-channel VALID convolution accelerator, one shared clock domain.
// AXI4-Lite configures a task; AXI4 reads/writes 64-bit aligned DDR beats.
// Reset assertion/release must be coordinated with the surrounding fabric.
module conv_accel_top #(
    parameter int MAX_WIDTH  = 2048,
    parameter int MAX_HEIGHT = 8192
) (
    input  logic        clk,
    input  logic        rst_n,
    output logic        irq,
    input  logic [31:0] s_axi_awaddr,
    input  logic [ 2:0] s_axi_awprot,
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
    input  logic [ 2:0] s_axi_arprot,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [ 1:0] s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    output logic [ 0:0] m_axi_arid,
    output logic [31:0] m_axi_araddr,
    output logic [ 7:0] m_axi_arlen,
    output logic [ 2:0] m_axi_arsize,
    output logic [ 1:0] m_axi_arburst,
    output logic        m_axi_arlock,
    output logic [ 3:0] m_axi_arcache,
    output logic [ 2:0] m_axi_arprot,
    output logic [ 3:0] m_axi_arqos,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [ 0:0] m_axi_rid,
    input  logic [63:0] m_axi_rdata,
    input  logic [ 1:0] m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic [ 0:0] m_axi_awid,
    output logic [31:0] m_axi_awaddr,
    output logic [ 7:0] m_axi_awlen,
    output logic [ 2:0] m_axi_awsize,
    output logic [ 1:0] m_axi_awburst,
    output logic        m_axi_awlock,
    output logic [ 3:0] m_axi_awcache,
    output logic [ 2:0] m_axi_awprot,
    output logic [ 3:0] m_axi_awqos,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [ 7:0] m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [ 0:0] m_axi_bid,
    input  logic [ 1:0] m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready
);
    import conv_accel_pkg::*;
    typedef enum logic [3:0] {
        IDLE,
        PREPARE_CONFIG,
        CHECK_CONFIG,
        CALCULATE_SPANS,
        ADD_BASES,
        ADD_ROW_BYTES,
        CHECK_RANGES,
        RUN,
        DRAIN_ERROR
    } state_t;
    state_t             state;
    logic               busy;
    logic               start_request;
    logic               engine_start;
    logic               cancel;
    logic               done_event;
    logic               error_event;
    logic               reject_event;
    logic        [31:0] saved_fault;
    logic        [63:0] cycle_count;
    logic        [ 2:0] cfg;
    logic        [ 2:0] task_cfg;
    logic        [15:0] width;
    logic        [15:0] height;
    logic        [15:0] task_width;
    logic        [15:0] task_height;
    logic        [15:0] out_width;
    logic        [15:0] out_height;
    logic        [31:0] src_addr;
    logic        [31:0] dst_addr;
    logic        [31:0] src_stride;
    logic        [31:0] dst_stride;
    logic        [31:0] task_src;
    logic        [31:0] task_dst;
    logic        [31:0] task_src_stride;
    logic        [31:0] task_dst_stride;
    logic        [ 4:0] out_shift;
    logic        [ 4:0] task_shift;
    logic signed [31:0] bias;
    logic signed [31:0] task_bias;
    logic        [15:0] weights           [0:24];
    logic signed [15:0] task_weights      [0:24];
    logic        [31:0] validation_error;
    logic        [ 2:0] kernel_size;
    logic        [17:0] source_row_bytes;
    logic        [17:0] target_row_bytes;
    // 16-bit row counts multiplied by 32-bit strides need 48 bits.
    // End-exclusive addresses need one extra bit for the base and row bytes.
    logic        [47:0] source_span;
    logic        [47:0] target_span;
    logic        [47:0] source_step;
    logic        [47:0] target_step;
    logic        [15:0] source_rows;
    logic        [15:0] target_rows;
    logic        [ 3:0] span_bit;
    logic        [48:0] source_end;
    logic        [48:0] target_end;

    logic               rd_idle;
    logic               rd_done;
    logic               rd_fault;
    logic               wr_idle;
    logic               wr_done;
    logic               wr_fault;
    logic               read_finished;
    logic        [31:0] rd_fault_code;
    logic signed [15:0] input_pixel;
    logic signed [15:0] output_pixel;
    logic               input_valid;
    logic               input_ready;
    logic               window_valid;
    logic               window_ready;
    logic signed [15:0] window            [ 0:4] [0:4];
    logic signed [39:0] accumulator;
    logic               result_valid;
    logic               result_ready;
    logic               output_valid;
    logic               output_ready;
    logic        [63:0] packed_data;
    logic        [ 7:0] packed_strobe;
    logic               packed_row_last;
    logic               packed_frame_last;
    logic               packed_valid;
    logic               packed_ready;

    // This implementation permits only ID 0 and one outstanding transaction
    // per direction; protection attributes do not alter register semantics.
    // Explicitly consume unused input attributes to keep lint clean.
    logic               unused_attributes;
    assign unused_attributes = ^{s_axi_awprot, s_axi_arprot, m_axi_rid, m_axi_bid};
    assign m_axi_arid        = 0;
    assign m_axi_awid        = 0;
    assign m_axi_arsize      = 3'd3;
    assign m_axi_awsize      = 3'd3;
    assign m_axi_arburst     = 2'b01;
    assign m_axi_awburst     = 2'b01;
    assign m_axi_arlock      = 0;
    assign m_axi_awlock      = 0;
    assign m_axi_arcache     = 0;
    assign m_axi_awcache     = 0;
    assign m_axi_arprot      = 0;
    assign m_axi_awprot      = 0;
    assign m_axi_arqos       = 0;
    assign m_axi_awqos       = 0;
    assign busy              = state != IDLE;
    assign cancel            = state == DRAIN_ERROR;

    assign kernel_size       = task_cfg[0] ? 3'd5 : 3'd3;

    // Check simple constraints first. Address arithmetic is performed in
    // separate states below, so no subtract/multiply/add/compare chain needs
    // to complete in one clock. Error priority remains dimension -> alignment
    // -> stride -> address overflow -> overlap.
    always_comb begin
        validation_error = ERR_NONE;
        if (task_width < 16'(kernel_size) || task_height < 16'(kernel_size) ||
            int'(task_width) > MAX_WIDTH || int'(task_height) > MAX_HEIGHT) begin
            validation_error = ERR_DIMENSION;
        end else if ((task_src[2:0] != 0) || (task_dst[2:0] != 0) ||
                     (task_src_stride[2:0] != 0) || (task_dst_stride[2:0] != 0)) begin
            validation_error = ERR_ALIGNMENT;
        end else if (task_src_stride < {14'd0, source_row_bytes} ||
                     task_dst_stride < {14'd0, target_row_bytes}) begin
            validation_error = ERR_STRIDE;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= IDLE;
            task_cfg         <= 0;
            task_width       <= 0;
            task_height      <= 0;
            task_src         <= 0;
            task_dst         <= 0;
            task_src_stride  <= 0;
            task_dst_stride  <= 0;
            task_shift       <= 0;
            task_bias        <= 0;
            out_width        <= 0;
            out_height       <= 0;
            source_row_bytes <= 0;
            target_row_bytes <= 0;
            source_span      <= 0;
            target_span      <= 0;
            source_step      <= 0;
            target_step      <= 0;
            source_rows      <= 0;
            target_rows      <= 0;
            span_bit         <= 0;
            source_end       <= 0;
            target_end       <= 0;
            engine_start     <= 0;
            done_event       <= 0;
            error_event      <= 0;
            reject_event     <= 0;
            saved_fault      <= ERR_NONE;
            read_finished    <= 0;
            cycle_count      <= 0;
            for (int i = 0; i < 25; i++) task_weights[i] <= 0;
        end else begin
            engine_start <= 0;
            done_event   <= 0;
            error_event  <= 0;
            reject_event <= 0;
            if (busy) begin
                cycle_count <= cycle_count + 1;
            end
            if (start_request && busy) begin
                reject_event <= 1;
            end
            case (state)
                IDLE: begin
                    if (start_request) begin
                        // Freeze the whole task. No live configuration reaches DMA.
                        task_cfg        <= cfg;
                        task_width      <= width;
                        task_height     <= height;
                        task_src        <= src_addr;
                        task_dst        <= dst_addr;
                        task_src_stride <= src_stride;
                        task_dst_stride <= dst_stride;
                        task_shift      <= out_shift;
                        task_bias       <= bias;
                        for (int i = 0; i < 25; i++) begin
                            task_weights[i] <= cfg[1] ? $signed(weights[i]) :
                                $signed({{8{weights[i][7]}}, weights[i][7:0]});
                        end
                        cycle_count   <= 0;
                        read_finished <= 0;
                        state         <= PREPARE_CONFIG;
                    end
                end
                PREPARE_CONFIG: begin
                    // A row can contain 65535 * 2 bytes; alignment to eight
                    // can round that up to 131072, which requires 18 bits.
                    source_row_bytes <= (({2'd0, task_width} << task_cfg[1]) + 18'd7) & ~18'd7;
                    target_row_bytes <= ((({2'd0, task_width} - 18'(kernel_size) +
                                          18'd1) << task_cfg[1]) + 18'd7) & ~18'd7;
                    source_rows <= task_height - 16'd1;
                    target_rows <= task_height - 16'(kernel_size);
                    source_step <= {16'd0, task_src_stride};
                    target_step <= {16'd0, task_dst_stride};
                    source_span <= 0;
                    target_span <= 0;
                    span_bit <= 0;
                    // Invalid dimensions can underflow the temporary values
                    // above, but CHECK_CONFIG rejects them before using them.
                    state <= CHECK_CONFIG;
                end
                CHECK_CONFIG: begin
                    if (validation_error != ERR_NONE) begin
                        saved_fault <= validation_error;
                        error_event <= 1;
                        state       <= IDLE;
                    end else begin
                        state <= CALCULATE_SPANS;
                    end
                end
                CALCULATE_SPANS: begin
                    // Shift-and-add multiplication: consume one row-count bit
                    // per clock. Both directions run in parallel for 16 clocks.
                    // Startup cost is per task, not per pixel or kernel row.
                    if (source_rows[0]) begin
                        source_span <= source_span + source_step;
                    end
                    if (target_rows[0]) begin
                        target_span <= target_span + target_step;
                    end
                    source_rows <= {1'b0, source_rows[15:1]};
                    target_rows <= {1'b0, target_rows[15:1]};
                    source_step <= {source_step[46:0], 1'b0};
                    target_step <= {target_step[46:0], 1'b0};
                    if (span_bit == 4'd15) begin
                        state <= ADD_BASES;
                    end else begin
                        span_bit <= span_bit + 4'd1;
                    end
                end
                ADD_BASES: begin
                    source_end <= {1'b0, source_span} + {17'd0, task_src};
                    target_end <= {1'b0, target_span} + {17'd0, task_dst};
                    state      <= ADD_ROW_BYTES;
                end
                ADD_ROW_BYTES: begin
                    source_end <= source_end + {31'd0, source_row_bytes};
                    target_end <= target_end + {31'd0, target_row_bytes};
                    state      <= CHECK_RANGES;
                end
                CHECK_RANGES: begin
                    // Wide values retain overflow above the AXI address width.
                    // An exclusive end exactly equal to 2^32 is still legal.
                    if (source_end > 49'h1_0000_0000 || target_end > 49'h1_0000_0000) begin
                        saved_fault <= ERR_ADDRESS;
                        error_event <= 1;
                        state       <= IDLE;
                    end else if ({17'd0, task_src} < target_end &&
                                 {17'd0, task_dst} < source_end) begin
                        saved_fault <= ERR_OVERLAP;
                        error_event <= 1;
                        state       <= IDLE;
                    end else begin
                        out_width    <= task_width - 16'(kernel_size) + 1;
                        out_height   <= task_height - 16'(kernel_size) + 1;
                        engine_start <= 1;
                        state        <= RUN;
                    end
                end
                RUN: begin
                    if (rd_done) begin
                        read_finished <= 1;
                    end
                    if (rd_fault || wr_fault) begin
                        saved_fault <= rd_fault ? rd_fault_code : ERR_WRITE;
                        state       <= DRAIN_ERROR;
                    end else if (wr_done && (read_finished || rd_done)) begin
                        done_event <= 1;
                        state      <= IDLE;
                    end
                end
                DRAIN_ERROR: begin
                    if (rd_idle && wr_idle) begin
                        // Report failure only after every committed transaction ends.
                        error_event <= 1;
                        state       <= IDLE;
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    axi_lite_regs #(
        .MAX_WIDTH (MAX_WIDTH),
        .MAX_HEIGHT(MAX_HEIGHT)
    ) registers (
        .clk,
        .rst_n,
        .s_axi_awaddr,
        .s_axi_awvalid,
        .s_axi_awready,
        .s_axi_wdata,
        .s_axi_wstrb,
        .s_axi_wvalid,
        .s_axi_wready,
        .s_axi_bresp,
        .s_axi_bvalid,
        .s_axi_bready,
        .s_axi_araddr,
        .s_axi_arvalid,
        .s_axi_arready,
        .s_axi_rdata,
        .s_axi_rresp,
        .s_axi_rvalid,
        .s_axi_rready,
        .busy,
        .done_event,
        .error_event,
        .reject_event,
        .error_code(saved_fault),
        .out_width,
        .out_height,
        .cycle_count,
        .start_request,
        .irq,
        .cfg,
        .width,
        .height,
        .src_addr,
        .dst_addr,
        .src_stride,
        .dst_stride,
        .out_shift,
        .bias,
        .weights
    );
    axi_read_dma reader (
        .clk,
        .rst_n,
        .start      (engine_start),
        .cancel,
        .mode16     (task_cfg[1]),
        .width      (task_width),
        .height     (task_height),
        .base_addr  (task_src),
        .stride     (task_src_stride),
        .idle       (rd_idle),
        .done       (rd_done),
        .fault      (rd_fault),
        .fault_code (rd_fault_code),
        .pixel      (input_pixel),
        .pixel_valid(input_valid),
        .pixel_ready(input_ready),
        .m_axi_araddr,
        .m_axi_arlen,
        .m_axi_arvalid,
        .m_axi_arready,
        .m_axi_rdata,
        .m_axi_rresp,
        .m_axi_rlast,
        .m_axi_rvalid,
        .m_axi_rready
    );
    conv_window #(
        .MAX_WIDTH(MAX_WIDTH)
    ) window_generator (
        .clk,
        .rst_n,
        .start      (engine_start),
        .cancel,
        .kernel5    (task_cfg[0]),
        .width      (task_width),
        .pixel      (input_pixel),
        .pixel_valid(input_valid),
        .pixel_ready(input_ready),
        .window,
        .window_valid,
        .window_ready
    );
    conv_mac mac (
        .clk,
        .rst_n,
        .start  (engine_start),
        .cancel,
        .kernel5(task_cfg[0]),
        .bias   (task_bias),
        .weights(task_weights),
        .window,
        .window_valid,
        .window_ready,
        .accumulator,
        .result_valid,
        .result_ready
    );
    conv_quantize quantizer (
        .clk,
        .rst_n,
        .start            (engine_start),
        .cancel,
        .mode16           (task_cfg[1]),
        .round_enable     (task_cfg[2]),
        .shift            (task_shift),
        .accumulator,
        .accumulator_valid(result_valid),
        .accumulator_ready(result_ready),
        .pixel            (output_pixel),
        .pixel_valid      (output_valid),
        .pixel_ready      (output_ready)
    );
    pixel_pack packer (
        .clk,
        .rst_n,
        .start          (engine_start),
        .cancel,
        .mode16         (task_cfg[1]),
        .width          (out_width),
        .height         (out_height),
        .pixel          (output_pixel),
        .pixel_valid    (output_valid),
        .pixel_ready    (output_ready),
        .beat_data      (packed_data),
        .beat_strobe    (packed_strobe),
        .beat_row_last  (packed_row_last),
        .beat_frame_last(packed_frame_last),
        .beat_valid     (packed_valid),
        .beat_ready     (packed_ready)
    );
    axi_write_dma writer (
        .clk,
        .rst_n,
        .start          (engine_start),
        .cancel,
        .base_addr      (task_dst),
        .stride         (task_dst_stride),
        .beat_data      (packed_data),
        .beat_strobe    (packed_strobe),
        .beat_row_last  (packed_row_last),
        .beat_frame_last(packed_frame_last),
        .beat_valid     (packed_valid),
        .beat_ready     (packed_ready),
        .idle           (wr_idle),
        .done           (wr_done),
        .fault          (wr_fault),
        .m_axi_awaddr,
        .m_axi_awlen,
        .m_axi_awvalid,
        .m_axi_awready,
        .m_axi_wdata,
        .m_axi_wstrb,
        .m_axi_wlast,
        .m_axi_wvalid,
        .m_axi_wready,
        .m_axi_bresp,
        .m_axi_bvalid,
        .m_axi_bready
    );

    // These are simulation/formal assertions, excluded from synthesis.
    // They verify protocol ownership, not the arithmetic implementation.
`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (!rst_n)
        m_axi_arvalid && !m_axi_arready |=> m_axi_arvalid &&
        $stable(
        {m_axi_araddr, m_axi_arlen}
    ));
    assert property (@(posedge clk) disable iff (!rst_n)
        m_axi_awvalid && !m_axi_awready |=> m_axi_awvalid &&
        $stable(
        {m_axi_awaddr, m_axi_awlen}
    ));
    assert property (@(posedge clk) disable iff (!rst_n)
        m_axi_wvalid && !m_axi_wready |=> m_axi_wvalid &&
        $stable(
        {m_axi_wdata, m_axi_wstrb, m_axi_wlast}
    ));
    assert property (@(posedge clk) disable iff (!rst_n)
        s_axi_rvalid && !s_axi_rready |=> s_axi_rvalid &&
        $stable(
        {s_axi_rdata, s_axi_rresp}
    ));
    assert property (@(posedge clk) disable iff (!rst_n)
        s_axi_bvalid && !s_axi_bready |=> s_axi_bvalid && $stable(
        s_axi_bresp
    ));
`endif
endmodule
