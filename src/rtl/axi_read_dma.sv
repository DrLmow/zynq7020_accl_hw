// One outstanding read burst. A complete 16-beat local buffer reserves
// space before AR is issued. The buffered data is then unpacked into pixels.
// This intentionally simple schedule trades throughput for predictable flow.
module axi_read_dma (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               cancel,
    input  logic               mode16,
    input  logic        [15:0] width,
    input  logic        [15:0] height,
    input  logic        [31:0] base_addr,
    input  logic        [31:0] stride,
    output logic               idle,
    output logic               done,
    output logic               fault,
    output logic        [31:0] fault_code,
    output logic signed [15:0] pixel,
    output logic               pixel_valid,
    input  logic               pixel_ready,
    output logic        [31:0] m_axi_araddr,
    output logic        [ 7:0] m_axi_arlen,
    output logic               m_axi_arvalid,
    input  logic               m_axi_arready,
    input  logic        [63:0] m_axi_rdata,
    input  logic        [ 1:0] m_axi_rresp,
    input  logic               m_axi_rlast,
    input  logic               m_axi_rvalid,
    output logic               m_axi_rready
);
    import conv_accel_pkg::*;
    typedef enum logic [2:0] {
        IDLE,
        PLAN_BURST,
        ADDRESS,
        RECEIVE,
        STREAM
    } state_t;
    state_t        state;
    logic   [63:0] buffer_data      [0:15];
    logic   [31:0] address;
    logic   [31:0] row_base;
    logic   [15:0] x;
    logic   [15:0] y;
    logic   [ 4:0] burst_beats;
    logic   [ 4:0] receive_index;
    logic   [ 3:0] stream_index;
    logic   [ 2:0] byte_index;
    logic          failed;
    logic   [15:0] remaining_pixels;
    logic   [14:0] remaining_beats;
    logic   [ 9:0] boundary_beats;
    logic   [ 4:0] planned_beats;
    logic   [15:0] raw_pixel;

    always_comb begin
        // At most 65535 pixels, two bytes each. Keep the rounding carry.
        remaining_pixels = width - x;
        remaining_beats  = 15'((({2'd0, remaining_pixels} << mode16) + 18'd7) >> 3);
        boundary_beats   = 10'((13'd4096 - {1'b0, address[11:0]}) >> 3);
        if (remaining_beats > 15'd16) begin
            planned_beats = 5'd16;
        end else begin
            planned_beats = remaining_beats[4:0];
        end
        if (10'(planned_beats) > boundary_beats) begin
            planned_beats = boundary_beats[4:0];
        end
    end
    assign m_axi_araddr = address;
    assign m_axi_arvalid = state == ADDRESS;
    assign m_axi_rready = state == RECEIVE;
    assign idle = state == IDLE;
    assign pixel_valid = state == STREAM && !cancel;
    assign raw_pixel = 16'(buffer_data[stream_index] >> (8 * byte_index));
    assign pixel = mode16 ? $signed(raw_pixel) : $signed({{8{raw_pixel[7]}}, raw_pixel[7:0]});

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= IDLE;
            address       <= 0;
            row_base      <= 0;
            x             <= 0;
            y             <= 0;
            burst_beats   <= 0;
            m_axi_arlen   <= 0;
            receive_index <= 0;
            stream_index  <= 0;
            byte_index    <= 0;
            failed        <= 0;
            done          <= 0;
            fault         <= 0;
            fault_code    <= ERR_NONE;
        end else begin
            done  <= 0;
            fault <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        address  <= base_addr;
                        row_base <= base_addr;
                        x        <= 0;
                        y        <= 0;
                        failed   <= 0;
                        state    <= PLAN_BURST;
                    end
                end
                // =====================================================================
                // 时序优化 3/3：为满足 100 MHz（10 ns）时序，AXI 读请求先寄存。
                // 切断 width/x -> 剩余拍数 -> 4 KiB/16 拍限制 -> ARLEN 的
                // 组合输出路径；本状态先寄存 burst_beats 和 m_axi_arlen，
                // 下一周期进入 ADDRESS 后再拉高 ARVALID。
                // 代价是每个 burst 增加一个规划周期；等待 ARREADY 时请求保持稳定。
                // 请保留 PLAN_BURST 和 ARLEN 寄存器，避免把计算直接接回 AXI 输出。
                // =====================================================================
                PLAN_BURST: begin
                    // Register the complete request before asserting ARVALID.
                    // Length no longer has a combinational path from width/x
                    // to the external AXI port, and stays stable under stalls.
                    if (cancel) begin
                        state <= IDLE;
                    end else begin
                        burst_beats <= planned_beats;
                        m_axi_arlen <= {3'd0, planned_beats - 5'd1};
                        state       <= ADDRESS;
                    end
                end
                ADDRESS: begin
                    // Even on cancellation ARVALID must not be withdrawn.
                    if (m_axi_arready) begin
                        receive_index <= 0;
                        state         <= RECEIVE;
                    end
                end
                RECEIVE: begin
                    if (m_axi_rvalid) begin
                        if (!failed && !cancel && receive_index < 16) begin
                            buffer_data[receive_index[3:0]] <= m_axi_rdata;
                        end

                        if (!failed && (m_axi_rresp != 0 ||
                                        m_axi_rlast != (receive_index == burst_beats - 1))) begin
                            fault      <= 1;
                            fault_code <= m_axi_rresp != 0 ? ERR_READ : ERR_RLAST;
                            failed     <= 1;
                        end
                        // Drain until RLAST, including after a missing expected
                        // RLAST. A permanently broken bus requires system reset.
                        if (receive_index < 16) begin
                            receive_index <= receive_index + 1;
                        end
                        if (m_axi_rlast) begin
                            if (failed || cancel || m_axi_rresp != 0 ||
                                receive_index != burst_beats - 1) begin
                                state <= IDLE;
                            end else begin
                                stream_index <= 0;
                                byte_index   <= 0;
                                state        <= STREAM;
                            end
                        end
                    end
                end
                STREAM: begin
                    if (cancel) begin
                        state <= IDLE;
                    end else if (pixel_valid && pixel_ready) begin
                        if (x == width - 1) begin
                            // Skip padding in the last beat of this row.
                            x        <= 0;
                            y        <= y + 1;
                            row_base <= row_base + stride;
                            address  <= row_base + stride;
                            if (y == height - 1) begin
                                done  <= 1;
                                state <= IDLE;
                            end else begin
                                state <= PLAN_BURST;
                            end
                        end else begin
                            x <= x + 1;
                            if (byte_index == (mode16 ? 3'd6 : 3'd7)) begin
                                byte_index <= 0;
                                if ({1'b0, stream_index} == burst_beats - 1) begin
                                    address <= address + (32'(burst_beats) << 3);
                                    state   <= PLAN_BURST;
                                end else begin
                                    stream_index <= stream_index + 1;
                                end
                            end else begin
                                byte_index <= byte_index + (mode16 ? 3'd2 : 3'd1);
                            end
                        end
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
