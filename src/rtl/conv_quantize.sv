// =====================================================================
// 时序优化 2/3：为满足 100 MHz（10 ns）时序，量化拆成五级寄存流水线。
// 依次完成幅值处理、舍入加法、可变移位、符号恢复、饱和截断，
// 避免加法器、桶形移位器、取负和比较逻辑形成一条过长的组合路径。
// 代价是增加流水线延迟；量化的数值规则保持不变。
// 请保留各级寄存器；反压时数据、配置元信息和 valid 必须同步冻结，
// 防止不同像素的数据与配置错位。
// =====================================================================
// Five-stage quantizer: magnitude -> rounding -> shift -> sign -> saturation.
// All stages advance together when the output slot is free. Backpressure
// freezes data, metadata and valid bits together, preserving pixel order.
module conv_quantize (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               cancel,
    input  logic               mode16,
    input  logic               round_enable,
    input  logic        [ 4:0] shift,
    input  logic signed [39:0] accumulator,
    input  logic               accumulator_valid,
    output logic               accumulator_ready,
    output logic signed [15:0] pixel,
    output logic               pixel_valid,
    input  logic               pixel_ready
);
    logic               advance;
    logic signed [40:0] extended_value;
    logic               use_rounding;

    logic               stage1_valid;
    logic signed [40:0] stage1_value;
    logic        [ 4:0] stage1_shift;
    logic               stage1_round;
    logic               stage1_negative;
    logic               stage1_mode16;

    logic               stage2_valid;
    logic signed [40:0] stage2_value;
    logic        [ 4:0] stage2_shift;
    logic               stage2_negative;
    logic               stage2_mode16;

    logic               stage3_valid;
    logic signed [40:0] stage3_value;
    logic               stage3_negative;
    logic               stage3_mode16;

    logic               stage4_valid;
    logic signed [40:0] stage4_value;
    logic               stage4_mode16;

    logic               stage5_valid;

    assign extended_value    = {accumulator[39], accumulator};
    assign use_rounding      = round_enable && shift != 0;
    assign advance           = !stage5_valid || pixel_ready;
    assign accumulator_ready = advance && !cancel && !start;
    assign pixel_valid       = stage5_valid && !cancel && !start;

    always_ff @(posedge clk) begin
        if (!rst_n || start || cancel) begin
            stage1_valid    <= 0;
            stage1_value    <= 0;
            stage1_shift    <= 0;
            stage1_round    <= 0;
            stage1_negative <= 0;
            stage1_mode16   <= 0;
            stage2_valid    <= 0;
            stage2_value    <= 0;
            stage2_shift    <= 0;
            stage2_negative <= 0;
            stage2_mode16   <= 0;
            stage3_valid    <= 0;
            stage3_value    <= 0;
            stage3_negative <= 0;
            stage3_mode16   <= 0;
            stage4_valid    <= 0;
            stage4_value    <= 0;
            stage4_mode16   <= 0;
            stage5_valid    <= 0;
            pixel           <= 0;
        end else if (advance) begin
            // Stage 1: rounded negative values use magnitude arithmetic.
            // Nonrounded values stay signed for floor arithmetic right shift.
            // The extra bit safely represents abs(-2^39).
            stage1_valid    <= accumulator_valid;
            stage1_value    <= use_rounding && accumulator[39] ? -extended_value : extended_value;
            stage1_shift    <= shift;
            stage1_round    <= use_rounding;
            stage1_negative <= use_rounding && accumulator[39];
            stage1_mode16   <= mode16;

            // Stage 2: add half of the divisor for ties away from zero.
            // A zero shift bypasses rounding, avoiding shift - 1 underflow.
            stage2_valid    <= stage1_valid;
            if (stage1_round) begin
                stage2_value <= stage1_value + (41'sd1 << (stage1_shift - 5'd1));
            end else begin
                stage2_value <= stage1_value;
            end
            stage2_shift    <= stage1_shift;
            stage2_negative <= stage1_negative;
            stage2_mode16   <= stage1_mode16;

            // Stage 3: isolate the variable barrel shifter from the adders.
            stage3_valid    <= stage2_valid;
            stage3_value    <= stage2_value >>> stage2_shift;
            stage3_negative <= stage2_negative;
            stage3_mode16   <= stage2_mode16;

            // Stage 4: restore the sign only for the rounded-magnitude path.
            stage4_valid    <= stage3_valid;
            stage4_value    <= stage3_negative ? -stage3_value : stage3_value;
            stage4_mode16   <= stage3_mode16;

            // Stage 5: saturate to the selected signed output width.
            stage5_valid    <= stage4_valid;
            if (stage4_mode16) begin
                if (stage4_value > 41'sd32767) begin
                    pixel <= 16'sd32767;
                end else if (stage4_value < -41'sd32768) begin
                    pixel <= -16'sd32768;
                end else begin
                    pixel <= stage4_value[15:0];
                end
            end else begin
                if (stage4_value > 41'sd127) begin
                    pixel <= 16'sd127;
                end else if (stage4_value < -41'sd128) begin
                    pixel <= -16'sd128;
                end else begin
                    pixel <= {{8{stage4_value[7]}}, stage4_value[7:0]};
                end
            end
        end
    end

`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (!rst_n || start || cancel)
        pixel_valid && !pixel_ready |=> pixel_valid && $stable(
        pixel
    ));
`endif
endmodule
