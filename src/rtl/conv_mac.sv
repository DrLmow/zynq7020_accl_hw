// Five explicitly sized signed multipliers. One kernel row is multiplied
// and then accumulated in two cycles. The whole window is snapshotted on
// acceptance, allowing the window generator to prepare the next window.
module conv_mac (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               cancel,
    input  logic               kernel5,
    input  logic signed [31:0] bias,
    input  logic signed [15:0] weights     [0:24],
    input  logic signed [15:0] window      [ 0:4][0:4],
    input  logic               window_valid,
    output logic               window_ready,
    output logic signed [39:0] accumulator,
    output logic               result_valid,
    input  logic               result_ready
);
    typedef enum logic [1:0] {
        IDLE,
        MULTIPLY,
        ACCUMULATE,
        OUTPUT_RESULT
    } state_t;
    state_t             state;
    logic signed [15:0] saved_window[0:4] [0:4];
    (* use_dsp = "yes" *)logic signed [31:0] products    [0:4];
    logic signed [39:0] row_sum;
    logic        [ 2:0] row;
    logic        [ 2:0] kernel_size;

    assign kernel_size  = kernel5 ? 3'd5 : 3'd3;
    assign window_ready = state == IDLE && !cancel;
    assign result_valid = state == OUTPUT_RESULT && !cancel;
    // Balanced signed tree; explicit sign extension avoids narrow additions.
    always_comb begin
        row_sum = ({{8{products[0][31]}}, products[0]} +
                   {{8{products[1][31]}}, products[1]}) +
                  ({{8{products[2][31]}}, products[2]} +
                   {{8{products[3][31]}}, products[3]}) +
                   {{8{products[4][31]}}, products[4]};
    end
    always_ff @(posedge clk) begin
        if (!rst_n || start || cancel) begin
            state       <= IDLE;
            row         <= 0;
            accumulator <= 0;
            for (int col = 0; col < 5; col++) products[col] <= 0;
            for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) saved_window[r][c] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (window_valid && window_ready) begin
                        for (int r = 0; r < 5; r++)
                        for (int c = 0; c < 5; c++) saved_window[r][c] <= window[r][c];
                        accumulator <= {{8{bias[31]}}, bias};
                        row         <= 0;
                        state       <= MULTIPLY;
                    end
                end
                MULTIPLY: begin
                    for (int col = 0; col < 5; col++) begin
                        if (col < int'(kernel_size)) begin
                            products[col] <= saved_window[row][col] *
                                             weights[int'(row)*int'(kernel_size)+col];
                        end else begin
                            products[col] <= 0;
                        end
                    end
                    state <= ACCUMULATE;
                end
                ACCUMULATE: begin
                    accumulator <= accumulator + row_sum;
                    if (row == kernel_size - 1) begin
                        state <= OUTPUT_RESULT;
                    end else begin
                        row   <= row + 1;
                        state <= MULTIPLY;
                    end
                end
                OUTPUT_RESULT: begin
                    if (result_valid && result_ready) begin
                        state <= IDLE;
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
