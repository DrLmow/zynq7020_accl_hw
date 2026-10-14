// Four circular line banks plus a 5x5 horizontal window. Read and overwrite
// happen on different cycles, avoiding read-during-write collision semantics.
module conv_window #(
    parameter int MAX_WIDTH = 2048
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               cancel,
    input  logic               kernel5,
    input  logic        [15:0] width,
    input  logic signed [15:0] pixel,
    input  logic               pixel_valid,
    output logic               pixel_ready,
    output logic signed [15:0] window      [0:4][0:4],
    output logic               window_valid,
    input  logic               window_ready
);
    localparam int COL_BITS = $clog2(MAX_WIDTH) > 0 ? $clog2(MAX_WIDTH) : 1;
    typedef enum logic [1:0] {
        WAIT_PIXEL,
        READ_LINES,
        UPDATE_WINDOW,
        HOLD_WINDOW
    } state_t;
    state_t             state;
    // Do not reset these arrays: resetting every location prevents BRAM inference.
    (* ram_style = "block" *)logic signed [15:0] lines         [0:3] [0:MAX_WIDTH-1];
    logic signed [15:0] old_pixel     [0:3];
    logic signed [15:0] current_pixel;
    logic        [15:0] x;
    logic        [15:0] y;
    logic        [ 2:0] kernel_size;
    logic        [ 1:0] write_bank;
    logic        [ 1:0] read_bank     [0:3];

    assign kernel_size = kernel5 ? 3'd5 : 3'd3;
    // Modulo 2/4 is a bit selection, not a hardware divider.
    assign write_bank  = kernel5 ? y[1:0] : {1'b0, y[0]};
    for (genvar row = 0; row < 4; row++) begin : bank_map
        wire [1:0] row_number = y[1:0] + 2'(row);
        assign read_bank[row] = kernel5 ? row_number : {1'b0, row_number[0]};
    end
    assign pixel_ready  = state == WAIT_PIXEL && !cancel;
    assign window_valid = state == HOLD_WINDOW && !cancel;

    // Each physical bank has one registered read and one independent write.
    // A task starts by invalidating history via y, rather than clearing memory.
    for (genvar bank = 0; bank < 4; bank++) begin : line_memory
        always_ff @(posedge clk) begin
            if (state == READ_LINES) begin
                old_pixel[bank] <= lines[bank][x[COL_BITS-1:0]];
            end
            if (state == UPDATE_WINDOW && !cancel && write_bank == 2'(bank)) begin
                lines[bank][x[COL_BITS-1:0]] <= current_pixel;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n || start || cancel) begin
            state         <= WAIT_PIXEL;
            x             <= 0;
            y             <= 0;
            current_pixel <= 0;
            for (int row = 0; row < 5; row++)
            for (int col = 0; col < 5; col++) window[row][col] <= 0;
        end else begin
            case (state)
                WAIT_PIXEL: begin
                    if (pixel_valid && pixel_ready) begin
                        current_pixel <= pixel;
                        state         <= READ_LINES;
                    end
                end
                READ_LINES: begin
                    state <= UPDATE_WINDOW;
                end
                UPDATE_WINDOW: begin
                    for (int row = 0; row < 5; row++) begin
                        for (int col = 0; col < 4; col++) begin
                            if (col < int'(kernel_size) - 1) begin
                                window[row][col] <= x == 0 ? 16'sd0 : window[row][col+1];
                            end
                        end
                        if (row == int'(kernel_size) - 1) begin
                            window[row][kernel_size-1] <= current_pixel;
                        end else if (row < int'(kernel_size) - 1) begin
                            window[row][kernel_size-1] <= old_pixel[read_bank[row]];
                        end
                    end
                    // Stale memory is harmless until enough complete rows exist.
                    if (y >= 16'(kernel_size - 1) && x >= 16'(kernel_size - 1)) begin
                        state <= HOLD_WINDOW;
                    end else begin
                        state <= WAIT_PIXEL;
                    end
                    if (x == width - 1) begin
                        x <= 0;
                        y <= y + 1;
                    end else begin
                        x <= x + 1;
                    end
                end
                HOLD_WINDOW: begin
                    if (window_valid && window_ready) begin
                        state <= WAIT_PIXEL;
                    end
                end
                default: begin
                    state <= WAIT_PIXEL;
                end
            endcase
        end
    end
endmodule
