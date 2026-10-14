// Packs little-endian output pixels. Each row is flushed independently;
// partial final beats carry strobes instead of overwriting padding bytes.
module pixel_pack (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               cancel,
    input  logic               mode16,
    input  logic        [15:0] width,
    input  logic        [15:0] height,
    input  logic signed [15:0] pixel,
    input  logic               pixel_valid,
    output logic               pixel_ready,
    output logic        [63:0] beat_data,
    output logic        [ 7:0] beat_strobe,
    output logic               beat_row_last,
    output logic               beat_frame_last,
    output logic               beat_valid,
    input  logic               beat_ready
);
    logic [ 2:0] byte_count;
    logic [15:0] x;
    logic [15:0] y;
    logic [ 3:0] next_byte_count;
    assign pixel_ready     = !beat_valid && !cancel;
    assign next_byte_count = {1'b0, byte_count} + (mode16 ? 4'd2 : 4'd1);
    always_ff @(posedge clk) begin
        if (!rst_n || start || cancel) begin
            byte_count      <= 0;
            x               <= 0;
            y               <= 0;
            beat_data       <= 0;
            beat_strobe     <= 0;
            beat_row_last   <= 0;
            beat_frame_last <= 0;
            beat_valid      <= 0;
        end else begin
            if (beat_valid && beat_ready) begin
                beat_valid <= 0;
                beat_data  <= 0;
                byte_count <= 0;
            end
            if (pixel_valid && pixel_ready) begin
                if (mode16) begin
                    beat_data[8*byte_count+:16] <= pixel;
                end else begin
                    beat_data[8*byte_count+:8] <= pixel[7:0];
                end
                byte_count <= next_byte_count[2:0];
                if (next_byte_count == 8 || x == width - 1) begin
                    beat_valid      <= 1;
                    beat_strobe     <= 8'((9'd1 << next_byte_count) - 9'd1);
                    beat_row_last   <= x == width - 1;
                    beat_frame_last <= x == width - 1 && y == height - 1;
                end
                if (x == width - 1) begin
                    x <= 0;
                    y <= y + 1;
                end else begin
                    x <= x + 1;
                end
            end
        end
    end
endmodule
