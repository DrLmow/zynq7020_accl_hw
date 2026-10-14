// Gather a complete burst before advertising AW or W. AW and W proceed
// independently, including the legal case where W is accepted before AW.
module axi_write_dma (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        cancel,
    input  logic [31:0] base_addr,
    input  logic [31:0] stride,
    input  logic [63:0] beat_data,
    input  logic [ 7:0] beat_strobe,
    input  logic        beat_row_last,
    input  logic        beat_frame_last,
    input  logic        beat_valid,
    output logic        beat_ready,
    output logic        idle,
    output logic        done,
    output logic        fault,
    output logic [31:0] m_axi_awaddr,
    output logic [ 7:0] m_axi_awlen,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [ 7:0] m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [ 1:0] m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready
);
    typedef enum logic [1:0] {
        IDLE,
        FILL,
        SEND,
        RESPONSE
    } state_t;
    state_t        state;
    logic   [63:0] buffer_data      [0:15];
    logic   [ 7:0] buffer_strobe    [0:15];
    logic   [ 4:0] fill_count;
    logic   [ 4:0] burst_beats;
    logic   [ 3:0] send_index;
    logic   [31:0] address;
    logic   [31:0] row_base;
    logic   [31:0] boundary_beats;
    logic          aw_sent;
    logic          w_sent;
    logic          row_last;
    logic          frame_last;
    logic          aw_handshake;
    logic          w_handshake;
    logic          last_w_handshake;

    assign boundary_beats   = (32'd4096 - {20'b0, address[11:0]}) >> 3;
    assign beat_ready       = state == FILL && !cancel;
    assign idle             = state == IDLE;
    assign m_axi_awaddr     = address;
    assign m_axi_awlen      = 8'(burst_beats - 1);
    assign m_axi_awvalid    = state == SEND && !aw_sent;
    assign m_axi_wdata      = buffer_data[send_index];
    assign m_axi_wstrb      = buffer_strobe[send_index];
    assign m_axi_wlast      = {1'b0, send_index} == burst_beats - 1;
    assign m_axi_wvalid     = state == SEND && !w_sent;
    assign m_axi_bready     = state == RESPONSE;
    assign aw_handshake     = m_axi_awvalid && m_axi_awready;
    assign w_handshake      = m_axi_wvalid && m_axi_wready;
    assign last_w_handshake = w_handshake && m_axi_wlast;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state       <= IDLE;
            fill_count  <= 0;
            burst_beats <= 0;
            send_index  <= 0;
            address     <= 0;
            row_base    <= 0;
            aw_sent     <= 0;
            w_sent      <= 0;
            row_last    <= 0;
            frame_last  <= 0;
            done        <= 0;
            fault       <= 0;
        end else begin
            done  <= 0;
            fault <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        address    <= base_addr;
                        row_base   <= base_addr;
                        fill_count <= 0;
                        state      <= FILL;
                    end
                end
                FILL: begin
                    if (cancel) begin
                        state <= IDLE;
                    end else if (beat_valid && beat_ready) begin
                        buffer_data[fill_count[3:0]]   <= beat_data;
                        buffer_strobe[fill_count[3:0]] <= beat_strobe;
                        fill_count                     <= fill_count + 1;
                        if (fill_count == 15 || beat_row_last ||
                            32'(fill_count)+1 == boundary_beats) begin
                            burst_beats <= fill_count + 1;
                            row_last    <= beat_row_last;
                            frame_last  <= beat_frame_last;
                            send_index  <= 0;
                            aw_sent     <= 0;
                            w_sent      <= 0;
                            state       <= SEND;
                        end
                    end
                end
                SEND: begin
                    // A committed request is fully drained even on cancel.
                    if (aw_handshake) begin
                        aw_sent <= 1;
                    end
                    if (w_handshake) begin
                        if (m_axi_wlast) begin
                            w_sent <= 1;
                        end else begin
                            send_index <= send_index + 1;
                        end
                    end
                    if ((aw_sent || aw_handshake) && (w_sent || last_w_handshake)) begin
                        state <= RESPONSE;
                    end
                end
                RESPONSE: begin
                    if (m_axi_bvalid) begin
                        if (m_axi_bresp != 0) begin
                            fault <= 1;
                            state <= IDLE;
                        end else if (cancel) begin
                            state <= IDLE;
                        end else if (frame_last) begin
                            done  <= 1;
                            state <= IDLE;
                        end else begin
                            if (row_last) begin
                                row_base <= row_base + stride;
                                address  <= row_base + stride;
                            end else begin
                                address <= address + (32'(burst_beats) << 3);
                            end
                            fill_count <= 0;
                            state      <= FILL;
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
