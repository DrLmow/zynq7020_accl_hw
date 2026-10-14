// Verilog-2001 module-reference entry point for Vivado Block Design.
// The datapath remains SystemVerilog; this wrapper only forwards signals.
// AXI interface attributes let Vivado group ports and associate clock/reset.
// All interfaces use clk. Supply a reset synchronized to this clock domain.
module conv_accel_bd_wrapper #(
    parameter integer MAX_WIDTH  = 2048,
    parameter integer MAX_HEIGHT = 8192
) (
    // Shared clock for both AXI interfaces and the convolution engine.
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, ASSOCIATED_BUSIF S_AXI_CTRL:M_AXI_DATA, ASSOCIATED_RESET rst_n" *)
    input  wire        clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME rst_n, POLARITY ACTIVE_LOW" *)
    input wire rst_n,
    // Level stays high until software clears enabled DONE/ERROR status.
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME irq, SENSITIVITY LEVEL_HIGH" *)
    output wire irq,

    // 32-bit AXI4-Lite slave: register access by the processor.
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_CTRL, PROTOCOL AXI4LITE, ADDR_WIDTH 32, DATA_WIDTH 32, READ_WRITE_MODE READ_WRITE, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWADDR" *)
    input  wire [31:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWPROT" *)
    input  wire [ 2:0] s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWVALID" *)
    input  wire        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWREADY" *)
    output wire        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WDATA" *)
    input  wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WSTRB" *)
    input  wire [ 3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WVALID" *)
    input  wire        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WREADY" *)
    output wire        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BRESP" *)
    output wire [ 1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BVALID" *)
    output wire        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BREADY" *)
    input  wire        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARADDR" *)
    input  wire [31:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARPROT" *)
    input  wire [ 2:0] s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARVALID" *)
    input  wire        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARREADY" *)
    output wire        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RDATA" *)
    output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RRESP" *)
    output wire [ 1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RVALID" *)
    output wire        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RREADY" *)
    input wire s_axi_rready,

    // 64-bit AXI4 master: INCR bursts, at most 16 beats, ID fixed to zero.
    // Read and write each permit one outstanding transaction.
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_DATA, PROTOCOL AXI4, ADDR_WIDTH 32, DATA_WIDTH 64, ID_WIDTH 1, READ_WRITE_MODE READ_WRITE, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARID" *)
    output wire [ 0:0] m_axi_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARADDR" *)
    output wire [31:0] m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARLEN" *)
    output wire [ 7:0] m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARSIZE" *)
    output wire [ 2:0] m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARBURST" *)
    output wire [ 1:0] m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARLOCK" *)
    output wire        m_axi_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARCACHE" *)
    output wire [ 3:0] m_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARPROT" *)
    output wire [ 2:0] m_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARQOS" *)
    output wire [ 3:0] m_axi_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARVALID" *)
    output wire        m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA ARREADY" *)
    input  wire        m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RID" *)
    input  wire [ 0:0] m_axi_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RDATA" *)
    input  wire [63:0] m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RRESP" *)
    input  wire [ 1:0] m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RLAST" *)
    input  wire        m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RVALID" *)
    input  wire        m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA RREADY" *)
    output wire        m_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWID" *)
    output wire [ 0:0] m_axi_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWADDR" *)
    output wire [31:0] m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWLEN" *)
    output wire [ 7:0] m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWSIZE" *)
    output wire [ 2:0] m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWBURST" *)
    output wire [ 1:0] m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWLOCK" *)
    output wire        m_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWCACHE" *)
    output wire [ 3:0] m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWPROT" *)
    output wire [ 2:0] m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWQOS" *)
    output wire [ 3:0] m_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWVALID" *)
    output wire        m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA AWREADY" *)
    input  wire        m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA WDATA" *)
    output wire [63:0] m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA WSTRB" *)
    output wire [ 7:0] m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA WLAST" *)
    output wire        m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA WVALID" *)
    output wire        m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA WREADY" *)
    input  wire        m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA BID" *)
    input  wire [ 0:0] m_axi_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA BRESP" *)
    input  wire [ 1:0] m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA BVALID" *)
    input  wire        m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DATA BREADY" *)
    output wire m_axi_bready
);

    // Explicit named connections keep this file valid as Verilog-2001.
    // MAX_WIDTH controls the line-buffer capacity inside the SV core.
    conv_accel_top #(
        .MAX_WIDTH (MAX_WIDTH),
        .MAX_HEIGHT(MAX_HEIGHT)
    ) u_core (
        .clk          (clk),
        .rst_n        (rst_n),
        .irq          (irq),
        .s_axi_awaddr (s_axi_awaddr),
        .s_axi_awprot (s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata  (s_axi_wdata),
        .s_axi_wstrb  (s_axi_wstrb),
        .s_axi_wvalid (s_axi_wvalid),
        .s_axi_wready (s_axi_wready),
        .s_axi_bresp  (s_axi_bresp),
        .s_axi_bvalid (s_axi_bvalid),
        .s_axi_bready (s_axi_bready),
        .s_axi_araddr (s_axi_araddr),
        .s_axi_arprot (s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata  (s_axi_rdata),
        .s_axi_rresp  (s_axi_rresp),
        .s_axi_rvalid (s_axi_rvalid),
        .s_axi_rready (s_axi_rready),
        .m_axi_arid   (m_axi_arid),
        .m_axi_araddr (m_axi_araddr),
        .m_axi_arlen  (m_axi_arlen),
        .m_axi_arsize (m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock (m_axi_arlock),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arprot (m_axi_arprot),
        .m_axi_arqos  (m_axi_arqos),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid    (m_axi_rid),
        .m_axi_rdata  (m_axi_rdata),
        .m_axi_rresp  (m_axi_rresp),
        .m_axi_rlast  (m_axi_rlast),
        .m_axi_rvalid (m_axi_rvalid),
        .m_axi_rready (m_axi_rready),
        .m_axi_awid   (m_axi_awid),
        .m_axi_awaddr (m_axi_awaddr),
        .m_axi_awlen  (m_axi_awlen),
        .m_axi_awsize (m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock (m_axi_awlock),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awprot (m_axi_awprot),
        .m_axi_awqos  (m_axi_awqos),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata  (m_axi_wdata),
        .m_axi_wstrb  (m_axi_wstrb),
        .m_axi_wlast  (m_axi_wlast),
        .m_axi_wvalid (m_axi_wvalid),
        .m_axi_wready (m_axi_wready),
        .m_axi_bid    (m_axi_bid),
        .m_axi_bresp  (m_axi_bresp),
        .m_axi_bvalid (m_axi_bvalid),
        .m_axi_bready (m_axi_bready)
    );

endmodule
