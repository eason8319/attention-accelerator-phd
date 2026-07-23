// Weight-stationary PE：INT8×INT8 → INT32
// a：左→右；psum：上→下；权重加载后驻留。
`timescale 1ns / 1ps

module pe (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               load_w,
  input  logic signed [7:0]  w_in,
  input  logic signed [7:0]  a_in,
  input  logic signed [31:0] psum_in,   // 顶行应驱动 0
  output logic signed [7:0]  a_out,
  output logic signed [31:0] psum_out
);

  logic signed [7:0] w_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      w_reg    <= '0;
      a_out    <= '0;
      psum_out <= '0;
    end else begin
      if (load_w)
        w_reg <= w_in;
      a_out    <= a_in;
      psum_out <= psum_in + (a_in * w_reg);
    end
  end

endmodule
