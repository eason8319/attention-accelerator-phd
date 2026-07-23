// 4×4 weight-stationary INT8 systolic array，输入 skew + 输出 deskew。
//   C[M,4] = A[M,4] @ W[4,4]  （INT8 输入，INT32 累加）
// 与 scripts/systolic_rtl_model.py 逐周期一致
`timescale 1ns / 1ps

module systolic_array #(
  parameter int N = 4
) (
  input  logic               clk,
  input  logic               rst_n,

  input  logic               load_w,
  input  logic signed [7:0]  w_mat [0:N-1][0:N-1],

  input  logic               valid_a,
  input  logic signed [7:0]  a_row [0:N-1],

  output logic               valid_c,
  output logic signed [31:0] c_row [0:N-1]
);

  // Python 概念 latency 为 2*(N-1)。可综合 RTL 中 always_ff
  // 在 PE NBA 前采样 bottom，故输出晚一周期。
  localparam int LAT = 2 * (N - 1) + 1;

  logic signed [7:0]  a_h [0:N-1][0:N];
  logic signed [31:0] p_v [0:N][0:N-1];

  genvar gr, gc;
  generate
    for (gc = 0; gc < N; gc++) begin : gn
      assign p_v[0][gc] = 32'sd0;
    end
    for (gr = 0; gr < N; gr++) begin : grow
      for (gc = 0; gc < N; gc++) begin : gcol
        pe u_pe (
          .clk,
          .rst_n,
          .load_w (load_w),
          .w_in   (w_mat[gr][gc]),
          .a_in   (a_h[gr][gc]),
          .psum_in(p_v[gr][gc]),
          .a_out  (a_h[gr][gc+1]),
          .psum_out(p_v[gr+1][gc])
        );
      end
    end
  endgenerate

  // -------- 输入 skew：west[0]=a_logic[0] 同周期；west[r] 延迟 r 周期 --------
  logic signed [7:0] a_logic [0:N-1];
  logic signed [7:0] skew_pipe [1:N-1][0:N-2];

  integer ri, si, ci;

  always_comb begin
    for (ri = 0; ri < N; ri++)
      a_logic[ri] = valid_a ? a_row[ri] : 8'sd0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (ri = 1; ri < N; ri++)
        for (si = 0; si < N - 1; si++)
          skew_pipe[ri][si] <= '0;
    end else begin
      for (ri = 1; ri < N; ri++) begin
        skew_pipe[ri][0] <= a_logic[ri];
        for (si = 1; si < ri; si++)
          skew_pipe[ri][si] <= skew_pipe[ri][si-1];
      end
    end
  end

  assign a_h[0][0] = a_logic[0];
  generate
    for (gr = 1; gr < N; gr++) begin : gwest
      assign a_h[gr][0] = skew_pipe[gr][gr-1];
    end
  endgenerate

  // -------- Deskew + valid 流水线（深度 LAT）--------
  // hist[c][0] = 本周期 bottom（已寄存）
  // 输出延迟 d=N-1-c：data = (d==0) ? bottom : hist[c][d-1]
  logic signed [31:0] hist [0:N-1][0:N-2];
  logic [LAT-1:0]     vpipe;
  logic signed [31:0] bottom [0:N-1];
  logic signed [31:0] c_comb [0:N-1];

  generate
    for (gc = 0; gc < N; gc++) begin : gbot
      assign bottom[gc] = p_v[N][gc];
    end
  endgenerate

  always_comb begin
    for (ci = 0; ci < N; ci++) begin
      if (N - 1 - ci == 0)
        c_comb[ci] = bottom[ci];
      else
        c_comb[ci] = hist[ci][N - 2 - ci];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (ci = 0; ci < N; ci++)
        for (si = 0; si < N - 1; si++)
          hist[ci][si] <= '0;
      vpipe   <= '0;
      valid_c <= 1'b0;
      for (ci = 0; ci < N; ci++)
        c_row[ci] <= '0;
    end else begin
      vpipe <= {vpipe[LAT-2:0], valid_a};

      for (ci = 0; ci < N; ci++) begin
        hist[ci][0] <= bottom[ci];
        for (si = 1; si < N - 1; si++)
          hist[ci][si] <= hist[ci][si-1];
      end

      // 当发射 valid 已老化 LAT 周期时输出（移位后 vpipe MSB）
      valid_c <= vpipe[LAT-1];
      for (ci = 0; ci < N; ci++)
        c_row[ci] <= c_comb[ci];
    end
  end

endmodule
