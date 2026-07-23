// exp_approx：范围规约 + 16 段 PWL 计算 exp(x)
//   输入  x : Q6.10 有符号 16-bit
//   输出  y : UQ0.24 无符号 24-bit
// 3 级流水线；latency = 3 周期（valid_in -> valid_out）。
// 与 scripts/exp_rtl_model.py 逐位一致
`timescale 1ns / 1ps

module exp_approx (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               valid_in,
  input  logic signed [15:0] x,
  output logic               valid_out,
  output logic        [23:0] y
);

  localparam signed [15:0] LOG2E_Q214 = 16'sd23637;

  localparam logic [23:0] INTERCEPT [0:15] = '{
    24'd4194304, 24'd4380002, 24'd4573921, 24'd4776426,
    24'd4987896, 24'd5208729, 24'd5439339, 24'd5680159,
    24'd5931642, 24'd6194258, 24'd6468501, 24'd6754886,
    24'd7053950, 24'd7366255, 24'd7692387, 24'd8032959
  };

  localparam logic [15:0] SLOPE [0:15] = '{
    16'd11606, 16'd12120, 16'd12657, 16'd13217,
    16'd13802, 16'd14413, 16'd15051, 16'd15718,
    16'd16414, 16'd17140, 16'd17899, 16'd18692,
    16'd19519, 16'd20383, 16'd21286, 16'd22228
  };

  // ============================================================ Stage 0: 捕获输入
  logic               v0;
  logic signed [15:0] x0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v0 <= 1'b0;
      x0 <= '0;
    end else begin
      v0 <= valid_in;
      if (valid_in)
        x0 <= x;
    end
  end

  // ============================================================ Stage 1: 乘法 + 拆分
  // y = x * log2(e): Q6.10 * Q2.14 -> Q8.24（有符号 32 位）
  logic signed [31:0] y_prod;
  logic signed [31:0] yi_s;
  logic signed [31:0] yf_s;

  assign y_prod = x0 * LOG2E_Q214;
  assign yi_s   = y_prod >>> 24;                 // floor(y)
  assign yf_s   = y_prod - (yi_s <<< 24);        // UQ0.24，范围 [0, 2^24)

  logic               v1;
  logic signed [7:0]  yi1;
  logic        [23:0] yf1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v1  <= 1'b0;
      yi1 <= '0;
      yf1 <= '0;
    end else begin
      v1 <= v0;
      if (v0) begin
        yi1 <= yi_s[7:0];
        yf1 <= yf_s[23:0];
      end
    end
  end

  // ============================================================ Stage 2: PWL + 2^yi 缩放
  logic [3:0]  seg;
  logic [19:0] seg_off;
  logic [39:0] slope_x_off;  // 16b * 20b
  logic [23:0] pwl;          // UQ2.22
  logic [31:0] wide;         // pwl << 2（对齐 UQ2.24）
  logic [31:0] shifted;
  logic [23:0] y_next;
  logic [5:0]  sh_amt;       // yi < 0 时为 -yi

  assign seg         = yf1[23:20];
  assign seg_off     = yf1[19:0];
  assign slope_x_off = SLOPE[seg] * seg_off;
  assign pwl         = INTERCEPT[seg] + slope_x_off[39:16];  // >> 16
  assign wide        = {8'b0, pwl} << 2;

  always_comb begin
    shifted = 32'd0;
    y_next  = 24'd0;
    sh_amt  = 6'd0;

    if (yi1 >= 8'sd0) begin
      // attention 中少见（x<=0）；仍做处理
      unique case (yi1[2:0])
        3'd0: shifted = wide;
        3'd1: shifted = wide << 1;
        3'd2: shifted = wide << 2;
        3'd3: shifted = wide << 3;
        default: shifted = 32'hFFFF_FFFF;  // 将饱和
      endcase
    end else begin
      sh_amt = 6'(-yi1);
      if (sh_amt >= 6'd31)
        shifted = 32'd0;
      else if (sh_amt == 6'd0)
        shifted = wide;
      else
        shifted = (wide + (32'd1 << (sh_amt - 6'd1))) >> sh_amt;
    end

    if (shifted > 32'h00FF_FFFF)
      y_next = 24'hFF_FFFF;
    else
      y_next = shifted[23:0];
  end

  logic        v2;
  logic [23:0] y2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v2 <= 1'b0;
      y2 <= '0;
    end else begin
      v2 <= v1;
      if (v1)
        y2 <= y_next;
    end
  end

  assign valid_out = v2;
  assign y         = y2;

endmodule
