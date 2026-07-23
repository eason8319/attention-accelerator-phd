// exp_approx 测试平台。
// 从 +vec_dir=/path 读取 Q6.10 向量（x.txt），写出 UQ0.24 dut_out.txt。
`timescale 1ns / 1ps

module tb_exp;
  logic               clk = 1'b0;
  logic               rst_n = 1'b0;
  logic               valid_in = 1'b0;
  logic signed [15:0] x = '0;
  logic               valid_out;
  logic        [23:0] y;

  exp_approx dut (
    .clk,
    .rst_n,
    .valid_in,
    .x,
    .valid_out,
    .y
  );

  initial forever #5 clk = ~clk;

  string vec_dir;
  integer fd_in, fd_out, code;
  integer signed x_val;
  localparam int MAX_N = 200000;

  logic signed [15:0] xin_mem  [0:MAX_N-1];
  logic        [23:0] yout_mem [0:MAX_N-1];
  integer n_samples;
  integer n_out;
  integer i;

  always_ff @(posedge clk) begin
    if (valid_out) begin
      yout_mem[n_out] <= y;
      n_out <= n_out + 1;
    end
  end

  initial begin
    n_samples = 0;
    n_out     = 0;

    if (!$value$plusargs("vec_dir=%s", vec_dir))
      vec_dir = "../vec_exp";

    fd_in = $fopen({vec_dir, "/x.txt"}, "r");
    if (fd_in == 0)
      $fatal(1, "cannot open %s/x.txt", vec_dir);

    while (!$feof(fd_in) && n_samples < MAX_N) begin
      code = $fscanf(fd_in, "%d\n", x_val);
      if (code == 1) begin
        xin_mem[n_samples] = x_val[15:0];
        n_samples++;
      end
    end
    $fclose(fd_in);
    $display("tb_exp: loaded %0d inputs from %s/x.txt", n_samples, vec_dir);

    repeat (2) @(posedge clk);
    #1 rst_n = 1'b1;
    @(posedge clk);

    for (i = 0; i < n_samples; i++) begin
      @(posedge clk);
      #1;
      valid_in = 1'b1;
      x        = xin_mem[i];
    end
    @(posedge clk);
    #1;
    valid_in = 1'b0;
    x        = '0;

    while (n_out < n_samples) begin
      @(posedge clk);
      #1;
    end
    @(posedge clk);

    fd_out = $fopen({vec_dir, "/dut_out.txt"}, "w");
    if (fd_out == 0)
      $fatal(1, "cannot open %s/dut_out.txt", vec_dir);
    for (i = 0; i < n_out; i++)
      $fwrite(fd_out, "%0d\n", yout_mem[i]);
    $fclose(fd_out);

    if (n_out != n_samples)
      $fatal(1, "tb_exp count mismatch: out=%0d in=%0d", n_out, n_samples);

    $display("tb_exp DONE: wrote %0d outputs to %s/dut_out.txt", n_out, vec_dir);
    $finish;
  end

  initial begin
    #50_000_000;
    $fatal(1, "tb_exp TIMEOUT (n_out=%0d n_samples=%0d)", n_out, n_samples);
  end
endmodule
