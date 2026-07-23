// 冒烟测试：验证 Verilator 工具链（make check）。
// 一个 8-bit 计数器数到 16 后结束；不代表任何 P4 模块。
`timescale 1ns / 1ps

module tb_smoke;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic [7:0] cnt;

  initial forever #5 clk = ~clk;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 8'd0;
    else cnt <= cnt + 8'd1;
  end

  initial begin
    repeat (2) @(posedge clk);
    #1 rst_n = 1'b1;  // 错开时钟沿释放 reset，避免与 posedge 竞争
    repeat (16) @(posedge clk);
    #1;  // 等 NBA 更新后再读
    if (cnt != 8'd16) $fatal(1, "smoke FAILED: cnt=%0d expected 16", cnt);
    $display("smoke PASSED: Verilator toolchain OK (cnt=%0d)", cnt);
    $finish;
  end
endmodule
