// online_softmax 测试平台。
`timescale 1ns / 1ps

module tb_softmax;
  logic               clk = 1'b0;
  logic               rst_n = 1'b0;
  logic               start = 1'b0;
  logic               score_valid = 1'b0;
  logic signed [15:0] score = '0;
  logic               block_last = 1'b0;
  logic               row_last = 1'b0;
  logic               busy;
  logic               accept_score;
  logic               done;
  logic signed [15:0] m_out;
  logic        [31:0] l_out;

  online_softmax #(.MAX_BLOCK(32)) dut (
    .clk,
    .rst_n,
    .start,
    .score_valid,
    .score,
    .block_last,
    .row_last,
    .busy,
    .accept_score,
    .done,
    .m_out,
    .l_out
  );

  initial forever #5 clk = ~clk;

  string vec_dir;
  string meta_line;
  integer fd, code, n_scores, block_size, i;
  integer signed sval;
  logic signed [15:0] score_mem [0:4095];
  logic signed [15:0] dut_m;
  logic        [31:0] dut_l;
  logic               got_done;

  initial begin
    n_scores   = 0;
    block_size = 8;
    got_done   = 1'b0;

    if (!$value$plusargs("vec_dir=%s", vec_dir))
      vec_dir = "../vec_softmax";

    fd = $fopen({vec_dir, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open meta.txt");
    while (!$feof(fd)) begin
      code = $fgets(meta_line, fd);
      if (code != 0) begin
        void'($sscanf(meta_line, "n %d", n_scores));
        void'($sscanf(meta_line, "block_size %d", block_size));
      end
    end
    $fclose(fd);

    fd = $fopen({vec_dir, "/scores.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open scores.txt");
    i = 0;
    while (!$feof(fd) && i < n_scores) begin
      code = $fscanf(fd, "%d\n", sval);
      if (code == 1) begin
        score_mem[i] = sval[15:0];
        i++;
      end
    end
    $fclose(fd);
    if (i != n_scores)
      $fatal(1, "score count %0d != meta n %0d", i, n_scores);
    $display("tb_softmax: n=%0d block_size=%0d", n_scores, block_size);

    repeat (2) @(posedge clk);
    #1 rst_n = 1'b1;
    @(posedge clk);

    @(posedge clk);
    #1 start = 1'b1;
    @(posedge clk);
    #1 start = 1'b0;

    for (i = 0; i < n_scores; i++) begin
      // 等待 DUT 接受 score
      while (!accept_score) begin
        @(posedge clk);
        #1;
      end
      score_valid = 1'b1;
      score       = score_mem[i];
      block_last  = (((i + 1) % block_size) == 0) || (i == n_scores - 1);
      row_last    = (i == n_scores - 1);
      @(posedge clk);
      #1;
      score_valid = 1'b0;
      block_last  = 1'b0;
      row_last    = 1'b0;
    end

    while (!got_done) begin
      @(posedge clk);
      #1;
      if (done) begin
        dut_m    = m_out;
        dut_l    = l_out;
        got_done = 1'b1;
      end
    end

    fd = $fopen({vec_dir, "/dut_m.txt"}, "w");
    $fwrite(fd, "%0d\n", $signed(dut_m));
    $fclose(fd);
    fd = $fopen({vec_dir, "/dut_l.txt"}, "w");
    $fwrite(fd, "%0d\n", dut_l);
    $fclose(fd);

    $display("tb_softmax DONE: m=%0d l=%0d", $signed(dut_m), dut_l);
    $finish;
  end

  initial begin
    #20_000_000;
    $fatal(1, "tb_softmax TIMEOUT");
  end
endmodule
