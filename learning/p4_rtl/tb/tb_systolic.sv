// 4x4 WS systolic_array 测试平台。
`timescale 1ns / 1ps

module tb_systolic;
  localparam int N     = 4;
  localparam int MAX_M = 64;

  logic               clk = 1'b0;
  logic               rst_n = 1'b0;
  logic               load_w = 1'b0;
  logic               valid_a = 1'b0;
  logic signed [7:0]  w_mat [0:N-1][0:N-1];
  logic signed [7:0]  a_row [0:N-1];
  logic               valid_c;
  logic signed [31:0] c_row [0:N-1];

  systolic_array #(.N(N)) dut (
    .clk,
    .rst_n,
    .load_w,
    .w_mat,
    .valid_a,
    .a_row,
    .valid_c,
    .c_row
  );

  initial forever #5 clk = ~clk;

  string vec_dir;
  string meta_line;
  integer fd, code, m_rows, i, r, c, n_out, feed_i, idle;
  integer signed tmp;
  logic signed [7:0]  a_mem [0:MAX_M-1][0:N-1];
  logic signed [31:0] c_mem [0:MAX_M-1][0:N-1];

  task automatic capture_if_valid;
    begin
      if (valid_c) begin
        if (n_out >= m_rows)
          $fatal(1, "extra valid_c at n_out=%0d", n_out);
        for (c = 0; c < N; c++)
          c_mem[n_out][c] = c_row[c];
        n_out++;
      end
    end
  endtask

  initial begin
    m_rows = 0;
    n_out  = 0;
    feed_i = 0;
    idle   = 0;

    if (!$value$plusargs("vec_dir=%s", vec_dir))
      vec_dir = "../vec_sa";

    fd = $fopen({vec_dir, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "meta.txt");
    while (!$feof(fd)) begin
      code = $fgets(meta_line, fd);
      if (code != 0)
        void'($sscanf(meta_line, "m %d", m_rows));
    end
    $fclose(fd);
    if (m_rows <= 0 || m_rows > MAX_M)
      $fatal(1, "bad m_rows=%0d", m_rows);

    fd = $fopen({vec_dir, "/w.txt"}, "r");
    if (fd == 0) $fatal(1, "w.txt");
    for (r = 0; r < N; r++)
      for (c = 0; c < N; c++) begin
        code = $fscanf(fd, "%d\n", tmp);
        w_mat[r][c] = tmp[7:0];
      end
    $fclose(fd);

    fd = $fopen({vec_dir, "/a.txt"}, "r");
    if (fd == 0) $fatal(1, "a.txt");
    for (i = 0; i < m_rows; i++)
      for (c = 0; c < N; c++) begin
        code = $fscanf(fd, "%d\n", tmp);
        a_mem[i][c] = tmp[7:0];
      end
    $fclose(fd);

    $display("tb_systolic: M=%0d N=%0d", m_rows, N);

    for (c = 0; c < N; c++)
      a_row[c] = '0;

    repeat (2) @(posedge clk);
    #1 rst_n = 1'b1;
    @(posedge clk);

    @(posedge clk);
    #1 load_w = 1'b1;
    @(posedge clk);
    #1 load_w = 1'b0;
    @(posedge clk);
    #1;

    // 每周期驱动一行 A；valid_c 时随时采样 C（可与喂数重叠）。
    while (n_out < m_rows) begin
      @(posedge clk);
      #1;
      capture_if_valid();

      if (feed_i < m_rows) begin
        valid_a = 1'b1;
        for (c = 0; c < N; c++)
          a_row[c] = a_mem[feed_i][c];
        feed_i++;
      end else begin
        valid_a = 1'b0;
        for (c = 0; c < N; c++)
          a_row[c] = '0;
        idle++;
        if (idle > 64)
          $fatal(1, "no more valid_c: n_out=%0d feed_i=%0d", n_out, feed_i);
      end
    end

    fd = $fopen({vec_dir, "/dut_c.txt"}, "w");
    for (i = 0; i < m_rows; i++)
      for (c = 0; c < N; c++)
        $fwrite(fd, "%0d\n", $signed(c_mem[i][c]));
    $fclose(fd);

    $display("tb_systolic DONE: wrote %0d rows", n_out);
    $finish;
  end

  initial begin
    #5_000_000;
    $fatal(1, "tb_systolic TIMEOUT n_out=%0d", n_out);
  end
endmodule
