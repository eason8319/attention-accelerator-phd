// online_softmax：单行 online softmax 归约（仅 m、l；无 PV/O）。
// 例化 exp_approx。以 block_last / row_last 流式输入 score。
// 格式：score/m = Q6.10；l = UQ8.24。
// 与 scripts/softmax_rtl_model.py 逐位一致
`timescale 1ns / 1ps

module online_softmax #(
  parameter int MAX_BLOCK = 32
) (
  input  logic               clk,
  input  logic               rst_n,

  input  logic               start,
  input  logic               score_valid,
  input  logic signed [15:0] score,
  input  logic               block_last,
  input  logic               row_last,

  output logic               busy,
  output logic               accept_score, // GATHER 态为高：TB 可提交 score
  output logic               done,
  output logic signed [15:0] m_out,
  output logic        [31:0] l_out
);

  localparam int IDX_W = $clog2(MAX_BLOCK);

  typedef enum logic [3:0] {
    ST_IDLE         = 4'd0,
    ST_GATHER       = 4'd1,
    ST_PREP         = 4'd2,
    ST_ISSUE_ALPHA  = 4'd3,
    ST_WAIT_ALPHA   = 4'd4,
    ST_ISSUE_SCORES = 4'd5,
    ST_WAIT_SCORES  = 4'd6,
    ST_UPDATE       = 4'd7,
    ST_DONE         = 4'd8
  } state_t;

  state_t state, state_n;

  logic               m_valid, m_valid_n;
  logic signed [15:0] m_reg, m_n;
  logic        [31:0] l_reg, l_n;

  logic signed [15:0] score_mem [0:MAX_BLOCK-1];
  logic [IDX_W:0]     blk_len, blk_len_n;
  logic [IDX_W:0]     wr_ptr;
  logic               row_end_f, row_end_f_n;
  logic signed [15:0] m_blk, m_blk_n;
  logic signed [15:0] m_new_r, m_new_n;

  logic [IDX_W:0] issue_i, issue_i_n;
  logic [IDX_W:0] got_i, got_i_n;
  logic [31:0]    sum_e, sum_e_n;
  logic [23:0]    alpha_r, alpha_n;

  logic               exp_vin;
  logic signed [15:0] exp_x;
  logic               exp_vout;
  logic        [23:0] exp_y;

  exp_approx u_exp (
    .clk,
    .rst_n,
    .valid_in (exp_vin),
    .x        (exp_x),
    .valid_out(exp_vout),
    .y        (exp_y)
  );

  function automatic logic signed [15:0] q610_sub(
      input logic signed [15:0] a,
      input logic signed [15:0] b
  );
    logic signed [16:0] d;
    begin
      d = {a[15], a} - {b[15], b};
      if (d > 17'sd32767)       q610_sub = 16'sh7FFF;
      else if (d < -17'sd32768) q610_sub = 16'sh8000;
      else                      q610_sub = d[15:0];
    end
  endfunction

  function automatic logic [31:0] mul_alpha_l(
      input logic [23:0] alpha,
      input logic [31:0] l_val
  );
    logic [55:0] prod;
    begin
      prod = alpha * l_val;
      mul_alpha_l = prod[55:24];
    end
  endfunction

  function automatic logic [31:0] sat_add32(
      input logic [31:0] a,
      input logic [31:0] b
  );
    logic [32:0] s;
    begin
      s = {1'b0, a} + {1'b0, b};
      sat_add32 = s[32] ? 32'hFFFF_FFFF : s[31:0];
    end
  endfunction

  integer ki;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= ST_IDLE;
      m_valid   <= 1'b0;
      m_reg     <= '0;
      l_reg     <= '0;
      blk_len   <= '0;
      wr_ptr    <= '0;
      row_end_f <= 1'b0;
      m_blk     <= 16'sh8000;
      m_new_r   <= '0;
      issue_i   <= '0;
      got_i     <= '0;
      sum_e     <= '0;
      alpha_r   <= '0;
      done      <= 1'b0;
      m_out     <= '0;
      l_out     <= '0;
      for (ki = 0; ki < MAX_BLOCK; ki++)
        score_mem[ki] <= '0;
    end else begin
      state     <= state_n;
      m_valid   <= m_valid_n;
      m_reg     <= m_n;
      l_reg     <= l_n;
      blk_len   <= blk_len_n;
      row_end_f <= row_end_f_n;
      m_blk     <= m_blk_n;
      m_new_r   <= m_new_n;
      issue_i   <= issue_i_n;
      got_i     <= got_i_n;
      sum_e     <= sum_e_n;
      alpha_r   <= alpha_n;

      done <= 1'b0;
      if (state == ST_GATHER && score_valid) begin
        score_mem[wr_ptr[IDX_W-1:0]] <= score;
        wr_ptr <= wr_ptr + 1'b1;
      end
      if ((state == ST_IDLE && start) || (state == ST_UPDATE))
        wr_ptr <= '0;

      if (state == ST_UPDATE && state_n == ST_DONE) begin
        done  <= 1'b1;
        m_out <= m_n;
        l_out <= l_n;
      end
    end
  end

  assign busy         = (state != ST_IDLE) && (state != ST_DONE);
  assign accept_score = (state == ST_GATHER);

  always_comb begin
    state_n     = state;
    m_valid_n   = m_valid;
    m_n         = m_reg;
    l_n         = l_reg;
    blk_len_n   = blk_len;
    row_end_f_n = row_end_f;
    m_blk_n     = m_blk;
    m_new_n     = m_new_r;
    issue_i_n   = issue_i;
    got_i_n     = got_i;
    sum_e_n     = sum_e;
    alpha_n     = alpha_r;

    exp_vin = 1'b0;
    exp_x   = '0;

    unique case (state)
      ST_IDLE: begin
        if (start) begin
          state_n     = ST_GATHER;
          m_valid_n   = 1'b0;
          m_n         = '0;
          l_n         = '0;
          blk_len_n   = '0;
          row_end_f_n = 1'b0;
          m_blk_n     = 16'sh8000;
          issue_i_n   = '0;
          got_i_n     = '0;
          sum_e_n     = '0;
        end
      end

      ST_GATHER: begin
        if (score_valid) begin
          if (blk_len == '0 || $signed(score) > $signed(m_blk))
            m_blk_n = score;
          blk_len_n = blk_len + 1'b1;
          if (row_last)
            row_end_f_n = 1'b1;
          if (block_last || row_last)
            state_n = ST_PREP;
        end
      end

      ST_PREP: begin
        m_new_n   = m_valid ? (($signed(m_reg) > $signed(m_blk)) ? m_reg : m_blk)
                            : m_blk;
        sum_e_n   = '0;
        issue_i_n = '0;
        got_i_n   = '0;
        if (m_valid)
          state_n = ST_ISSUE_ALPHA;
        else
          state_n = ST_ISSUE_SCORES;
      end

      ST_ISSUE_ALPHA: begin
        exp_vin = 1'b1;
        exp_x   = q610_sub(m_reg, m_new_r);
        state_n = ST_WAIT_ALPHA;
      end

      ST_WAIT_ALPHA: begin
        if (exp_vout) begin
          alpha_n = exp_y;
          l_n     = mul_alpha_l(exp_y, l_reg);
          state_n = ST_ISSUE_SCORES;
        end
      end

      ST_ISSUE_SCORES: begin
        // 流水线：持续发射，同时收集更早的结果
        if (issue_i < blk_len) begin
          exp_vin   = 1'b1;
          exp_x     = q610_sub(score_mem[issue_i[IDX_W-1:0]], m_new_r);
          issue_i_n = issue_i + 1'b1;
        end
        if (exp_vout) begin
          sum_e_n = sat_add32(sum_e, {8'b0, exp_y});
          got_i_n = got_i + 1'b1;
        end
        if (issue_i + ((issue_i < blk_len) ? 1 : 0) >= blk_len)
          state_n = ST_WAIT_SCORES;
      end

      ST_WAIT_SCORES: begin
        if (exp_vout) begin
          sum_e_n = sat_add32(sum_e, {8'b0, exp_y});
          got_i_n = got_i + 1'b1;
        end
        if ((got_i + (exp_vout ? 1 : 0)) >= blk_len)
          state_n = ST_UPDATE;
      end

      ST_UPDATE: begin
        l_n       = sat_add32(l_reg, sum_e);
        m_n       = m_new_r;
        m_valid_n = 1'b1;
        blk_len_n = '0;
        m_blk_n   = 16'sh8000;
        if (row_end_f)
          state_n = ST_DONE;
        else
          state_n = ST_GATHER;
        row_end_f_n = 1'b0;
      end

      ST_DONE: begin
        state_n = ST_IDLE;
      end

      default: state_n = ST_IDLE;
    endcase
  end

endmodule
