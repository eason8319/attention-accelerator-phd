# P4 RTL — Verilator 仿真 Makefile
#
# 用法：
#   make check         # 冒烟测试：验证 Verilator 工具链可用
#   make sim-exp       # 模块 A：exp 近似单元 对拍
#   make sim-softmax   # 模块 B：online softmax 归约单元 对拍
#   make sim-sa        # 模块 C：4x4 systolic array 对拍
#   make sim-all       # 三个模块全部跑一遍
#   make clean
#
# 约定：
# - Python 一律通过 conda env `p4-rtl` 运行（见 environment.yml）。
# - 测试向量与 DUT 输出均为文本文件，格式见 scripts/README.md。
# - 每个 sim-* 目标 = 生成向量(gen) -> verilate+运行(TB 自比对或落盘) -> Python 比对(compare)。
#   TB 内部比对失败时以非零码退出（$fatal），make 随之失败。

VERILATOR    ?= verilator
PYTHON       ?= conda run -n p4-rtl python
BUILD_DIR    := build
VFLAGS       := --binary -j 0 -Wall --trace --x-assign unique --x-initial unique \
                -Wno-UNUSEDSIGNAL -Wno-SYNCASYNCNET -Wno-MULTIDRIVEN

# 各目标的源文件与顶层（模块实现后填充/生效）
EXP_SRCS     := exp_unit/exp_approx.sv tb/tb_exp.sv
SOFTMAX_SRCS := exp_unit/exp_approx.sv softmax_unit/online_softmax.sv tb/tb_softmax.sv
SA_SRCS      := systolic_array/pe.sv systolic_array/systolic_array.sv tb/tb_systolic.sv
SMOKE_SRCS   := tb/tb_smoke.sv

.PHONY: check sim-exp sim-softmax sim-sa sim-all clean

# ---------------------------------------------------------------- smoke
check: $(BUILD_DIR)/smoke/Vtb_smoke
	$(BUILD_DIR)/smoke/Vtb_smoke

$(BUILD_DIR)/smoke/Vtb_smoke: $(SMOKE_SRCS)
	mkdir -p $(BUILD_DIR)/smoke
	$(VERILATOR) $(VFLAGS) --Mdir $(BUILD_DIR)/smoke --top-module tb_smoke $(SMOKE_SRCS)

# ---------------------------------------------------------------- 模块 A: exp
sim-exp: $(EXP_SRCS)
	mkdir -p $(BUILD_DIR)/exp
	$(PYTHON) scripts/gen_vecs_exp.py --out $(BUILD_DIR)/vec_exp
	$(VERILATOR) $(VFLAGS) --Mdir $(BUILD_DIR)/exp --top-module tb_exp $(EXP_SRCS)
	cd $(BUILD_DIR)/exp && ./Vtb_exp +vec_dir=../vec_exp
	$(PYTHON) scripts/compare_exp.py --vec $(BUILD_DIR)/vec_exp

# ---------------------------------------------------------------- 模块 B: softmax
sim-softmax: $(SOFTMAX_SRCS)
	mkdir -p $(BUILD_DIR)/softmax
	$(PYTHON) scripts/gen_vecs_softmax.py --out $(BUILD_DIR)/vec_softmax --seq 64 --block-size 8
	$(VERILATOR) $(VFLAGS) --Mdir $(BUILD_DIR)/softmax --top-module tb_softmax $(SOFTMAX_SRCS)
	cd $(BUILD_DIR)/softmax && ./Vtb_softmax +vec_dir=../vec_softmax
	$(PYTHON) scripts/compare_softmax.py --vec $(BUILD_DIR)/vec_softmax

# ---------------------------------------------------------------- 模块 C: systolic array
sim-sa: $(SA_SRCS)
	mkdir -p $(BUILD_DIR)/sa
	$(PYTHON) scripts/gen_vecs_sa.py --out $(BUILD_DIR)/vec_sa
	$(VERILATOR) $(VFLAGS) --Mdir $(BUILD_DIR)/sa --top-module tb_systolic $(SA_SRCS)
	cd $(BUILD_DIR)/sa && ./Vtb_systolic +vec_dir=../vec_sa
	$(PYTHON) scripts/compare_sa.py --vec $(BUILD_DIR)/vec_sa

sim-all: sim-exp sim-softmax sim-sa

clean:
	rm -rf $(BUILD_DIR)
