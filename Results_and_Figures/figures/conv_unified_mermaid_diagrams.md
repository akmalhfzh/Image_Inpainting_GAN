## Block Diagram 1: MAC Array (`mac_array.v`)

Catatan: Semua operasi paralel untuk 16 tap (m = 0..15).
Input dari luar conv_unified: `bias_val[15:0]` dari **generator_top (BIAS_MEM)**.
Input dari dalam conv_unified: `weight_vec`, `ifmap_vec` dari **VEC_PACK**; `valid_vec_w` dari **addr_gen**; `mac_in_valid` dari **AXI4_MASTER_FSM** (`state == ST_COMPUTE_ISSUE`).

```mermaid
graph TD
    %% ── External Inputs ──
    EXT_W["<b>WEIGHT_CACHE</b><br/>(inside conv_unified)<br/>weight_tile[0:15]"]
    EXT_I["<b>IFMAP_TILE</b><br/>(inside conv_unified)<br/>ifmap_tile[0:15]"]
    EXT_V["<b>ADDR_GEN</b><br/>(inside conv_unified)<br/>valid_vec_w[15:0]"]
    EXT_CLK["<b>AXI4_MASTER_FSM</b><br/>(inside conv_unified)<br/>mac_in_valid"]

    %% ── Vec Flatten (generate loop inside conv_unified) ──
    FLAT_W["VEC_PACK<br/>(generate loop)<br/>weight_vec[255:0]"]
    FLAT_I["VEC_PACK<br/>(generate loop)<br/>ifmap_vec[255:0]"]
    EXT_W -->|"weight_tile[m]"| FLAT_W
    EXT_I -->|"ifmap_tile[m]"| FLAT_I

    %% ── Stage 0: Zero-Pad MUX + Multiplier (x16 parallel) ──
    subgraph STAGE0["<b>Stage 0: 16x Parallel Multiply</b> (combinational)"]
        direction TB
        ZERO["Constant<br/>16'd0"]
        MUX_ZP["MUX 2:1<br/>(zero-pad)"]
        WVAL["wval[15:0]<br/>(signed extract)"]
        IVAL["ival[15:0]"]
        MULT["<b>MULTIPLIER</b><br/>Signed 16b × 16b"]
        PROD["prod[m]<br/>[31:0] signed"]
        SEXT["SIGN EXTEND<br/>32b → 48b"]
        PSX["prod_sx[m]<br/>[47:0] signed"]
    end

    FLAT_W -->|"weight_vec[m*16 +: 16]"| WVAL
    FLAT_I -->|"ifmap_vec[m*16 +: 16]"| MUX_ZP
    EXT_V -->|"valid_vec[m]<br/>(select)"| MUX_ZP
    ZERO -->|"if !valid"| MUX_ZP
    MUX_ZP --> IVAL
    WVAL -->|"wval_32"| MULT
    IVAL -->|"ival_32"| MULT
    MULT --> PROD
    PROD --> SEXT
    SEXT --> PSX

    %% ── Stage 1: Pipeline Register ──
    PREG["<b>PIPELINE REG</b><br/>(D-FlipFlop)<br/>prod_reg[m][47:0]"]
    PSX -->|"prod_sx[m]"| PREG

    %% ── Stage 1 cont.: Binary Adder Tree ──
    subgraph TREE["<b>Stage 1: Binary Adder Tree</b> (adder_tree.v, combinational)"]
        direction TB
        L1_0["<b>ADDER</b> L1_0<br/>prod_reg[0]+prod_reg[1]"]
        L1_1["<b>ADDER</b> L1_1<br/>prod_reg[2]+prod_reg[3]"]
        L1_2["<b>ADDER</b> L1_2<br/>prod_reg[4]+prod_reg[5]"]
        L1_3["<b>ADDER</b> L1_3<br/>prod_reg[6]+prod_reg[7]"]
        L1_4["<b>ADDER</b> L1_4<br/>prod_reg[8]+prod_reg[9]"]
        L1_5["<b>ADDER</b> L1_5<br/>prod_reg[10]+prod_reg[11]"]
        L1_6["<b>ADDER</b> L1_6<br/>prod_reg[12]+prod_reg[13]"]
        L1_7["<b>ADDER</b> L1_7<br/>prod_reg[14]+prod_reg[15]"]

        L2_0["<b>ADDER</b> L2_0"]
        L2_1["<b>ADDER</b> L2_1"]
        L2_2["<b>ADDER</b> L2_2"]
        L2_3["<b>ADDER</b> L2_3"]

        L3_0["<b>ADDER</b> L3_0"]
        L3_1["<b>ADDER</b> L3_1"]

        L4["<b>ADDER</b> FINAL<br/>tree_out[47:0]"]

        L1_0 -->|"sum_left"| L2_0
        L1_1 -->|"sum_right"| L2_0
        L1_2 -->|"sum_left"| L2_1
        L1_3 -->|"sum_right"| L2_1
        L1_4 -->|"sum_left"| L2_2
        L1_5 -->|"sum_right"| L2_2
        L1_6 -->|"sum_left"| L2_3
        L1_7 -->|"sum_right"| L2_3

        L2_0 -->|"sum_left"| L3_0
        L2_1 -->|"sum_right"| L3_0
        L2_2 -->|"sum_left"| L3_1
        L2_3 -->|"sum_right"| L3_1

        L3_0 -->|"sum_left"| L4
        L3_1 -->|"sum_right"| L4
    end

    PREG -->|"prod_flat[m*48 +: 48]"| L1_0
    PREG -->|"prod_flat[m*48 +: 48]"| L1_1
    PREG -->|"prod_flat[m*48 +: 48]"| L1_2
    PREG -->|"prod_flat[m*48 +: 48]"| L1_3
    PREG -->|"prod_flat[m*48 +: 48]"| L1_4
    PREG -->|"prod_flat[m*48 +: 48]"| L1_5
    PREG -->|"prod_flat[m*48 +: 48]"| L1_6
    PREG -->|"prod_flat[m*48 +: 48]"| L1_7

    %% ── Stage 2: Output Register ──
    OREG["<b>PIPELINE REG</b><br/>(D-FlipFlop)<br/>sum_out_reg[47:0]"]
    L4 -->|"tree_out"| OREG

    %% ── Valid Shift Register ──
    VSREG["<b>SHIFT REG</b><br/>valid_sr[1:0]<br/>2-cycle delay"]
    EXT_CLK -->|"in_valid"| VSREG

    %% ── Output ──
    OUT_SUM["<b>OUTPUT:</b><br/>sum_out = row_sum[47:0]<br/>→ ke ACCUMULATOR"]
    OUT_VAL["<b>OUTPUT:</b><br/>out_valid = mac_out_valid<br/>→ ke ACCUMULATOR"]
    OREG -->|"sum_out_reg"| OUT_SUM
    VSREG -->|"valid_sr[1]"| OUT_VAL
```


## Block Diagram 2: Address Generator (`addr_gen.v`)

Catatan: Diagram ini menunjukkan datapath untuk SATU tap (t). Hardware ini di-generate 16x secara paralel.
Semua operasi bersifat **combinational** (tidak ada clock/register di dalam addr_gen).
Mode: `cfg_mode=0` → Conv2D; `cfg_mode=1` → ConvTranspose2D. Diagram ini menunjukkan path Conv2D.

```mermaid
graph TD
    %% ── External Inputs ──
    EXT_CFG["<b>CONFIG_LUT</b><br/>(dari generator_top)<br/>cfg_h_in, cfg_w_in,<br/>cfg_c_in, cfg_c_out,<br/>cfg_k, cfg_stride,<br/>cfg_pad, cfg_mode,<br/>cfg_weight_base[25:0]"]
    EXT_POS["<b>AXI4_MASTER_FSM</b><br/>(dari conv_unified)<br/>h_out[9:0], w_out[9:0],<br/>c_out[11:0], c_in[11:0]"]

    %% ── Pre-compute ──
    subgraph PRECOMP["<b>Pre-compute</b> (combinational wires)"]
        direction TB
        HW_MUL["<b>MULTIPLIER</b><br/>hw_in = cfg_h_in × cfg_w_in"]
        KK_MUL["<b>MULTIPLIER</b><br/>kk = cfg_k × cfg_k"]
        CKK_MUL["<b>MULTIPLIER</b><br/>cin_x_kk = cfg_c_in × kk"]
        SC_MUL["<b>MULTIPLIER</b><br/>stride_cout = c_out × cin_x_kk"]
        SK_MUL["<b>MULTIPLIER</b><br/>step_cin = c_in × kk"]
    end

    EXT_CFG -->|"cfg_h_in, cfg_w_in"| HW_MUL
    EXT_CFG -->|"cfg_k"| KK_MUL
    EXT_CFG -->|"cfg_c_in"| CKK_MUL
    KK_MUL -->|"kk"| CKK_MUL
    EXT_POS -->|"c_out"| SC_MUL
    CKK_MUL -->|"cin_x_kk"| SC_MUL
    EXT_POS -->|"c_in"| SK_MUL
    KK_MUL -->|"kk"| SK_MUL

    %% ── Per-tap (t = 0..15) Coordinate Calculation ──
    subgraph TAP["<b>Per-Tap Coordinate</b> (×16 paralel, cfg_mode=0 Conv2D)"]
        direction TB

        DY["dy = t / cfg_k<br/>dx = t % cfg_k<br/>(integer divide)"]

        MUL_Y["<b>MULTIPLIER</b><br/>h_out × cfg_stride"]
        SUB_Y["<b>SUBTRACTOR</b><br/>− cfg_pad"]
        ADD_Y["<b>ADDER</b><br/>+ dy"]
        INY["in_y"]

        MUL_X["<b>MULTIPLIER</b><br/>w_out × cfg_stride"]
        SUB_X["<b>SUBTRACTOR</b><br/>− cfg_pad"]
        ADD_X["<b>ADDER</b><br/>+ dx"]
        INX["in_x"]

        EXT_POS2["h_out, w_out"]
    end

    EXT_POS -->|"h_out"| MUL_Y
    EXT_CFG -->|"cfg_stride"| MUL_Y
    MUL_Y --> SUB_Y
    EXT_CFG -->|"cfg_pad"| SUB_Y
    SUB_Y --> ADD_Y
    DY -->|"dy"| ADD_Y
    ADD_Y --> INY

    EXT_POS -->|"w_out"| MUL_X
    EXT_CFG -->|"cfg_stride"| MUL_X
    MUL_X --> SUB_X
    EXT_CFG -->|"cfg_pad"| SUB_X
    SUB_X --> ADD_X
    DY -->|"dx"| ADD_X
    ADD_X --> INX

    %% ── Boundary Check ──
    subgraph VALID["<b>Boundary/Padding Check</b>"]
        direction TB
        CMP_YGE["<b>COMPARATOR</b><br/>in_y >= 0"]
        CMP_YLT["<b>COMPARATOR</b><br/>in_y < cfg_h_in"]
        CMP_XGE["<b>COMPARATOR</b><br/>in_x >= 0"]
        CMP_XLT["<b>COMPARATOR</b><br/>in_x < cfg_w_in"]
        AND1["<b>AND</b>"]
        AND2["<b>AND</b>"]
        AND_F["<b>AND</b><br/>valid_vec[t]"]
    end

    INY -->|"in_y"| CMP_YGE
    INY -->|"in_y"| CMP_YLT
    EXT_CFG -->|"cfg_h_in"| CMP_YLT
    INX -->|"in_x"| CMP_XGE
    INX -->|"in_x"| CMP_XLT
    EXT_CFG -->|"cfg_w_in"| CMP_XLT

    CMP_YGE --> AND1
    CMP_YLT --> AND1
    CMP_XGE --> AND2
    CMP_XLT --> AND2
    AND1 -->|"is_y_valid"| AND_F
    AND2 -->|"is_x_valid"| AND_F

    %% ── ifmap Address ──
    subgraph IFADDR["<b>ifmap Address Calculation</b>"]
        direction TB
        MUL_CIN["<b>MULTIPLIER</b><br/>c_in × hw_in"]
        MUL_IY["<b>MULTIPLIER</b><br/>in_y × cfg_w_in"]
        ADD_IF1["<b>ADDER</b><br/>+ in_x"]
        ADD_IF2["<b>ADDER</b><br/>c_in_offset + row_offset"]
        IFOUT["ifmap_addrs[t]<br/>[17:0]"]
    end

    EXT_POS -->|"c_in"| MUL_CIN
    HW_MUL -->|"hw_in"| MUL_CIN
    INY -->|"in_y"| MUL_IY
    EXT_CFG -->|"cfg_w_in"| MUL_IY
    MUL_IY --> ADD_IF1
    INX -->|"in_x"| ADD_IF1
    MUL_CIN --> ADD_IF2
    ADD_IF1 --> ADD_IF2
    ADD_IF2 --> IFOUT

    %% ── Weight Address ──
    subgraph WADDR["<b>Weight Address Calculation</b>"]
        direction TB
        ADD_W1["<b>ADDER</b><br/>cfg_weight_base + stride_cout"]
        ADD_W2["<b>ADDER</b><br/>+ step_cin"]
        ADD_W3["<b>ADDER</b><br/>+ t"]
        WOUT["weight_addrs[t]<br/>[25:0]"]
    end

    EXT_CFG -->|"cfg_weight_base"| ADD_W1
    SC_MUL -->|"stride_cout"| ADD_W1
    ADD_W1 --> ADD_W2
    SK_MUL -->|"step_cin"| ADD_W2
    ADD_W2 --> ADD_W3
    ADD_W3 --> WOUT

    %% ── Outputs ──
    OUT1["<b>OUTPUT:</b><br/>valid_vec[15:0]<br/>→ ke MAC_ARRAY + PIPE_REG"]
    OUT2["<b>OUTPUT:</b><br/>ifmap_addrs[16×18b]<br/>→ ke PIPE_REG → BURST_CALC → DDR"]
    OUT3["<b>OUTPUT:</b><br/>weight_addrs[16×26b]<br/>→ ke PIPE_REG → BURST_CALC → DDR"]

    AND_F --> OUT1
    IFOUT --> OUT2
    WOUT --> OUT3
```


## Block Diagram 3: Accumulator + Requantization (inside `conv_unified.v`)

Input dari luar conv_unified: `bias_val[15:0]` dari **generator_top (BIAS_MEM)**.
Input dari dalam conv_unified: `row_sum[47:0]` + `mac_out_valid` dari **MAC_ARRAY**; `is_first` dari **AXI4_MASTER_FSM** (`c_in_cnt == 0`).

```mermaid
graph TD
    %% ── External Inputs ──
    EXT_BIAS["<b>BIAS_MEM</b><br/>(dari generator_top)<br/>bias_val[15:0] signed"]
    EXT_MAC["<b>MAC_ARRAY</b><br/>(output)<br/>row_sum[47:0] signed"]
    EXT_MACV["<b>MAC_ARRAY</b><br/>(output)<br/>mac_out_valid"]
    EXT_FIRST["<b>AXI4_MASTER_FSM</b><br/>(control)<br/>is_first = (c_in_cnt == 0)"]

    %% ── Bias Sign Extension + Shift ──
    subgraph BIAS_EXT["<b>Bias Extension</b>"]
        SEXT_B["<b>SIGN EXTEND</b><br/>16b → 48b"]
        SHFT_B["<b>LEFT SHIFT</b><br/>&lt;&lt;&lt; FRAC_W (=8)"]
        BEXT["bias_extended<br/>[47:0] signed"]
    end
    EXT_BIAS -->|"bias_val[15:0]"| SEXT_B
    SEXT_B --> SHFT_B
    SHFT_B --> BEXT

    %% ── Accumulation Logic ──
    subgraph ACCUM["<b>Accumulator</b>"]
        direction TB
        MUX_BA["<b>MUX 2:1</b><br/>is_first ? bias_ext : acc_reg"]
        ADD_ACC["<b>ADDER</b><br/>48-bit signed"]
        ACCNXT["acc_next<br/>[47:0]"]
        ACCREG["<b>D-FLIPFLOP</b><br/>acc_reg[47:0]<br/>(latched when mac_out_valid)"]
    end

    BEXT -->|"bias_extended"| MUX_BA
    ACCREG -->|"acc_reg (feedback)"| MUX_BA
    EXT_FIRST -->|"is_first (select)"| MUX_BA
    MUX_BA -->|"mux_out"| ADD_ACC
    EXT_MAC -->|"row_sum[47:0]"| ADD_ACC
    ADD_ACC --> ACCNXT
    ACCNXT -->|"acc_next"| ACCREG
    EXT_MACV -->|"mac_out_valid<br/>(clock enable)"| ACCREG

    %% ── Requantization: Rounding ──
    subgraph REQUANT["<b>Requantize + Saturate</b> (combinational)"]
        direction TB
        HALF["Constant<br/>1 &lt;&lt; (FRAC_W−1)<br/>= 128 (0.5 ULP)"]
        ADD_RND["<b>ADDER</b><br/>48-bit: acc_reg + 0.5 ULP"]
        ARND["acc_rounded[47:0]"]
        RSHIFT["<b>ARITH RIGHT SHIFT</b><br/>&gt;&gt;&gt; FRAC_W (=8)<br/>acc_rounded[47:8]"]
        ASHFT["acc_shifted[39:0]"]

        %% Saturation
        CMAX["Constant<br/>+32767"]
        CMIN["Constant<br/>−32768"]
        CMP_POS["<b>COMPARATOR</b><br/>acc_shifted > +32767"]
        CMP_NEG["<b>COMPARATOR</b><br/>acc_shifted < −32768"]
        MUX_S1["<b>MUX 2:1</b><br/>sat_neg ? SAT_MIN : acc_shifted"]
        MUX_S2["<b>MUX 2:1</b><br/>sat_pos ? SAT_MAX : mux1_out"]
        ATRUNC["acc_trunc[15:0]<br/>signed INT16"]
    end

    ACCREG -->|"acc_reg[47:0]"| ADD_RND
    HALF --> ADD_RND
    ADD_RND --> ARND
    ARND --> RSHIFT
    RSHIFT --> ASHFT

    ASHFT --> CMP_POS
    CMAX --> CMP_POS
    ASHFT --> CMP_NEG
    CMIN --> CMP_NEG
    ASHFT -->|"acc_shifted[15:0]"| MUX_S1
    CMIN --> MUX_S1
    CMP_NEG -->|"sat_neg (select)"| MUX_S1
    MUX_S1 --> MUX_S2
    CMAX --> MUX_S2
    CMP_POS -->|"sat_pos (select)"| MUX_S2
    MUX_S2 --> ATRUNC

    %% ── Outputs ──
    OUT_TRUNC["<b>OUTPUT:</b><br/>acc_trunc[15:0]<br/>→ ke ACTIVATION MUX"]
    OUT_COUT["<b>OUTPUT:</b><br/>current_c_out = c_out_cnt<br/>→ ke generator_top (BIAS_MEM)"]
    ATRUNC --> OUT_TRUNC
```


## Block Diagram 4: Parallel Activation + Output MUX (inside `conv_unified.v`)

Input: `acc_trunc[15:0]` dari **REQUANT_SAT** (diagram 3).
Control: `cur_act[1:0]` dari **CONFIG_LUT** (di generator_top).
Output: `final_pixel[15:0]` → ke **AXI4 WRITE** → DDR.

```mermaid
graph TD
    %% ── Input ──
    IN_TRUNC["<b>REQUANT_SAT</b><br/>acc_trunc[15:0]<br/>signed INT16"]
    IN_ACT["<b>CONFIG_LUT</b><br/>(dari generator_top)<br/>cur_act[1:0]"]

    %% ── Leaky ReLU (leaky_relu.v) ──
    subgraph LRELU["<b>Leaky ReLU</b> (u_lrelu, combinational)"]
        direction TB
        LR_CMP["<b>COMPARATOR</b><br/>x_in >= 0"]
        LR_CONST["Constant<br/>SLOPE = 51<br/>(≈ 0.2 × 256)"]
        LR_MULT["<b>MULTIPLIER</b><br/>x_in × 51<br/>(16b × 16b = 32b)"]
        LR_SHIFT["<b>ARITH RIGHT SHIFT</b><br/>neg_prod >>> FRAC_W (=8)"]
        LR_NEGV["neg_val[15:0]"]
        LR_MUX["<b>MUX 2:1</b><br/>x_in >= 0 ? x_in : neg_val"]
        LR_OUT["lrelu_out[15:0]"]
    end

    IN_TRUNC -->|"x_in"| LR_CMP
    IN_TRUNC -->|"x_in"| LR_MULT
    LR_CONST --> LR_MULT
    LR_MULT -->|"neg_prod[31:0]"| LR_SHIFT
    LR_SHIFT --> LR_NEGV
    IN_TRUNC -->|"x_in (pos path)"| LR_MUX
    LR_NEGV -->|"neg_val (neg path)"| LR_MUX
    LR_CMP -->|"is_pos (select)"| LR_MUX
    LR_MUX --> LR_OUT

    %% ── ReLU (relu.v) ──
    subgraph RELU["<b>ReLU</b> (u_relu, combinational)"]
        direction TB
        R_CMP["<b>COMPARATOR</b><br/>x_in >= 0"]
        R_ZERO["Constant<br/>16'd0"]
        R_MUX["<b>MUX 2:1</b><br/>x_in >= 0 ? x_in : 0"]
        R_OUT["relu_out[15:0]"]
    end

    IN_TRUNC -->|"x_in"| R_CMP
    IN_TRUNC -->|"x_in (pos path)"| R_MUX
    R_ZERO -->|"0 (neg path)"| R_MUX
    R_CMP -->|"is_pos (select)"| R_MUX
    R_MUX --> R_OUT

    %% ── Tanh (tanh_act.v) ──
    subgraph TANH["<b>Tanh</b> (u_tanh, combinational)"]
        direction TB
        T_SIGN["<b>COMPARATOR</b><br/>x_in[15] (sign bit)"]
        T_NEG["<b>NEGATOR</b><br/>−x_in (2s complement)"]
        T_ABSMUX["<b>MUX 2:1</b><br/>x_in[15] ? −x_in : x_in"]
        T_ABS["x_abs[15:0]"]
        T_RSHIFT["<b>RIGHT SHIFT</b><br/>x_abs >> 5<br/>(÷ 0.125 seg)"]
        T_SEGCLAMP["<b>COMPARATOR + MUX</b><br/>seg >= 32 ? 31 : seg"]
        T_FRAC["frac = x_abs[4:0]<br/>(lower bits)"]
        T_LUT_LO["<b>LUT READ</b><br/>lut[seg_idx]<br/>= lo[15:0]"]
        T_LUT_HI["<b>LUT READ</b><br/>lut[seg_idx+1]<br/>= hi[15:0]"]
        T_DIFF_SUB["<b>SUBTRACTOR</b><br/>diff = hi − lo"]
        T_INTERP_MUL["<b>MULTIPLIER</b><br/>diff × frac"]
        T_INTERP_SHR["<b>RIGHT SHIFT</b><br/>interp_full >> 5"]
        T_ADD["<b>ADDER</b><br/>lo + interp_val"]
        T_SAT_MUX["<b>MUX 2:1</b><br/>sat ? ONE : interp_val"]
        T_MAG["mag[15:0]"]
        T_SIGN_MUX["<b>MUX 2:1</b><br/>x_in[15] ? −mag : mag"]
        T_OUT["tanh_out_w[15:0]"]
    end

    IN_TRUNC -->|"x_in"| T_SIGN
    IN_TRUNC -->|"x_in"| T_NEG
    IN_TRUNC -->|"x_in"| T_ABSMUX
    T_NEG -->|"−x_in"| T_ABSMUX
    T_SIGN -->|"sign (select)"| T_ABSMUX
    T_ABSMUX --> T_ABS
    T_ABS --> T_RSHIFT
    T_RSHIFT -->|"seg_full"| T_SEGCLAMP
    T_ABS -->|"lower bits"| T_FRAC
    T_SEGCLAMP -->|"seg_idx"| T_LUT_LO
    T_SEGCLAMP -->|"seg_idx+1"| T_LUT_HI
    T_LUT_HI -->|"hi"| T_DIFF_SUB
    T_LUT_LO -->|"lo"| T_DIFF_SUB
    T_DIFF_SUB -->|"diff"| T_INTERP_MUL
    T_FRAC -->|"frac"| T_INTERP_MUL
    T_INTERP_MUL -->|"interp_full"| T_INTERP_SHR
    T_LUT_LO -->|"lo"| T_ADD
    T_INTERP_SHR -->|"interp_val"| T_ADD
    T_ADD -->|"interp_val"| T_SAT_MUX
    T_SEGCLAMP -->|"sat (select)"| T_SAT_MUX
    T_SAT_MUX --> T_MAG
    T_MAG --> T_SIGN_MUX
    T_SIGN -->|"sign (select)"| T_SIGN_MUX
    T_SIGN_MUX --> T_OUT

    %% ── Final 4:1 MUX ──
    subgraph FMUX["<b>Final Activation MUX</b> (combinational)"]
        direction TB
        ACT_MUX["<b>MUX 4:1</b><br/>cur_act[1:0]:<br/>0 → lrelu_out<br/>1 → relu_out<br/>2 → tanh_out_w<br/>3 → acc_trunc (bypass)"]
        FP["final_pixel[15:0]"]
    end

    LR_OUT -->|"lrelu_out"| ACT_MUX
    R_OUT -->|"relu_out"| ACT_MUX
    T_OUT -->|"tanh_out_w"| ACT_MUX
    IN_TRUNC -->|"acc_trunc<br/>(bypass, cur_act=3)"| ACT_MUX
    IN_ACT -->|"cur_act[1:0]<br/>(select)"| ACT_MUX
    ACT_MUX --> FP

    %% ── Output ──
    OUT_FP["<b>OUTPUT:</b><br/>final_pixel[15:0]<br/>→ ke AXI WRITE → DDR"]
    FP --> OUT_FP
```
