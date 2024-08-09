import streamlit as st

from conv_parser import ConvParser1d

st.set_page_config(layout="wide")

with st.sidebar:
    kernel_size = st.slider("Kernel size", 1, 5, 3)
    advanced_options = st.toggle("Advanced options")
    if advanced_options:
        num_points = st.slider("Number of points", 80, 120, 100)
        left_padding = st.slider("Left padding", 0, 5, 0)
        right_padding = st.slider("Right padding", 0, 5, 0)
        stride = st.slider("Stride", 1, 4, 1)
        lhs_dilation = st.slider("lhs dilation", 1, 3, 1)
        rhs_dilation = st.slider("rhs dilation", 1, 3, 1)
    else:
        num_points = 100
        left_padding = 0
        right_padding = 0
        stride = 1
        lhs_dilation = 1
        rhs_dilation = 1

parser = ConvParser1d(
    kernel_size=kernel_size,
    stride=stride,
    padding=(left_padding, right_padding),
    lhs_dilation=lhs_dilation,
    rhs_dilation=rhs_dilation,
    array_size=num_points,
)

cols = st.columns(3)

with cols[0]:
    st.title("Primal")
    st.latex(r"z = w \star x = \text{conv}(x, w)")
    st.code(parser.get_primal_representation(simplified=not advanced_options))
    delta_N = parser.get_primal_delta_N()
    st.latex(f"\Delta N = {delta_N}")

with cols[1]:
    st.title("Input vJp")
    st.latex(r"\bar{x} = \text{conv}(\bar{z}, w)")
    st.code(parser.get_input_vjp_representation(simplified=not advanced_options))
    delta_N = parser.get_input_vjp_delta_N()
    st.latex(f"\Delta N = {delta_N}")

with cols[2]:
    st.title("Kernel vJp")
    st.latex(r"\bar{w} = \text{conv}(x, \bar{z})")
    st.code(parser.get_kernel_vjp_representation(simplified=not advanced_options))
    delta_N = parser.get_kernel_vjp_delta_N()
    st.latex(f"\Delta N = {delta_N}")

st.markdown("---")
st.markdown("IMPORTANT: despite saying it is 'convolution', this is actually cross-correlation!")

with st.expander("Insights"):
    st.markdown(
r"""
## Simplified options

Means:

- No padding ("VALID")
- no modifications of convolution (no striding and no dilations of input or
  kernel)

In this case, we have for the primal computation:

$$ N_o = N_i - K + 1 $$

Hence,

$$ \Delta N = N_o - N_i = 1 - K $$

The filter vJp has to compensate by adding $\Delta N$ per side in padding.
Additionally, it flips the filter (reverse the spatial axis) and permutes $C_i$
and $C_o$ axis of the filter.

The kernel vJp has to permute the original input $x$ (swap batch $B$ and in
channel $C_i$ axis) and has to permute the cotangent output $\bar{z}$ which acts
as the kernel.

## Advanced options

### Influence of Padding

TODO

### Incluence of Stride, Left-Hand Side Dilation and Right-Hand Side Dilation

If we do the following in the primal, this affects the respective vJps as:

- primal: stride
    - input vJp: lhs dilation
    - kernel vjp: rhs dilation
- primal: lhs dilation  ("transposed convolution")
    - input vJp: stride
    - kernel vjp: lhs dilation
- primal: rhs dilation  ("dilated convolution")
    - input vJp: rhs dilation
    - kernel vjp: stride

(Note that there are potential adjustments to the padding.)
"""
    )