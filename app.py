import streamlit as st

from conv_parser import ConvParser1d

st.set_page_config(layout="wide")

with st.sidebar:
    kernel_size = st.slider("Kernel size", 1, 5, 3)
    if st.toggle("Advanced options"):
        left_padding = st.slider("Left padding", 0, 5, 0)
        right_padding = st.slider("Right padding", 0, 5, 0)
        stride = st.slider("Stride", 1, 4, 1)
        lhs_dilation = st.slider("lhs dilation", 1, 3, 1)
        rhs_dilation = st.slider("rhs dilation", 1, 3, 1)
    else:
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
)

cols = st.columns(3)

with cols[0]:
    st.title("Primal")
    st.latex(r"z = w \star x = \text{conv}(x, w)")
    st.code(parser.get_primal_representation())

with cols[1]:
    st.title("Input vJp")
    st.latex(r"\bar{x} = \text{conv}(\bar{z}, w)")
    st.code(parser.get_input_vjp_representation())

with cols[2]:
    st.title("Kernel vJp")
    st.latex(r"\bar{w} = \text{conv}(x, \bar{z})")
    st.code(parser.get_kernel_vjp_representation())