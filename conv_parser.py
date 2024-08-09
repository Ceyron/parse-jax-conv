import jax
from jax.lax import ConvDimensionNumbers
import jax.numpy as jnp
from typing import Literal
from dataclasses import dataclass
import equinox as eqx

class DimensionRepresentation1d(eqx.Module):
    lhs: tuple[
        Literal["B", "C_i"],
        Literal["C_i", "N_i"],
        Literal["N_i"],
    ] = ("B", "C_i", "N_i")
    rhs: tuple[
        Literal["C_o", "C_i"],
        Literal["C_i", "C_o"],
        Literal["K"],
    ] = ("C_o", "C_i", "K")
    out: tuple[
        Literal["B"],
        Literal["C_o"],
        Literal["N_o"],
    ] = ("B", "C_o", "N_o")

class ConvRepresentation1d(eqx.Module):
    pad: tuple[int, int] = (0, 0)
    stride: int = 1
    lhs_dil: int = 1
    rhs_dil: int = 1
    flip_kernel: bool = False
    permute_input: bool = False
    permute_kernel: bool = False

class ConvRepresentation1dSimple(eqx.Module):
    pad: tuple[int, int] = (0, 0)
    flip_kernel: bool = False
    permute_input: bool = False
    permute_kernel: bool = False

    def __init__(
        self,
        representation: ConvRepresentation1d,
    ):
        self.pad = representation.pad
        self.flip_kernel = representation.flip_kernel
        self.permute_input = representation.permute_input
        self.permute_kernel = representation.permute_kernel


def parse_dimension_representation(dimensions: ConvDimensionNumbers) -> DimensionRepresentation1d:
    lhs_dict = {
        0: "B",
        1: "C_i",
        2: "N_i",
    }
    rhs_dict = {
        0: "C_o",
        1: "C_i",
        2: "K",
    }
    out_dict = {
        0: "B",
        1: "C_o",
        2: "N_o",
    }
    return DimensionRepresentation1d(
        lhs=tuple(lhs_dict[i] for i in dimensions.lhs_spec),
        rhs=tuple(rhs_dict[i] for i in dimensions.rhs_spec),
        out=tuple(out_dict[i] for i in dimensions.out_spec),
    )

def parse_permuations(dimension_representation: DimensionRepresentation1d) -> tuple[bool, bool]:
    permute_input = dimension_representation.lhs == ("C_i", "B", "N_i")
    permute_kernel = dimension_representation.rhs == ("C_i", "C_o", "K")
    return permute_input, permute_kernel

def find_conv_start(splitted: list[str]):
    for i, line in enumerate(splitted):
        if "conv_general_dilated" in line:
            return i
    raise ValueError("conv not found")

def uses_flip(splitted: list[str]):
    uses_flip = False
    for line in splitted:
        if "rev[dimensions=(2,)]" in line:
            uses_flip = True
    return uses_flip

def extract_args(splitted: list[str], start: int) -> dict[str, str]:
    args = [
        "batch_group_count",
        "dimension_numbers",
        "feature_group_count",
        "lhs_dilation",
        "padding",
        "precision",
        "preferred_element_type",
        "rhs_dilation",
        "window_strides",
    ]
    args_mapped = {}
    ARGS_LEN = 9
    for i, arg in zip(
        range(start + 1, start+ARGS_LEN+1),
        args,
    ):
        line = splitted[i].strip()
        arg_len = len(arg)
        # Need plus one to skip the equal sign
        args_mapped[arg] = line[arg_len+1:]
    return args_mapped

def parse_args(args: dict[str, str]) -> dict[str, any]:
    parsed = {
        "pad": eval(args["padding"])[0],
        "stride": eval(args["window_strides"])[0],
        "lhs_dilation": eval(args["lhs_dilation"])[0],
        "rhs_dilation": eval(args["rhs_dilation"])[0],
        "dimension_numbers": eval(args["dimension_numbers"]),
    }
    return parsed

def parse_conv_jaxpr(jaxpr: jax.core.Jaxpr) -> ConvRepresentation1d:
    jaxpr_printed = jaxpr.pretty_print()
    jaxpr_printed_splitted: list[str] = jaxpr_printed.split("\n")
    conv_start = find_conv_start(jaxpr_printed_splitted)
    args_mapped = extract_args(jaxpr_printed_splitted, conv_start)
    args_parsed = parse_args(args_mapped)

    flip = uses_flip(jaxpr_printed_splitted)
    dimension_representation = parse_dimension_representation(args_parsed["dimension_numbers"])
    permute_input, permute_kernel = parse_permuations(dimension_representation)


    conv_representation = ConvRepresentation1d(
        pad=args_parsed["pad"],
        stride=args_parsed["stride"],
        lhs_dil=args_parsed["lhs_dilation"],
        rhs_dil=args_parsed["rhs_dilation"],
        flip_kernel=flip,
        permute_input=permute_input,
        permute_kernel=permute_kernel,
    )

    return conv_representation


class ConvParser1d:
    def __init__(
            self,
            kernel_size: int,
            stride: int,
            padding: tuple[int, int],
            lhs_dilation: int,
            rhs_dilation: int,
            array_size: int = 10,
        ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lhs_dilation = lhs_dilation
        self.rhs_dilation = rhs_dilation
        self.array_size = array_size

    def _get_input_array(self):
        return jax.random.normal(jax.random.PRNGKey(0), (1, 1, self.array_size))
    
    def _get_kernel_array(self):
        return jax.random.normal(jax.random.PRNGKey(1), (1, 1, self.kernel_size))

    def get_conv_fun(self):
        def conv_fun(lhs, rhs):
            return jax.lax.conv_general_dilated(
                lhs, rhs,
                window_strides=(self.stride,),
                padding=(self.padding,),
                lhs_dilation=(self.lhs_dilation,),
                rhs_dilation=(self.rhs_dilation,),
            )
        return conv_fun
    
    def _get_conv_fun_input(self):
        return lambda lhs: self.get_conv_fun()(lhs, self._get_kernel_array())
    
    def _get_conv_fun_kernel(self):
        return lambda rhs: self.get_conv_fun()(self._get_input_array(), rhs)
    
    def _get_conv_fun_jaxpr(self):
        return jax.make_jaxpr(self.get_conv_fun())(self._get_input_array(), self._get_kernel_array())
    
    def _get_input_vjp_fun(self):
        vjp_fun = jax.vjp(self._get_conv_fun_input(), self._get_input_array())[1]
        return lambda cotangent: vjp_fun(cotangent)[0]
    
    def _get_kernel_vjp_fun(self):
        vjp_fun = jax.vjp(self._get_conv_fun_kernel(), self._get_kernel_array())[1]
        return lambda cotangent: vjp_fun(cotangent)[0]
    
    def _get_output_array(self):
        out_shape = self.get_conv_fun()(self._get_input_array(), self._get_kernel_array()).shape
        return jax.random.normal(jax.random.PRNGKey(2), out_shape)
    
    def _get_input_vjp_fun_jaxpr(self):
        return jax.make_jaxpr(self._get_input_vjp_fun())(self._get_output_array())
    
    def _get_kernel_vjp_fun_jaxpr(self):
        return jax.make_jaxpr(self._get_kernel_vjp_fun())(self._get_output_array())
    
    def get_primal_representation(self, simplified=False) -> ConvRepresentation1d:
        primal_representation = parse_conv_jaxpr(self._get_conv_fun_jaxpr())

        if simplified:
            return ConvRepresentation1dSimple(primal_representation)
        else:
            return primal_representation
    
    def get_input_vjp_representation(self, simplified=False) -> ConvRepresentation1d:
        input_vjp_representation = parse_conv_jaxpr(self._get_input_vjp_fun_jaxpr())

        if simplified:
            return ConvRepresentation1dSimple(input_vjp_representation)
        else:
            return input_vjp_representation
    
    def get_kernel_vjp_representation(self, simplified=False) -> ConvRepresentation1d:
        kernel_vjp_representation = parse_conv_jaxpr(self._get_kernel_vjp_fun_jaxpr())

        if simplified:
            return ConvRepresentation1dSimple(kernel_vjp_representation)
        else:
            return kernel_vjp_representation
    
    def _get_delta_N(self, fun, input):
        input_N = input.shape[-1]
        output = fun(input)
        output_N = output.shape[-1]
        return output_N - input_N
    
    def get_primal_delta_N(self):
        return self._get_delta_N(self._get_conv_fun_input(), self._get_input_array())
    
    def get_input_vjp_delta_N(self):
        return self._get_delta_N(self._get_input_vjp_fun(), self._get_output_array())
    
    def get_kernel_vjp_delta_N(self):
        return self._get_delta_N(self._get_kernel_vjp_fun(), self._get_output_array())