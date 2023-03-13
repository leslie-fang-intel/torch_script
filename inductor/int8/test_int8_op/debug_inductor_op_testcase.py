import torch

def test():
    raw_x = [[[[ 0.0000, -3.0000,  1.5000],
               [ 1.5000,  1.5000, -1.5000],
               [-3.0000, -1.5000,  0.0000]],

             [[ 1.5000, -3.0000, -3.0000],
             [-1.5000,  0.0000,  1.5000],
             [ 0.0000,  1.5000,  1.5000]],

          [[ 1.5000, -3.0000, -3.0000],
           [-1.5000, -1.5000, -1.5000],
          [-3.0000, -1.5000, -1.5000]],  

         [[-1.5000,  0.0000,  1.5000],
          [ 0.0000,  1.5000,  1.5000],
           [ 1.5000, -3.0000, -3.0000]]],

         [[[ 0.0000,  0.0000, -3.0000],
           [ 1.5000, -3.0000,  0.0000],
           [-3.0000,  0.0000,  1.5000]],

          [[ 0.0000, -1.5000, -3.0000],
           [ 0.0000, -1.5000,  1.5000],
           [ 0.0000,  0.0000, -1.5000]],

          [[ 0.0000, -1.5000,  0.0000],
          [ 0.0000,  0.0000, -1.5000],
          [ 0.0000,  0.0000,  1.5000]],

          [[-3.0000, -1.5000, -3.0000],
           [-3.0000, -1.5000, -1.5000],
          [ 1.5000,  0.0000,  0.0000]]],

         [[[ 0.0000,  0.0000, -1.5000],
          [-3.0000,  0.0000, -1.5000],
           [ 1.5000,  0.0000,  1.5000]],

          [[-1.5000, -3.0000,  1.5000],
           [-1.5000,  0.0000,  0.0000],
           [-1.5000,  0.0000, -1.5000]],
 
          [[ 1.5000, -1.5000,  1.5000],
          [-3.0000,  0.0000,  0.0000],
          [ 0.0000, -1.5000,  1.5000]],
 
          [[-3.0000,  1.5000,  1.5000],
           [-1.5000,  0.0000,  0.0000],
           [ 1.5000,  0.0000,  1.5000]]]]

    x = torch.tensor(raw_x, dtype=torch.float)
    x_scale = 1.5
    x_zero_point = 2
    x_q = torch.quantize_per_tensor(x, scale=x_scale, zero_point=x_zero_point,
            dtype=torch.quint8)
    # x_q = torch._make_per_tensor_quantized_tensor(x, x_scale, x_zero_point)
    x = x_q.dequantize()

    raw_w = [[[[ 1.5000, -6.0000, -3.0000],
           [-4.5000,  1.5000,  1.5000],
           [-3.0000, -6.0000,  0.0000]],
 
          [[ 0.0000,  3.0000,  1.5000],
           [ 4.5000, -7.5000,  1.5000],
           [ 4.5000,  6.0000, -7.5000]]],
 
 
         [[[-3.0000,  1.5000,  1.5000],
           [ 6.0000, -4.5000, -6.0000],
           [ 4.5000, -4.5000, -6.0000]],
 
         [[-7.5000, -1.5000,  4.5000],
           [-1.5000, -1.5000,  4.5000],
           [ 3.0000,  3.0000,  1.5000]]],


         [[[ 4.5000, -4.5000, -3.0000],
           [-3.0000, -7.5000,  3.0000],
           [ 1.5000,  4.5000, -1.5000]],
 
          [[ 1.5000,  6.0000, -6.0000],
           [-4.5000,  4.5000, -7.5000],
           [ 6.0000, -7.5000,  6.0000]]],
 
 
         [[[-7.5000, -6.0000,  6.0000],
          [ 6.0000,  3.0000, -4.5000],
           [-6.0000, -3.0000,  0.0000]],
 
          [[ 4.5000, -4.5000,  6.0000],
           [ 4.5000,  3.0000,  1.5000],
          [-3.0000, -3.0000, -1.5000]]]]
    w = torch.tensor(raw_w, dtype=torch.float)
    w_scale = 1.5
    w_zero_point = 0
    w_q = torch.quantize_per_tensor(w, scale=w_scale, zero_point=w_zero_point,
            dtype=torch.qint8)
    #w_q = torch._make_per_tensor_quantized_tensor(w, w_scale, w_zero_point)
    w = w_q.dequantize()

    raw_x2 = [[[[-1.2000, -2.4000],
           [-2.4000, -4.8000]],
 
          [[-3.6000, -4.8000],
           [-1.2000, -4.8000]],
 
          [[-2.4000, -2.4000],
          [-1.2000, -3.6000]],

         [[-4.8000, -2.4000],
           [-3.6000, -2.4000]]],

 
         [[[-3.6000, -4.8000],
          [-3.6000, -3.6000]],
 
          [[-2.4000, -1.2000],
           [-2.4000, -2.4000]],
 
          [[-4.8000, -4.8000],
          [-2.4000, -4.8000]],
 
          [[-1.2000, -3.6000],
           [-2.4000, -1.2000]]],
 
 
         [[[-3.6000, -1.2000],
           [-4.8000, -1.2000]],
 
          [[-1.2000, -2.4000],
           [-2.4000, -3.6000]],
 
         [[-2.4000, -1.2000],
          [-1.2000, -1.2000]],
 
         [[-4.8000, -2.4000],
           [-3.6000, -3.6000]]]]
    x2 = torch.tensor(raw_x2, dtype=torch.float)
    x2_scale = 1.2
    x2_zero_point = 4
    x2_q = torch.quantize_per_tensor(x2, scale=x2_scale, zero_point=x2_zero_point,
            dtype=torch.quint8)
    #x2_q = torch._make_per_tensor_quantized_tensor(x2, x2_scale, x2_zero_point)
    x2 = x2_q.dequantize()

    y_scale = 4.2
    y_zero_point = 0 

    input_channels = 4
    output_channels = 4
    kernels = (3, 3)
    strides = (2, 2)
    pads = (1, 1)
    dilations = (1, 1)
    groups = 2

    qconv_inductor = torch.ops.quantized.conv_add_int8_packed_weight
    qconv_prepack_inductor = torch.ops.quantized.conv_prepack_cpu_tensor
    conv_op = torch.nn.Conv2d(
        input_channels,
        output_channels,
        kernels,
        strides,
        pads,
        dilations,
        groups,
    )

    conv_op.weight = torch.nn.Parameter(w, requires_grad=False)
    conv_op.bias = None
    result_ref = conv_op(x)
    result_ref = result_ref + x2
    result_ref_q = torch.quantize_per_tensor(
        result_ref, scale=y_scale, zero_point=y_zero_point,
        dtype=torch.quint8)

    # Step2
    x_q_inductor = x_q.int_repr()
    w_q_inductor = w_q.int_repr()

    weight_scale = torch.tensor(w_q.q_scale(), dtype=torch.float)
    weight_zero_point = torch.tensor(w_q.q_zero_point(), dtype=torch.int64)
    packed_weight, packed_bias = torch.ops.quantized.conv_prepack_cpu_tensor(
        w_q_inductor, weight_scale,
        x_q_inductor.size(), x_scale, x_zero_point,
        None, strides, pads, dilations, groups)

    x2_q_inductor = x2_q.int_repr()
    y_q_inductor = torch.ops.quantized.conv_add_int8_packed_weight(
        x_q_inductor, x_scale, x_zero_point,
        x2_q_inductor, x2_scale, x2_zero_point,
        packed_weight, weight_scale, weight_zero_point,
        packed_bias, strides, pads, dilations, groups,
        y_scale, y_zero_point
    )

    print("x_q_inductor: {}".format(x_q_inductor.cpu().numpy()), flush=True)
    print("w_q_inductor: {}".format(w_q_inductor.cpu().numpy()), flush=True)
    print("result_ref_q is: {}".format(result_ref_q.int_repr().cpu().numpy()), flush=True)
    print("y_q_inductor is: {}".format(y_q_inductor.cpu().numpy()), flush=True)
    print("x2_q_inductor is: {}".format(x2_q_inductor.cpu().numpy()), flush=True)

    import numpy as np
    np.testing.assert_array_almost_equal(
        #result_ref_q.int_repr().cpu().numpy(), y_q_inductor.cpu().numpy(), decimal=0,
        0, 255, decimal=0,
        err_msg=f'''X: {x_q}, W: {w_q}, b: {None}, strides: {strides},
        pads: {pads}, o_pads: {None}, dilations: {dilations},
        groups: {groups}, y_s: {y_scale}, y_zp: {y_zero_point}, X2: {x2_q}''')

if __name__ == "__main__":
    torch.backends.quantized.engine = 'x86'
    test()
