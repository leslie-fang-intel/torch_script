import torch
import intel_extension_for_pytorch as ipex

CodeFreeAutocastEnabled = False

def set_optimized_attr(model):
    model.optimized = True
    for child_name, child in model.named_children():
        set_optimized_attr(child)

def forward_pre_hook(m, input):

    def check_function(m):
        return (not hasattr(m, "optimized")) and \
            (isinstance(m, torch.nn.Module) and not (isinstance(m, torch.jit._script.ScriptModule)) and not (isinstance(m, torch.fx.GraphModule)))

    #if not hasattr(m, "optimized"):
    if check_function(m):
        print("----inside forward_pre_hook----", flush=True)
        # Step1:
        # Symbolic trace inside ipex.optimize will invoke forward method of submodules
        # Add optimized attr to all the submodule to avoid recursion invoking of ipex.optimize in submodules
        set_optimized_attr(m)

        # Step2:
        # Invoke optimize to generate new optimized module
        # Use the default parameters of optimize
        dataType = torch.bfloat16 if (CodeFreeAutocastEnabled == True) else torch.float32
        optimized_m = ipex.optimize(m.eval(), dtype=dataType).eval()


        import warnings
        from typing import List
        def graph_capture(model, train = False):
            r"""

            Args:
                model
            train (bool): Whether to train model or not. Default value is ``False``

            Returns:
                The wrapped model, that will generate graph on first run.
            """
            class M(torch.nn.Module):
                def __init__(self):
                    super(M, self).__init__()
                    self.model = model
                    self.tried_dynamo = False
                def forward(self, input):
                    if isinstance(self.model, torch.jit.ScriptModule) or self.tried_dynamo == True:
                        # the model is already a script module, or has tried TorchDynamo.
                        return self.model(input)

                    else:
                        try:
                            ## ##TODO## uncomment this line if you like to try Dynamo Path
                            # raise Exception("not use jit trace")

                            # try JIT trace.
                            warnings.filterwarnings('error')
                            with torch.no_grad():
                                ## ##TODO## <--Add Autocast for BF16 of graph capture API-->
                                ## Not needed for code free, since Autocast is already in the outside: optimized_m_forward
                                # with torch.cpu.amp.autocast(enabled=CodeFreeAutocastEnabled):
                                traced_model = torch.jit.trace(self.model.eval(), input).eval()
                                freeze_model = torch.jit.freeze(traced_model)

                                ## ##TODO## <--Add Code Free for graph capture API-->
                                #set_optimized_attr(freeze_model)

                                output = freeze_model(input)
                                self.model = freeze_model
                                return output

                        except:
                            try:
                                # JIT trace failed, try torchdynamo with JIT trace backend
                                self.tried_dynamo = True

                                def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

                                    ## ##TODO## <--Add Code Free for graph capture API-->
                                    #set_optimized_attr(gm)

                                    ## ##TODO## <--Add Autocast for BF16 of graph capture API-->
                                    ## Not needed for code free, since Autocast is already in the outside: optimized_m_forward
                                    # with torch.cpu.amp.autocast(enabled=CodeFreeAutocastEnabled):
                                    traced_gm = torch.jit.trace(gm.eval(), example_inputs).eval()
                                    freeze_gm = torch.jit.freeze(traced_gm)

                                    ## ##TODO## <--Add Code Free for graph capture API-->
                                    #set_optimized_attr(freeze_gm)
                                    return freeze_gm

                                class DynamoM(torch.nn.Module):
                                    def __init__(_self):
                                        super(DynamoM, _self).__init__()
                                        _self.model = self.model
                                    def forward(_self, x):
                                        import torchdynamo
                                        ## ##TODO## <--Add Autocast for BF16 of graph capture API-->
                                        ## Not needed for code free, since Autocast is already in the outside: optimized_m_forward
                                        # with torch.cpu.amp.autocast(enabled=CodeFreeAutocastEnabled):
                                        with torchdynamo.optimize(compiler), torch.no_grad():
                                            y = _self.model(x)
                                        return y

                                dynamo_model = DynamoM()

                                ## ##TODO## <--Add Code Free for graph capture API-->
                                set_optimized_attr(dynamo_model)

                                output = dynamo_model(input)
                                self.model = dynamo_model
                                return output

                            except:
                                print("Both JIT and TorchDynamo failed, return original model.")
                                return self.model(input)
            return M()

        optimized_m_graph_capture = graph_capture(optimized_m)

        # Set optimized attr for the new optimized module
        set_optimized_attr(optimized_m_graph_capture)

        # Step3:
        # Substitue the forward method of m with forward method of new optimized module
        def optimized_m_forward(*args, **kwargs):
            with torch.cpu.amp.autocast(enabled=CodeFreeAutocastEnabled), torch.no_grad():
                return optimized_m_graph_capture(*args, **kwargs)
        m.forward = optimized_m_forward

    return input

handle = torch.nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)

import torch
import torch.fx.experimental.optimization as optimization
import torchvision.models as models
import time
import argparse

"""
export LD_PRELOAD="/pytorch/leslie/jemalloc/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=$LD_PRELOAD:/pytorch/leslie/anaconda3/pkgs/intel-openmp-2021.4.0-h06a4308_3561/lib/libiomp5.so
KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python baseline.py --datatype int8
"""
parser = argparse.ArgumentParser(description='AI everywhere experiments')
parser.add_argument('--datatype', default='int8', help='path to dataset')

if __name__ == "__main__":
    args = parser.parse_args()

    if args.datatype == "bf16":
        CodeFreeAutocastEnabled = True

    iteration = 100
    batch_size = 56

    print("datatype is:{}".format(args.datatype))
    print("batch_size is:{}".format(batch_size))

    model = models.__dict__["resnet50"](pretrained=True)
    if args.datatype == "fp32" or args.datatype == "bf16":
        model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    if args.datatype == "fp32":
        # model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        # model = ipex.optimize(model, dtype=torch.float32, auto_kernel_selection=True, inplace=True)
        # traced_model = torch.jit.trace(model, x).eval()
        # traced_model = torch.jit.freeze(traced_model)
        # traced_model = model
        pass
    elif args.datatype == "bf16":
        # model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        # x = x.to(torch.bfloat16)
        # with torch.cpu.amp.autocast(), torch.no_grad():
        #     traced_model = torch.jit.trace(model, x).eval()
        # traced_model = torch.jit.freeze(traced_model)
        # traced_model = model
        pass
    else:
        print("unsupported data type", flush=True)
        exit(-1)

    with torch.no_grad():
        # warm up
        for i in range(3):
            print("----inside warm up step:{}----".format(i), flush=True)
            model(x)
        print("----finish warm up step----", flush=True)

        start = time.time()
        for i in range(iteration):
            if i == 29:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_log")) as prof:
                    model(x)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                model(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))
