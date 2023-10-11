import torch
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
aten = torch.ops.aten

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

def correct_fn(input_ids, pad_token_id, logits, batch_size, sequence_lengths):
    sequence_lengths = (torch.ne(input_ids, pad_token_id).sum(-1) - 1).to("cpu")
    pooled_logits = logits[torch.arange(batch_size, device="cpu"), sequence_lengths]
    return pooled_logits

def failed_fn(input_ids, pad_token_id, logits, batch_size, sequence_lengths):
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(
        "cpu"
    )
    pooled_logits = logits[torch.arange(batch_size, device="cpu"), sequence_lengths]
    return pooled_logits

pad_token_id = 0
input_ids = torch.load("./input_ids.pt")
logits = torch.load("./logits.pt")
batch_size = 1
sequence_lengths = torch.tensor([-1])

# correct1 = correct_fn(input_ids, pad_token_id, logits, batch_size, sequence_lengths)
# compiled1 = torch._dynamo.optimize_assert(compile_fx)(correct_fn)(input_ids, pad_token_id, logits, batch_size, sequence_lengths)
# print("correct1 is: {}".format(correct1), flush=True)
# print("compiled1 is: {}".format(compiled1), flush=True)

correct2 = failed_fn(input_ids, pad_token_id, logits, batch_size, sequence_lengths)
compiled2 = torch._dynamo.optimize_assert(compile_fx)(failed_fn)(input_ids, pad_token_id, logits, batch_size, sequence_lengths)
print("correct2 is: {}".format(correct2), flush=True)
print("compiled2 is: {}".format(compiled2), flush=True)
