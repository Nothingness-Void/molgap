# Terminal decision - CPU execution probe

Outcome: NO_TRAIN. The seven-case scratch probe completed; the worker failed
after atomic checkpoint save when publication metadata called `st_size()` instead
of reading the integer property. No checkpoint-resume result or completion receipt
was produced. Do not relabel the overall worker COMPLETE.

Loss, full pre/post-clip gradients, parameters, BN buffers, optimizer tensors and
RNG match exactly for the optional layout and unchanged-optimizer cases over three
scratch steps. Foreach is exact in this CPU sample but changes optimizer contract;
fused has a maximum observed tensor difference2.384185791015625e-7. Neither optimizer
implementation is promoted. Four balanced timing windows have mixed signs and do
not establish a stable speed gain. No GPU speed measurement or MAE claim.

The complete timing suite will not be repeated to repair a metadata receipt.
A separately prospective checkpoint-only recovery may qualify serialization and
continued CPU execution. Native GPU/T4 qualification remains absent. No new remote
training, production change or full gate is released. Recommendation unchanged.

Git disposition: accepted diagnostic evidence and reviewed reusable implementation
are eligible for the non-promotion desktop route; not an adopted model.
