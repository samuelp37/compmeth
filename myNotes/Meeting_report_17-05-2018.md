# Report of the meeting of the 17/05/2018

## 1. General infos

From modem to decoder there should be 8*270 Mbits/s of throughput (transfer of the LLRs (8 bits per LLR)).
From decoder to modem there should be 280 Mbits/s (if we transfer the information bits and not the LLRs, 8*280 Mbits/s otherwise).

## 2. Main issues to resolve

- Where can the conversion from fixed point to floating point occur ? On the CPU ? On the GPU ?
- Is it single or double precision floating point unit on GPUs ?
- Can we run and benchmark the LDPC inplementation of the cuLDPC repository ?
