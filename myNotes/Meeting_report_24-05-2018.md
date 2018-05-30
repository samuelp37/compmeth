# Report of the meeting of the 24/05/2018

## 1. General infos

- The information bits (not encoded original message) are sent to the device in order to calculate the bit error rate.
- The conversion inside the GPU seems to be feasible. It would increase the throughput and decrease the latency.
- Currently, running the program takes too long. The parameters should be changed to reduce the execution time : MIN_CODEWORD and MIN_FER divided by 1000, SNR lowered.
Then, we have to generate numbers similar to what is in the documentation and test it on different machines.

As all the groups, we will have to make a presentation of our project to the class.

## 2. Main issues to resolve

- Make sure that we can compile and run the LDPC code in Windows and Linux environments.
- Try to make the conversion of floats in the device.
- Look at the steps done on the host and the ones done on the GPU : where is the border ?
- Make measurements.


