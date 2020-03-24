This is the code for the combination of our CSDL method and iSQRT-COV (named as "Ours" in Table 1 in our paper). This code is modified from iSQRT-COV.

To run this CSDL-iSQRT-COV code, 
- put data in ``data`` folder (data could be downloaded/formated as instructed in ../README.md)
- modify finetune.sh (more specifically, line 41 and 44) with the proper dataset name and proper number of classes. 

Once the modification is done, just simply run 
```
CUDA_VISIBLE_DEVICES=0 bash finetune.sh
```