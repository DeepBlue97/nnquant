sudo chmod 777 /root -R

conda create -n nnquant python=3.10
conda activate nnquant
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


vai_c_xir -x quantize_result/Net_int.xmodel  -a quant/arch_3eg_vai3.0.json -o . -n Mnist
vai_c_xir -x quantize_result/Net_int.xmodel  -a 02_quanted_model/arch_3eg_vai3.0.json -o . -n Mnist

xir subgraph ./Mnist.xmodel | grep DPU

