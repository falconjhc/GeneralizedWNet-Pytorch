# GeneralizedWNet-Pytorch
This is the implementation of the Generalised WNet. In particular, it is a transplanted repository of source code from https://github.com/falconjhc/GeneralizedWNet-Tensorflow1.x


To train:

# try a plain wnet
cd Scripts
python PipelineScripts.py --wnet plain   --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 11 --resumeTrain 1 --config PF64-PF50  --device 0

# try a generalized wnet with only basic resblocks
python PipelineScripts.py --wnet general  --encoder EncoderCvCvCvCv  --decoder DecoderCvCvCvCv  --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50  --device 0

# try a generalized wnet with only bottleneck resblocks
python PipelineScripts.py --wnet general  --encoder EncoderCbnCbnCbnCbn  --decoder DecoderCbnCbnCbnCbn  --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50  --device 0


# try a generalized wnet with Vision Transformers
python PipelineScripts.py --wnet general  --encoder EncoderCbnCbnCbnVit@2@24  --decoder DecoderVit@2@24CbnCbnCbn  --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50  --device 0


add --debug 1 when you want to perform debug. eliminate it when you run the code. 

It is still in construction so updates are expected from time to time.

Enjoy.