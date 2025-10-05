# Ackowledgements
You can check Terminal Prompts too but this has more info ;)

Pretained models, datasets have been borrowed from the original paper but our implementation is original
Base paper:https://arxiv.org/abs/2103.17249

To start, run these commands in terminal

cd StyleCLIP (if you haven't changed the directory)
pip install -r requirements.txt

# Training Our Mapper
To train the model, we have implemented two methods

1. Static mapper gates: train_1.py (was used to train Beyonce, Hillary clinton models)
2. Dynamic mapper gates: train_2.py (was used to train annoyed, surprised models)

Path info: 
Train data is in /StyleCLIP/data
Stylegan weights are in /StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt
ir_se50_weights are in /StyleCLIP/pretrained_models/model_ir_se50.pth

Change the parameters as you like from the args parser in the train files
The entire code from architecture to train is train_1.py (Method 1) and  train_2.py (Method 2)

Terminal prompts:

*Method 1
python train_1.py --prompt "Hillary Clinton" 

*Method 2
python train_2.py --prompt "Surprised" 

If the paths creates issues pls change the paths in args parse as per your system. We have placed exact models in those folders mentioned in Path Info

The trained results will be saved in train_1_results and train_2_results folder respectively

# Running Inference with Pretrained Mappers

For Inference, we used Celebrity data (named as celebrity_data in the StyleCLIP folder)

For model Checkpoints, please check Best_Checkpoints folder, we have 4 .pt files
path: /StyleCLIP/Best_Checkpoints/Suprised.pt
Please use inference_1.py to check Hillary Clinton, Beyonce's results and use inference_2.py to check surprised and annoyed results

You have to give the prompt that was used to train the model:
prompt for 'Beyonce_Best.pt' = "Beyonce Knowles"
prompt for 'Clinton_Best.pt' = "Hillary Clinton"
prompt for 'Annoyed_Best.pt' = "Annoyed face"
prompt for 'Surprised.pt'    = "Surprised face"

Change the prompt option in arg parser

We have kept the paths we used but just in case, Path info (Paths are in args parse in the code):
Saved models: /StyleCLIP/Best_Checkpoints/Suprised.pt
latents for celebrity data/w_plus: /StyleCLIP/celebrity_data/w_plus.npy
stylegan weights: /StyleCLIP/pretrained_models/stylegan2-ffhq-config-f.pt
ir_se50_weights: /StyleCLIP/pretrained_models/model_ir_se50.pth

final terminal prompt:
python inference_1.py  
python inference_2.py

Right now the default is to check 'Hillary Clinton' results from inference_1.py and 'Surprised face' results in inference_2.py

The inference results are saved in inference_1_results, inference_2_results respectively










