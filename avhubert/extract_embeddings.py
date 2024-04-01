import fairseq
import hubert_pretraining, hubert
import json
import utils as custom_utils
import numpy as np
import torch
import argparse
import os
import glob
import sys
import tqdm

DEVICE="cuda:2"


# ckpt_path = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt"
ckpt_path= "large_vox_iter5.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
model = model.eval()
model = model.to(DEVICE)


def proc_one_image(image_fp:str,out_fp:str):
    im_input = custom_utils.load_image(path=image_fp,
                                   task_cfg=task.cfg)
    with torch.no_grad():
        model_input = {
            "audio": torch.from_numpy(np.zeros( (1,104) ).astype(np.float32)).unsqueeze(0).transpose(1, 2),
            "video": im_input.unsqueeze(0).permute((0, 4, 1, 2, 3)).contiguous(),
        }
        model_input = {k: v.to(DEVICE) for k,v in model_input.items()}
        embeddings, _  = model.extract_features(
            source= model_input
        )
        # embeddings: (bsz, T, F)
        embeddings = embeddings.squeeze()
        assert len(embeddings.shape) == 1
        embeddings = embeddings.cpu()
    
    torch.save(embeddings,out_fp)

def proc_one_audio(audio_fp:str,out_fp:str):
    audio_input = custom_utils.load_audio(path=audio_fp,
                                   task_cfg=task.cfg)
    with torch.no_grad():
        model_input = {
            "audio": audio_input.unsqueeze(0).transpose(1, 2),
            "video": torch.from_numpy(np.zeros( (audio_input.shape[0],task.cfg.image_crop_size,task.cfg.image_crop_size, 1  ) ).astype(np.float32)).unsqueeze(0).permute((0, 4, 1, 2, 3)).contiguous(),
        }
        model_input = {k: v.to(DEVICE) for k,v in model_input.items()}
        embeddings, _  = model.extract_features(
            source= model_input
        )
        # embeddings: (bsz, T, F)
        # print(embeddings.shape)
        # temporal pooling
        embeddings = torch.mean(embeddings,dim=1)
        embeddings = embeddings[0]
        assert len(embeddings.shape) == 1
        embeddings = embeddings.cpu()
    torch.save(embeddings,out_fp)
    
def list_all_files_endswith(directory,extension):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                jpg_files.append(os.path.join(root, file))
    return jpg_files



# proc_one_image(image_fp="/saltpool0/data/tseng/SoundSymbolism/generated_images/round/circular/circular-05602614ad2946d88e2c8f0ab3b52ff8.jpg")
# proc_one_audio(audio_fp="/saltpool0/data/yjshih/SoundSymbolism/generated_sounds/wav/bababa-en-US-Neural2-D.wav")
# im_input = custom_utils.load_image(path="/saltpool0/data/tseng/SoundSymbolism/generated_images/round/circular/circular-05602614ad2946d88e2c8f0ab3b52ff8.jpg",
#                                    task_cfg=task.cfg)
# audio_input = custom_utils.load_audio(path="/saltpool0/data/yjshih/SoundSymbolism/generated_sounds/wav/bababa-en-US-Neural2-D.wav",
#                                       task_cfg=task.cfg)


# with torch.no_grad():
    # model_input = {
    #     "audio": audio_input.unsqueeze(0).transpose(1, 2),
    #     "video": im_input.unsqueeze(0).permute((0, 4, 1, 2, 3)).contiguous(),
    # }
    # model_output = model.forward(
    #     source= model_input,
    #     mask = False,
    #     features_only = True,
    # )

    # embeddings, _  = model.extract_features(
    #     source= model_input
    # )

    # print(embeddings.shape)

# print(type(model))

def main(args):
    print("[AV HubERT] - Get audio embeddings")
    print("args:",args)

    # get all images
    all_images = sorted(list_all_files_endswith(args["input_image_dir"],".jpg"))
    # get all wavs
    all_wavs = sorted(list_all_files_endswith(args["input_audio_dir"],".wav"))

    print(f"[AV HubERT] - Get #{len(all_wavs)} audio  and #{len(all_images)} images")
    os.makedirs(args["output_emb_dir"],exist_ok=True)
    print(f"[AV HubERT] - dump embs to {args['output_emb_dir']}")
    os.makedirs( os.path.join(args["output_emb_dir"],"image"),exist_ok=True)
    os.makedirs( os.path.join(args["output_emb_dir"],"audio"),exist_ok=True)

    # for image_fp in tqdm.tqdm(all_images,"Processing images"):
    #     output_fp = os.path.join(args["output_emb_dir"],"image",os.path.basename(image_fp))
    #     proc_one_image(
    #         image_fp=image_fp,
    #         out_fp=output_fp
    #     )
    #     # print(image_fp,output_fp)
    #     # exit(0)
    
    for audio_fp in tqdm.tqdm(all_wavs,"Processing audio"):
        output_fp = os.path.join(args["output_emb_dir"],"audio",os.path.basename(audio_fp))
        # print(audio_fp,output_fp)
        proc_one_audio(
            audio_fp=audio_fp,
            out_fp=output_fp
        )
        # exit(0)





if __name__ == "__main__":
    # if not len(sys.argv) == 3:
    #     print("[AV HubERT] - Usage: python extract_embeddings.py <input_audio_dir> <input_image_dir> <output_emb_dir>")

    args = {
        "input_audio_dir": "/saltpool0/data/yjshih/SoundSymbolism/generated_sounds/wav",
        "input_image_dir": "/saltpool0/data/tseng/SoundSymbolism/generated_images",
        "output_emb_dir":"/saltpool0/data/yjshih/SoundSymbolism/model_embs/avhubert/large_vox_iter5"
    }
    main(args)

