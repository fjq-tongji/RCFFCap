# RGFRCap: Enhancing Image Captioning with Retrieval-Guided Semantic Feature Refinement


## Project Overview
1. Submitted to The Visual Computer
2. Our framework utilizes the MSCOCO-2014, Flickr30K, City_Cap, and FoggyCity_Cap dataset for model training and testing, aiming to develop image captioning capabilities, enabling the generation of descriptive text that accurately corresponds to input images.
3. 


## Contributions
1. We propose a novel retrieval-guided semantic feature refinement network (RGFRCap) for image caption generation, covering both conventional environments and traffic scenarios. In our framework, the image-text retrieval (ITR) mechanism provides prior cues that help refine semantic knowledge, enabling the model to filter out irrelevant information.
2. Guided by the retrieval results of the ITR module, heterogeneous semantic representations from detection and segmentation modules are selectively filtered. The final retained information is further aggregated with region-level visual features in the VSF module, obtaining enriched inputs that can enhance the reasoning capability of the language decoder.
3. Extensive experiments on the MSCOCO, Flickr30K, and two self-constructed traffic captioning datasets demonstrate the effectiveness of RGFRCap in semantic scene understanding and caption generation, achieving state-of-the-art performance on certain evaluation metrics.

## Usage

### Start training
This project uses Python 3.10, PyTorch 1.11.0, and CUDA 11.3.
 
> CUDA_VISIBLE_DEVICES=0 sh train_coco_Large.sh

See `opts.py` for the options, and you can enlarge `--max_epochs` in `train_coco_Large.sh` to train the model for more epochs.

> CUDA_VISIBLE_DEVICES=0 sh train_SCST_coco-Large.sh

After training under the cross-entropy loss, another 15 epochs needs to be trained under SCST loss.



### Start evaluation
> CUDA_VISIBLE_DEVICES=0 sh eval_karpathy_test_coco.sh

Here, you can set `--infos_path`, `--model`, `--save_path_seq`, and `--save_path_loss_index`.

