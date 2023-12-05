# PixelLM
> #### Zhongwei Ren\*, Zhicheng Huang\*, Yunchao Wei<sup>&dagger;</sup>, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin\*<sup>&dagger;</sup><sup>&ddagger;</sup>
> \* Equally contributing first authors, <sup>&dagger;</sup>Correspondence, <sup>&ddagger;</sup>Project Lead

> Beijing Jiaotong University, University of Science and Technology Beijing, ByteDance


<img width="1000" alt="image" src='fig/results_show_v5.png'>

# Highlight

1. We present PixelLM, a novel LMM for pixel-level reasoning and understanding. PixelLM proficiently handles tasks with an arbitrary number of open-set targets and diverse reasoning complexities. Its design maintains the fundamental structure of LMMs while avoiding additional, costly segmentation models, enhancing both efficiency and transferability to diverse applications. 

2. We construct MUSE, a high-quality multi-target reasoning segmentation dataset, facilitating model training and evaluation in future research. Utilizing a GPT-4V-aided data curation pipeline, we create 246k question-answer pairs, covering 0.9 million instances. Our extensive ablation studies confirm the dataset's effectiveness in stimulating the model’s pixel reasoning capabilities.

3. PixelLM achieves new state-of-the-art results across a spectrum of benchmarks, significantly surpassing competing methods.

**Our code, model and data are under preparation, please stay tuned!**


# Introduction
While large multimodal models (LMMs) have achieved remarkable progress, generating pixel-level masks for image reasoning tasks involving multiple open-world targets remains a challenge. To bridge this gap, we introduce PixelLM, an effective and efficient LMM for pixel-level reasoning and understanding. Central to PixelLM is a novel, lightweight pixel decoder and a comprehensive segmentation codebook. The decoder efficiently produces masks from the hidden embeddings of the codebook tokens, which encode detailed target-relevant information. With this design, PixelLM harmonizes with the structure of popular LMMs and avoids the need for additional costly segmentation models. Furthermore, we propose a target refinement loss to enhance the model's ability to differentiate between multiple targets, leading to substantially improved mask quality. To advance research in this area, we construct MUSE, a high-quality multi-target reasoning segmentation benchmark. PixelLM excels across various pixel-level image reasoning and understanding tasks, outperforming well-established methods in multiple benchmarks, including MUSE, single-
and multi-referring segmentation. Comprehensive ablations confirm the efficacy of each proposed component.

# Demo Video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/sw2co_xaqPA/0.jpg)](https://www.youtube.com/watch?v=sw2co_xaqPA "PixelLM Demo")


# Architecture

<img width="1000" alt="image" src='fig/overallV3.png'>

PixelLM features a streamlined architecture, comprising four main parts: i) a pretrained CLIP-ViT vision encoder 
 which aligns with text, ii) a large language model, iii) a lightweight pixel decoder 
 and iv) a segmentation codebook. PixelLM processes image and query text, yielding interleaved text description and corresponding masks for varied target. At the core of PixelLM is the novel lightweight decoder and the holistic segmentation codebook. The codebook contains learnable tokens which encode contexts and knowledge pertinent to targets referencing at different visual scales. The pixel decoder then produces target masks based on the hidden embeddings from the codebook tokens in conjunction with image features. Thanks to this design, PixelLM can generate high-quality masks without external segmentation models, significantly boosting its efficiency. Furthermore, we propose a target refinement loss to enhance the model's capability of differentiating between multiple targets, thus further improving the mask quality.


 # Architecture

<img width="1000" alt="image" src='fig/data_example.png'>

To facilitate model training and evaluation in this area of research, we develop MUSE, the first comprehensive multi-target reasoning segmentation dataset. MUSE stands out with its open-set concepts, detailed object descriptions, complex multi-target question-answer pairs, and instance-level mask annotations. Specifically, we feed all the instance category names and corresponding bounding box coordinates in the image to GPT-4V. Using carefully crafted prompts, GPT-4V autonomously selects instances to construct question-answer pairs relevant to the image content. The left panel of the figure above illustrates the prompt employed in our GPT-4V data generation pipeline. The right panel showcases an example of the generated data.