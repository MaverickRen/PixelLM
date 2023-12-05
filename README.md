# PixelLM
#### Zhongwei Ren\*, Zhicheng Huang\*, Yunchao Wei<sup>&dagger;</sup>, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin\*<sup>&dagger;</sup><sup>&ddagger;</sup>
\* Equally contributing first authors, <sup>&dagger;</sup>Correspondence, <sup>&ddagger;</sup>Project Lead

Beijing Jiaotong University, University of Science and Technology Beijing, ByteDance

We present PixelLM, a novel LMM for pixel-level reasoning and understanding. PixelLM proficiently handles tasks with an arbitrary number of open-set targets and diverse reasoning complexities. Its design maintains the fundamental structure of LMMs while avoiding additional, costly segmentation models, enhancing both efficiency and transferability to diverse applications. 

We construct MUSE, a high-quality multi-target reasoning segmentation dataset, facilitating model training and evaluation in future research. Utilizing a GPT-4V-aided data curation pipeline, we create 246k question-answer pairs, covering 0.9 million instances. Our extensive ablation studies confirm the dataset's effectiveness in stimulating the model’s pixel reasoning capabilities.

PixelLM achieves new state-of-the-art results across a spectrum of benchmarks, significantly surpassing competing methods.

**Our code, model and data are under preparation, please stay tuned!**


<img width="1000" alt="image" src='fig/results_show_v5.png'>

# Introduction
While large multimodal models (LMMs) have achieved remarkable progress, generating pixel-level masks for image reasoning tasks involving multiple open-world targets remains a challenge. To bridge this gap, we introduce PixelLM, an effective and efficient LMM for pixel-level reasoning and understanding. Central to PixelLM is a novel, lightweight pixel decoder and a comprehensive segmentation codebook. The decoder efficiently produces masks from the hidden embeddings of the codebook tokens, which encode detailed target-relevant information. With this design, PixelLM harmonizes with the structure of popular LMMs and avoids the need for additional costly segmentation models. Furthermore, we propose a target refinement loss to enhance the model's ability to differentiate between multiple targets, leading to substantially improved mask quality. To advance research in this area, we construct MUSE, a high-quality multi-target reasoning segmentation benchmark. PixelLM excels across various pixel-level image reasoning and understanding tasks, outperforming well-established methods in multiple benchmarks, including MUSE, single-
and multi-referring segmentation. Comprehensive ablations confirm the efficacy of each proposed component.


<video width="1000" controls>
  <source src="video/PixelLM_show_V2.mp4" type="video/mp4">
</video>
