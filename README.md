# PixelLM
#### Zhongwei Ren\*, Zhicheng Huang\*, Yunchao Wei<sup>&dagger;</sup>, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin\*<sup>&dagger;</sup><sup>&ddagger;</sup>
\* Equally contributing first authors, <sup>&dagger;</sup>Correspondence, <sup>&ddagger;</sup>Project Lead

Beijing Jiaotong University, University of Science and Technology Beijing, ByteDance

While large multimodal models (LMMs) have achieved remarkable progress, generating pixel-level masks for image reasoning tasks involving multiple open-world targets remains a challenge. To bridge this gap, we introduce PixelLM, an effective and efficient LMM for pixel-level reasoning and understanding. Central to PixelLM is a novel, lightweight pixel decoder and a comprehensive segmentation codebook. The decoder efficiently produces masks from the hidden embeddings of the codebook tokens, which encode detailed target-relevant information. With this design, PixelLM harmonizes with the structure of popular LMMs and avoids the need for additional costly segmentation models. Furthermore, we propose a target refinement loss to enhance the model's ability to differentiate between multiple targets, leading to substantially improved mask quality. To advance research in this area, we construct MUSE, a high-quality multi-target reasoning segmentation benchmark. PixelLM excels across various pixel-level image reasoning and understanding tasks, outperforming well-established methods in multiple benchmarks, including MUSE, single-
and multi-referring segmentation. Comprehensive ablations confirm the efficacy of each proposed component. **Our code, model and data are under preparation, please stay tuned!**


