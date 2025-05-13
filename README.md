# ST-LLM+ : Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction
This repository provides the official Pytorch implementation of our manuscript titled "ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction". This work is an extension of the original [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM/blob/main/ST-LLM.pdf) model. The foundational training framework is derived from the open-source codebase developed by [ChenxiLiu-HNU](https://github.com/ChenxiLiu-HNU/ST-LLM/tree/main).

## Abstract
> *Traffic prediction is a crucial component of data management systems, leveraging historical data to learn spatio-temporal dynamics for forecasting future traffic and enabling efficient decision-making and resource allocation. Despite efforts to develop increasingly complex architectures, existing traffic prediction models often struggle to generalize across diverse datasets and contexts, limiting their adaptability in real-world applications. In contrast to existing traffic prediction models, large language models (LLMs) progress mainly through parameter expansion and extensive pre-training while maintaining their fundamental structures. In this paper, we propose ST-LLM+, the graph enhanced spatio-temporal large language models for traffic prediction. Through incorporating a proximity-based adjacency matrix derived from the traffic network into the calibrated LLMs, ST-LLM+ captures complex spatio-temporal dependencies within the traffic network. The Partially Frozen Graph Attention (PFGA) module is designed to retain global dependencies learned during LLMs pre-training while modeling localized dependencies specific to the traffic domain. To reduce computational overhead, ST-LLM+ adopts the LoRA-augmented training strategy, allowing attention layers to be fine-tuned with fewer learnable parameters.
Comprehensive experiments on real-world traffic datasets demonstrate that ST-LLM+ outperforms state-of-the-art models. In particular, ST-LLM+ also exhibits robust performance in both few-shot and zero-shot prediction scenarios. Additionally, our case study demonstrates that ST-LLM+ captures global and localized dependencies between stations, verifying its effectiveness for traffic prediction tasks.*

<img width="1098" alt="image" src="https://github.com/ChenxiLiu-HNU/ST-LLM/assets/46647878/15bf40a4-333f-42ed-a241-32432a5484ce">

## Dependencies

* Python 3.8.19
* PyTorch 2.4.1
* cuda 11.7
* torchvision 0.19.1

```bash
> conda env create -f env_ubuntu.yaml
```

## Datasets
We provide preprocessed datasets, which you can access [here](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link).   
If you need the original datasets, please refer to the [ESG](https://github.com/LiuZH-19/ESG).

## Training

```bash
CUDA_VISIBLE_DEVICES=0
nohup python train.py --data taxi_pick > your_log_name.log &
```

## BibTex
> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:
```bibtex
@inproceedings{liu2024spatial,
  title={Spatial-Temporal Large Language Model for Traffic Prediction},
  author={Liu, Chenxi and Yang, Sun and Xu, Qianxiong and Li, Zhishuai and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={MDM},
  year={2024}
}
```

## Further Reading
[**TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment**](https://arxiv.org/abs/2406.01638), in *AAAI* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeCMA)

**Authors**: Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

[**Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation**](https://arxiv.org/abs/2505.02138), in *ICDE* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeKD)

**Authors**: Chenxi Liu, Hao Miao, Qianxiong Xu, Shaowen Zhou, Cheng Long, Yan Zhao, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2025timekd,
  title={Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation},
  author={Chenxi Liu and Hao Miao and Qianxiong Xu and Shaowen Zhou and Cheng Long and Yan Zhao and Ziyue Li and Rui Zhao},
  booktitle    = {ICDE},
  year={2025}
}
```

## Acknowledgement
Our implementation adapts [OFA](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and has extensively modified it for our purposes. We are grateful to the authors for providing their implementations and related resources.

## Contact Us
For inquiries or further assistance, contact us at [chenxi.liu@ntu.edu.sg](mailto:chenxi.liu@ntu.edu.sg) or open an issue on this repository.
