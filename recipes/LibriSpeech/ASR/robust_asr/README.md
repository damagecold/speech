# 方案B: WavLM-base + Conformer + 课程对抗 + 特征级对抗

## 任务
噪声鲁棒 ASR 发论文（C 类会议）

## 方法
- **主干网络**: WavLM-base + Conformer Encoder + GRU Decoder
- **损失函数**: CTC + CrossEntropy (label smoothing)
- **两个创新点**:
  1. **课程对抗训练**: 高 SNR(20dB) → 中 SNR(10dB) → 低 SNR(0dB) 渐进学习
  2. **特征级对抗**: 对 WavLM 中间层特征加 FGSM 扰动

## 数据集
| 数据集 | 用途 |
|--------|------|
| train-clean-100 | 训练 |
| dev-clean | 验证 |
| test-clean | 基线测试 |
| test-other | 鲁棒性测试 |
| DEMAND | 合成噪声 |

## 框架
SpeechBrain

## 运行方法

### 1. 解压数据
```bash
cd /e/vscode/data/LibriSpeech/LibriSpeech
# 解压 train-clean-100, dev-clean, test-clean
```

### 2. 安装依赖
```bash
pip install speechbrain transformers sentencepiece
```

### 3. 运行训练
```bash
cd /e/vscode/speechbrain
python recipes/LibriSpeech/ASR/robust_asr/train.py \
    recipes/LibriSpeech/ASR/robust_asr/hparams/train_wavlm_base_conformer.yaml \
    --data_folder /e/vscode/data/LibriSpeech/LibriSpeech
```

## 超参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `curriculum_adversarial` | 启用课程对抗训练 | True |
| `curriculum_epochs` | 各阶段切换的 epoch | [10, 20] |
| `curriculum_snrs` | 各阶段 SNR (dB) | [20, 10, 0] |
| `adversarial_weight` | 对抗 loss 权重 | 0.1 |
| `feature_adversarial` | 启用特征级对抗 | True |
| `adversarial_epsilon` | FGSM 扰动大小 | 0.01 |
| `freeze_wav2vec` | 是否冻结 WavLM | True |

## 实验设计

### 基线对比
1. **Baseline**: WavLM-base + Conformer (无对抗训练)
2. **+ 课程对抗**: 只加课程对抗训练
3. **+ 特征级对抗**: 只加特征级对抗
4. **方案B (完整)**: 课程对抗 + 特征级对抗

### 评测指标
- WER (Word Error Rate) on test-clean
- WER on test-other

## 文件结构
```
robust_asr/
├── train.py          # 训练脚本
├── README.md         # 本文件
└── hparams/
    └── train_wavlm_base_conformer.yaml  # 超参数配置
```

## 注意事项
1. 首次运行会自动下载 WavLM-base 模型和预训练 LM
2. 课程对抗训练会自动根据 epoch 调整噪声 SNR
3. 特征级对抗目前为简化实现，实际使用可能需要根据 WavLM 结构调整
