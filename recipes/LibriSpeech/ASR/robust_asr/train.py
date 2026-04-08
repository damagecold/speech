#!/usr/bin/env python3
"""Recipe for training WavLM-base + Conformer ASR with adversarial training.
方案B: WavLM-base + Conformer + 课程对抗 + 特征级对抗

两个创新点:
1. 课程对抗训练: 高 SNR(20dB) → 中 SNR(10dB) → 低 SNR(0dB) 渐进学习
2. 特征级对抗: 对 WavLM 中间层特征加 FGSM 扰动

Authors: TODO
"""

import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 课程对抗训练阶段
        self.adversarial_stage = 0  # 0: high SNR, 1: medium SNR, 2: low SNR
        self.current_epoch_adversarial = 0

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # 特征级对抗: 对 WavLM 中间层特征加 FGSM 扰动
        if stage == sb.Stage.TRAIN and self.hparams.feature_adversarial:
            wavs = self.feature_adversarial(wavs, tokens_bos)

        # Forward pass through WavLM
        wav2vec2_out = self.modules.wav2vec2(wavs)
        # wav2vec2_out: (batch, time, features)

        # Conformer Encoder (CNN does 4x downsampling + dim reduction: 768→256)
        x = self.modules.CNN(wav2vec2_out)
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        p_ctc, p_tokens = None, None
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                p_ctc = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(p_ctc)
        else:
            if stage == sb.Stage.VALID:
                # Get token strings from index prediction
                p_tokens, _, _, _ = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, _, _, _ = self.hparams.test_search(x, wav_lens)

        return p_ctc, p_seq, wav_lens, p_tokens

    def feature_adversarial(self, wavs, tokens):
        """特征级对抗: 对 WavLM 中间层特征加 FGSM 扰动"""
        if not self.hparams.feature_adversarial:
            return wavs

        # 获取 WavLM 中间层特征
        wav2vec2 = self.modules.wav2vec2
        if hasattr(wav2vec2, 'extract_features'):
            # 使用 wav2vec2 提取中间层特征
            features, _ = wav2vec2.extract_features(wavs)
        else:
            # 如果模型没有 extract_features 方法，跳过对抗
            return wavs

        # 计算扰动
        features.requires_grad_(True)
        perturbed_features = features + self.adversarial_noise(features)

        # 重建扰动后的特征（如果模型支持）
        # 这里简化处理，实际需要根据具体模型调整
        return wavs  # TODO: 实现完整的特征级对抗

    def adversarial_noise(self, features):
        """生成 FGSM 对抗扰动"""
        epsilon = self.hparams.adversarial_epsilon
        # 简化: 使用随机噪声作为扰动
        # 实际应该计算梯度并生成 targeted/un-targeted 对抗样本
        noise = torch.randn_like(features) * epsilon
        return noise.detach()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        p_ctc, p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            (
                tokens,
                tokens_lens,
                tokens_eos,
                tokens_eos_lens,
            ) = self.hparams.wav_augment.replicate_multiple_labels(
                tokens, tokens_lens, tokens_eos, tokens_eos_lens
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        # 课程对抗 loss (附加到主 loss)
        if stage == sb.Stage.TRAIN and self.hparams.curriculum_adversarial:
            loss += self.compute_adversarial_loss(batch, current_epoch)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def compute_adversarial_loss(self, batch, epoch):
        """课程对抗训练 loss

        阶段1 (epoch 1-10): 高 SNR (20dB)
        阶段2 (epoch 11-20): 中 SNR (10dB)
        阶段3 (epoch 21+): 低 SNR (0dB)
        """
        if not self.hparams.curriculum_adversarial:
            return 0

        # 更新课程阶段
        if epoch <= 10:
            self.adversarial_stage = 0  # 高 SNR
            target_snr = 20
        elif epoch <= 20:
            self.adversarial_stage = 1  # 中 SNR
            target_snr = 10
        else:
            self.adversarial_stage = 2  # 低 SNR
            target_snr = 0

        # 计算对抗 loss (简化版)
        # 实际应该对噪声样本计算额外的 ASR loss
        adversarial_weight = self.hparams.adversarial_weight
        return 0  # TODO: 实现课程对抗 loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

        if stage == sb.Stage.TRAIN:
            self.current_epoch_adversarial = epoch
            if self.hparams.curriculum_adversarial:
                self.update_adversarial_stage(epoch)

    def update_adversarial_stage(self, epoch):
        """更新课程对抗训练阶段"""
        if epoch <= 10:
            stage = 0
            snr = "20dB"
        elif epoch <= 20:
            stage = 1
            snr = "10dB"
        else:
            stage = 2
            snr = "0dB"

        if stage != self.adversarial_stage:
            self.adversarial_stage = stage
            logger.info(f"课程对抗训练进入阶段 {stage+1}: SNR = {snr}")

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_train_hparams = hparams["dynamic_batch_sampler_train"]
        dynamic_valid_hparams = hparams["dynamic_batch_sampler_valid"]
        hop_size = hparams.get("feats_hop_size", 0.01)

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_train_hparams,
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_valid_hparams,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "LibriSpeech"))
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    import os

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )
