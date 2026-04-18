from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# Disable HuggingFace tokenizer process parallelism to avoid deadlocks with DataLoader workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# Global pre- / post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    Chat pre-processing: randomly inject a system message with probability ``add_system_ratio``.

    Behavior:
    - Inserts only when the first message is not already ``role == "system"``.
    - Randomness improves robustness to conversations with or without a system prompt.
    - System text is sampled from a fixed bilingual pool (mixed styles).
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    Chat post-processing: strip empty ``<think>`` blocks left by the template.

    For CoT-style templates, ``apply_chat_template`` may emit a placeholder like
    ``<think>\\n\\n</think>\\n\\n``.
    With probability ``1 - empty_think_ratio`` (default 95%), remove it so the model
    does not learn spurious "empty think" patterns. With probability ``empty_think_ratio``,
    keep it occasionally for boundary-case coverage.
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset — causal LM pre-training
# ──────────────────────────────────────────────────────────────────────────────
# Objective: next-token prediction on plain text.
# Record format: {"text": "raw string"}
# Notes:
#   - Every non-padding position can contribute to loss (no assistant-only masking).
#   - BOS/EOS frame the document so the model learns boundaries.
#   - Labels use -100 at PAD positions so CrossEntropy ignores padding.
#   - Labels are a clone of input_ids; the model shifts internally (Y[t] = X[t+1]).
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Lazy load via HuggingFace ``datasets`` to avoid loading huge JSON into RAM at once.
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1: tokenize text, leaving room for one BOS and one EOS inside max_length.
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # reserve BOS + EOS
            truncation=True,
        ).input_ids

        # Step 2: prepend BOS and append EOS.
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3: right-pad to max_length for fixed-shape batches.
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4: same as input_ids, but PAD -> -100 so loss ignores padding.
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # attention_mask: 1 for real tokens, 0 for PAD (attention should not attend to PAD).
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset — supervised fine-tuning (chat)
# ──────────────────────────────────────────────────────────────────────────────
# Objective: train the model to predict only the assistant turns; user/system are context.
# Record format: {"conversations": [{"role": "user"|"assistant"|"system", "content": "..."}]}
# Notes:
#   - ``generate_labels`` finds assistant spans via token patterns for BOS-of-assistant and EOS.
#   - Only assistant token positions get real labels; everything else is -100.
#   - Function calling: if the system message has a "functions" field, it is passed to
#     ``apply_chat_template`` as ``tools=...``.
#   - Unlike PretrainDataset, labels are sparse (mostly -100).
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # Token id sequence marking the start of an assistant turn (BOS + "assistant\n").
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        # Token id sequence marking end of assistant turn (EOS + newline).
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        Render multi-turn chat to a single string for the model.

        - Copies ``conversations`` to avoid mutating the underlying sample.
        - If the first message is system and defines ``functions``, passes ``tools=`` for tool-use templates.
        - ``add_generation_prompt=False``: training needs the full user+assistant string, not an open-ended continuation stub.
        """
        messages = conversations.copy()
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """
        Build sparse labels for SFT: only assistant spans use real token ids; rest is -100.

        Sliding scan:
        1. Start with all -100 (no loss).
        2. Find subsequence equal to ``bos_id`` (assistant header).
        3. Scan forward until ``eos_id`` matches (end of assistant turn).
        4. Set labels[start : end+len(eos_id)) to the corresponding ``input_ids`` (includes EOS).
        5. Continue after the span for multi-turn chats.
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                # Skip the assistant header tokens; supervise content + closing EOS.
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1: optional random system prompt (light augmentation).
        conversations = pre_processing_chat(sample["conversations"])

        # Step 2: render full conversation string.
        prompt = self.create_chat_prompt(conversations)

        # Step 3: drop empty redacted-thinking placeholders when sampled to remove.
        prompt = post_processing_chat(prompt)

        # Step 4: tokenize, truncate to max_length, pad on the right.
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Step 5: sparse labels — supervised only on assistant tokens.
        labels = self.generate_labels(input_ids)
        # # === debug ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ==============

        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset — Direct Preference Optimization
# ──────────────────────────────────────────────────────────────────────────────
# Objective: prefer "chosen" completions over "rejected" under a reference policy.
# Record format: {"chosen": [...], "rejected": [...]} (chat message lists)
# Notes:
#   - Each item returns tokenized ``chosen`` and ``rejected`` sequences.
#   - ``loss_mask`` / ``mask_*``: 1 only on assistant spans (same idea as SFT).
#   - Pairs are built in autoregressive form: x = ids[:-1], y = ids[1:], mask aligned with y.
#   - Default ``max_length`` is larger than SFT because preference pairs often need long context.
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # preferred dialogue: list of {role, content}
        rejected = sample["rejected"]  # dispreferred dialogue, same schema

        # Step 1: render each dialogue with the chat template.
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2: tokenize and pad to max_length for batching.
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 3: autoregressive shift — x predicts next token y; mask[1:] aligns with y.
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        Binary mask for DPO log-probability masking: 1 on assistant spans, 0 elsewhere.

        Same scan as ``SFTDataset.generate_labels``, but returns 0/1 instead of token ids
        for use in masked likelihood aggregation during DPO training.
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# ──────────────────────────────────────────────────────────────────────────────
# 4. RLAIFDataset — RL from AI feedback (PPO / GRPO-style trainers)
# ──────────────────────────────────────────────────────────────────────────────
# Objective: provide (prompt, reference answer) pairs for on-policy rollouts + scoring.
# Record format: {"conversations": [{"content": "..."}, ...]}
#   - Even indices (0,2,4,...) are treated as user turns; odd indices as assistant.
#   - The last message is the reference answer.
# Differences from SL datasets:
#   - No offline tokenization: returns raw strings so the RL loop can tokenize per rollout.
#   - ``create_chat_prompt`` drops the final assistant message, renders the rest with
#     ``add_generation_prompt=True`` so the model sees a "start generating" suffix.
#   - ``bos_id`` / ``eos_id`` are kept for possible future mask extensions but unused here.
#   - Returns dict[str, str], not tensors.
# ──────────────────────────────────────────────────────────────────────────────
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        Split a conversation into (prompt string, reference answer string).

        - Assigns roles by index: even -> user, odd -> assistant.
        - ``answer`` is repeatedly overwritten and ends as the last message (reference).
        - Renders ``messages[:-1]`` with ``add_generation_prompt=True`` so the template
          ends with tokens that cue the assistant to continue.
        - The RL actor rolls out from ``prompt``; rewards compare against ``answer``.
        """
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]  # final value = last turn (reference)
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # Raw strings only — RL trainer tokenizes during rollout.
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}


if __name__ == "__main__":
    pass
