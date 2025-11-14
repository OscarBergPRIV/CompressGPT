"""
collect_real_hidden_stream.py
→ CUDA + Streaming + No Padding + Verbose + Safe for 52k+ sequences

Collects real (non-padded) hidden states, using forward hooks and attention masks.
Streams to disk in batches to support large datasets.

Features:
- Fully configurable via CLI (argparse)
- No padding tokens saved
- Safe for 52k+ sequences
- Verbose logging
- Professional structure
"""

from __future__ import annotations

import torch
from torch import Tensor
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, BatchEncoding
import logging
import argparse

# Local imports (must be after argparse to avoid circular issues)
from dataset_alpaca import train_dataloader, cfg_m, model_name
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modeling_GPT_compress import GPT2LMHeadModelCompress, GPT2LMHeadModel


# --------------------------------------------------------------------------- #
#                               CLI ARGUMENT PARSER                           #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream real hidden states from GPT-2 (before bottleneck) to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument(
        "--save-root", type=Path, default=Path("collected_hidden_states"),
        help="Root directory to save collected hidden states"
    )
    parser.add_argument(
        "--save-name", type=str,
        default=f"real_hidden_data_{cfg_m.bl_layer}_{cfg_m.bl_ratio}",
        help="Name of the subdirectory inside save-root"
    )

    # Model & Bottleneck
    parser.add_argument(
        "--bl-layer", type=int, default=cfg_m.bl_layer,
        help="Bottleneck layer index (0 = after embeddings)"
    )
    parser.add_argument(
        "--bl-ratio", type=float, default=cfg_m.bl_ratio,
        help="Bottleneck compression ratio"
    )

    # Collection limits
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Maximum number of sequences to collect (None = all)"
    )
    parser.add_argument(
        "--print-every", type=int, default=10,
        help="Print batch info every N batches"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Force device (default: auto-detect)"
    )

    return parser


# --------------------------------------------------------------------------- #
#                                  CONFIG CLASS                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """Runtime configuration from CLI."""
    save_dir: Path
    bl_layer: int
    bl_ratio: float
    max_sequences: Optional[int]
    print_every: int
    device: torch.device

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Config":
        save_dir = args.save_root / args.save_name
        save_dir.mkdir(parents=True, exist_ok=True)

        device = (
            torch.device(args.device) if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        return Config(
            save_dir=save_dir,
            bl_layer=args.bl_layer,
            bl_ratio=args.bl_ratio,
            max_sequences=args.max_sequences,
            print_every=args.print_every,
            device=device
        )


# --------------------------------------------------------------------------- #
#                                  LOGGING SETUP                                #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                             MODEL & TOKENIZER SETUP                         #
# --------------------------------------------------------------------------- #

def setup_model_and_tokenizer(config: Config):
    """Load model, tokenizer, and tie weights."""
    log.info("=" * 70)
    log.info("LOADING MODEL & TOKENIZER")
    log.info("=" * 70)

    # Load config and models
    gpt_config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModelCompress(gpt_config, bl_layer=config.bl_layer, bl_ratio=config.bl_ratio)
    original_model = GPT2LMHeadModel.from_pretrained(model_name)

    # Transfer weights (skip bottleneck)
    model.transformer.load_state_dict(original_model.transformer.state_dict(), strict=False)

    # Tie lm_head to wte
    model.lm_head.weight = model.transformer.wte.weight

    # Move to device
    model.to(config.device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Device: {config.device}")
    log.info(f"Bottleneck layer: {model.transformer.bl_layer}")
    log.info(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    log.info(f"Save directory: {config.save_dir}")

    return model, tokenizer


# --------------------------------------------------------------------------- #
#                          REAL TOKEN STREAM COLLECTOR                        #
# --------------------------------------------------------------------------- #

class RealTokenStreamCollector:
    """
    Collects real (non-padded) hidden states from a target layer using forward hooks.
    Streams results to disk in batches.
    """

    def __init__(self, model: GPT2LMHeadModelCompress, tokenizer: AutoTokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.save_dir = config.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.hook = None
        self.current_mask: Optional[Tensor] = None
        self.batch_data: List[Tensor] = []
        self.total_saved: int = 0

    # ------------------------------------------------------------------- #
    def _forward_hook(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        """Hook: Extract real hidden states using attention mask."""
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.detach()
        B, S, D = hidden.shape

        log.debug(f"[HOOK] Hidden shape: {hidden.shape} | Device: {hidden.device}")

        # Determine mask
        if self.current_mask is None:
            mask = torch.ones(B, S, dtype=torch.bool, device=hidden.device)
            log.debug("[HOOK] No attention mask → using all tokens")
        else:
            mask = self.current_mask.bool()
            log.debug(f"[HOOK] Mask shape: {mask.shape}")
            log.debug(f"[HOOK] First mask: {mask[0].tolist()}")

        # Extract real tokens
        for b in range(B):
            real_tokens = hidden[b][mask[b]]
            if real_tokens.numel() == 0:
                log.warning(f"[HOOK] Sequence {b} has 0 real tokens!")
                continue
            self.batch_data.append(real_tokens.cpu())
            log.debug(f"[HOOK] → Saved seq {b}: {real_tokens.shape} | norm={real_tokens.norm().item():.1f}")

    # ------------------------------------------------------------------- #
    def collect(self, dataloader, max_sequences: Optional[int] = None, print_every: int = 1) -> None:
        """Main collection loop."""
        max_sequences = max_sequences if max_sequences is not None else self.config.max_sequences
        print_every = print_every or self.config.print_every

        # Hook into layer BEFORE bottleneck
        bl = self.model.transformer.bl_layer
        target = self.model.transformer.drop if bl == 0 else self.model.transformer.h[bl - 1]
        where = "after embeddings" if bl == 0 else f"after block {bl - 1}"
        log.info(f"\nHOOK → {where} (before bottleneck at layer {bl})")

        self.hook = target.register_forward_hook(self._forward_hook)

        batch_idx = 0
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if batch_idx % print_every == 0:
                        log.info(f"\n{'='*70}")
                        log.info(f"BATCH {batch_idx}")
                        log.info(f"{'='*70}")

                    # Parse batch
                    input_ids, self.current_mask = self._parse_batch(batch)
                    input_ids = input_ids.to(self.config.device)
                    if self.current_mask is not None:
                        self.current_mask = self.current_mask.to(self.config.device)

                    sequences_in_batch = input_ids.size(0)
                    log.info(f"Input IDs: {input_ids.shape}")
                    if self.current_mask is not None:
                        log.info(f"Mask sum per seq: {self.current_mask.sum(1).tolist()}")

                    # Log first sequence
                    self._log_first_sequence(input_ids, self.current_mask)

                    # Slice if needed
                    if max_sequences is not None:
                        remaining = max_sequences - self.total_saved
                        if remaining <= 0:
                            log.info(f"Reached max_sequences ({max_sequences}). Stopping.")
                            break
                        if remaining < sequences_in_batch:
                            log.info(f"Only need {remaining} more → slicing batch")
                            input_ids = input_ids[:remaining]
                            if self.current_mask is not None:
                                self.current_mask = self.current_mask[:remaining]
                            sequences_in_batch = remaining

                    # Forward pass
                    log.info(f"Forward pass on {self.config.device}...")
                    _ = self.model(input_ids=input_ids, attention_mask=self.current_mask)
                    log.info("Forward done.")

                    # Save batch
                    if self.batch_data:
                        path = self.save_dir / f"batch_{batch_idx:05d}.pt"
                        torch.save(self.batch_data, path)
                        saved_count = len(self.batch_data)
                        self.total_saved += saved_count
                        log.info(f"SAVED {path.name} → {saved_count} sequences")
                        log.info(f"Total saved so far: {self.total_saved:,}")
                        self.batch_data = []

                    batch_idx += 1

                    if max_sequences and self.total_saved >= max_sequences:
                        log.info(f"\nSTOPPING: Reached {self.total_saved} >= {max_sequences}")
                        break

        finally:
            if self.hook:
                self.hook.remove()
                log.info("Hook removed.")

        log.info(f"\nCOLLECTION COMPLETE!")
        log.info(f"Total sequences saved: {self.total_saved:,}")
        log.info(f"Files saved in: {self.save_dir}")
        if self.config.device.type == "cuda":
            torch.cuda.empty_cache()
            log.info("GPU memory cleared.")

    # ------------------------------------------------------------------- #
    def _parse_batch(self, batch: Any) -> tuple[Tensor, Optional[Tensor]]:
        """Parse batch into input_ids and attention_mask."""
        if isinstance(batch, BatchEncoding):
            input_ids = batch["input_ids"]
            mask = batch.get("attention_mask")
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            mask = batch.get("attention_mask")
        elif isinstance(batch, (tuple, list)):
            input_ids = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        else:
            input_ids = batch
            mask = None

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        return input_ids, mask

    # ------------------------------------------------------------------- #
    def _log_first_sequence(self, input_ids: Tensor, mask: Optional[Tensor]) -> None:
        """Decode and log first sequence."""
        first_ids = input_ids[0].cpu().tolist()
        full_text = self.tokenizer.decode(first_ids, skip_special_tokens=False)
        real_len = mask[0].sum().item() if mask is not None else len(first_ids)
        clean_text = self.tokenizer.decode(first_ids[:real_len], skip_special_tokens=True)

        log.info("\nFIRST SEQUENCE TEXT:")
        log.info(f"  Full:  \"{full_text}\"")
        log.info(f"  Clean: \"{clean_text}\"")

    # ------------------------------------------------------------------- #
    @staticmethod
    def list_files(save_dir: Path) -> None:
        """List saved batch files."""
        files = sorted(save_dir.glob("batch_*.pt"))
        log.info(f"\nSaved {len(files)} batch files:")
        for f in files[:5]:
            data = torch.load(f, map_location="cpu")
            log.info(f"  {f.name}: {len(data)} sequences")
        if len(files) > 5:
            log.info(f"  ... and {len(files)-5} more")

    # ------------------------------------------------------------------- #
    @staticmethod
    def load_all(save_dir: Path, max_seqs: Optional[int] = None) -> Tensor:
        """Load and concatenate all real tokens."""
        files = sorted(save_dir.glob("batch_*.pt"))
        all_tokens = []
        count = 0
        for f in files:
            batch = torch.load(f, map_location="cpu")
            all_tokens.append(torch.cat(batch, dim=0))
            count += sum(len(x) for x in batch)
            if max_seqs and count >= max_seqs:
                break
        result = torch.cat(all_tokens, dim=0)
        log.info(f"Loaded {result.shape[0]:,} real tokens → {result.shape}")
        return result


# --------------------------------------------------------------------------- #
#                                    MAIN                                     #
# --------------------------------------------------------------------------- #

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = Config.from_args(args)

    log.info("\n" + "="*70)
    log.info("STARTING STREAMING COLLECTION")
    log.info("="*70)

    model, tokenizer = setup_model_and_tokenizer(config)
    collector = RealTokenStreamCollector(model, tokenizer, config)

    collector.collect(
        dataloader=train_dataloader,
        max_sequences=config.max_sequences,
        print_every=config.print_every
    )

    RealTokenStreamCollector.list_files(config.save_dir)


if __name__ == "__main__":
    main()