"""
Configuration file for GPT-2 compression experiments.
Allows easy modification of hyperparameters.
"""

class CFG_M:
    def __init__(self):    
        # LoRA configuration
        self.r_LoRA = 8  # LoRA rank (low-rank adaptation dimension)
        
        # Bottleneck configuration
        self.bl_layer = 3  # Layer position for bottleneck (0-11 for GPT-2)
        self.bl_ratio = 16  # Compression ratio (hidden_dim / bl_ratio)
        self.BL_type = "linear" # linear or attention or conv
        # Training configuration
        self.num_epochs = 1  # Number of training epochs
        
        # quant params
        self.quant_bits = 0  # 0 means no quantization, i.e. full fp32 smashed data

        # DCT params
        self.dct_BLOCK = 48
        self.dct_k = 0  # self.dct_BLOCK // 2 # 0 means no DCT compression 
        self.dct_reg = False
        
        # Model selection
        self.default = False  # If True, use standard GPT-2; if False, use compressed version
    
    def print_cfg(self):
        """Print current configuration settings."""
        print(f"Configuration:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  LoRA rank: {self.r_LoRA}")
        print(f"  Bottleneck layer: {self.bl_layer}")
        print(f"  Bottleneck ratio: {self.bl_ratio}")
        print(f"  Bottleneck type: {self.BL_type}")
        print(f"  Using default model: {self.default}")


if __name__ == "__main__":
    # Standalone test
    cfg = CFG_M()
    cfg.print_cfg()
    
    print("\n--- Testing configuration modification ---")
    cfg.r_LoRA = 16
    cfg.bl_layer = 5
    cfg.bl_ratio = 8
    cfg.print_cfg()
