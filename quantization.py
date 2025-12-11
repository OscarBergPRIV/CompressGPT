import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans

# ------------------- STE Heaviside with Temperature -------------------
class HeavisideSTETemp(torch.autograd.Function):
    """
    Forward: Heavside Step (Input >= 0 -> 1.0, else 0.0)
    Backward: Gradient of Sigmoid(T * x)
    """
    @staticmethod
    def forward(ctx, input, T):
        # Save input and T for backward pass calculation
        ctx.save_for_backward(input, T)
        return (input >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, T = ctx.saved_tensors
        # Surrogate gradient: derivative of sigmoid(T * x)
        # As T -> inf, this approaches a Dirac delta, so we must be careful with T choice.
        sig = torch.sigmoid(T * input)
        grad_input = grad_output * T * sig * (1 - sig)
        return grad_input, None  # No gradient for T

class TempStep(nn.Module):
    def forward(self, x: torch.Tensor, T: torch.Tensor):
        return HeavisideSTETemp.apply(x, T)

# ------------------- KMeans-Initialized Quantizer with Per-Step Alphas -------------------
class KMeansTrainableQuantizer(nn.Module):
    def __init__(self, num_bits: int, train_params: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        
        # We need (Levels - 1) thresholds and (Levels - 1) alphas
        self.num_thresholds = self.num_levels - 1
        
        # Learned thresholds
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_thresholds)]
        )
        
        # Learned per-step alphas (increments)
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_thresholds)]
        )
        
        # Base level (b)
        self.b = nn.Parameter(torch.tensor(0.0))
        
        # Enable or disable training of parameters
        self.b.requires_grad_(train_params)
        for p in self.thresholds:
            p.requires_grad_(train_params)
        for p in self.alphas:
            p.requires_grad_(train_params)
        
        self.step_fn = TempStep()

    def initialize_from_data(self, data: torch.Tensor):
        """
        1. Fit K-Means to find optimal centroids.
        2. Set thresholds to midpoints between centroids.
        3. Analytically set b = c0, alphas_i = c_{i+1} - c_i (exact differences).
        """
        device = data.device
        
        # 1. Run KMeans (on CPU)
        with torch.no_grad():
            x_np = data.detach().flatten().cpu().numpy().reshape(-1, 1)
            
            # K-Means initialization
            kmeans = KMeans(n_clusters=self.num_levels, random_state=42, n_init=10)
            kmeans.fit(x_np)
            
            # Get centroids and sort them
            centers = np.sort(kmeans.cluster_centers_.flatten())
            #print(f"[{self.num_bits} bit] Optimal Centroids: {np.round(centers, 3)}")
            
            # 2. Calculate Thresholds (Midpoints)
            new_thresholds = [(centers[i] + centers[i+1]) / 2.0 for i in range(len(centers) - 1)]
            for param, val in zip(self.thresholds, new_thresholds):
                param.data.fill_(val)
            
            # 3. Analytical Alphas and b (Exact!)
            self.b.data.fill_(centers[0])
            for i, alpha in enumerate(self.alphas):
                alpha.data.fill_(centers[i+1] - centers[i])
            
            # Print for verification
            reconstructed = [centers[0]]
            for diff in centers[1:] - centers[:-1]:
                reconstructed.append(reconstructed[-1] + diff)
            #print(f"[{self.num_bits} bit] Analytical b: {centers[0]:.4f}, Alphas: {np.round(centers[1:] - centers[:-1], 4)}")
            #print(f"[{self.num_bits} bit] Reconstructed Levels: {np.round(reconstructed, 3)}")

    def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Start from base b
        y = self.b + torch.zeros_like(x).to(self.b.device)  # Broadcast
        
        # Cumulatively add alpha_i * step(x - th_i)
        for i, th in enumerate(self.thresholds):
            #print("th: ", th.device)
            #print("alpha: ", self.alphas[i].device)
            #print("x: ", x.device)

            y += self.alphas[i] * self.step_fn(x - th, T.to(self.b.device))
        
        return y

# ------------------- Testing & Visualization -------------------
def visualize_quantizer(data: torch.Tensor, num_bits: int, T_val: float = 10.0, save_dir: str = "imgs"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Init model
    quant = KMeansTrainableQuantizer(num_bits=num_bits).to(data.device)
    quant.initialize_from_data(data)
    
    # Create visualization range covering the data
    x_vis = torch.linspace(data.min()*1.1, data.max()*1.1, 1000, device=data.device)
    
    # Forward pass
    T = torch.tensor(T_val, device=data.device)
    y_hard = quant(x_vis, T)
    
    # Calculate soft version manually for visualization (Soft Sigmoid approximation)
    y_soft = torch.zeros_like(x_vis)
    with torch.no_grad():
        y_soft += quant.b  # Broadcast scalar to [1000]
        for i, th in enumerate(quant.thresholds):
            y_soft += quant.alphas[i] * torch.sigmoid(T_val * (x_vis - th))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Input Distribution (Histogram)
    plt.hist(data.cpu().numpy().flatten(), bins=100, density=True,
             color='silver', alpha=0.4, label='Input Density')
    
    # 2. Plot Quantization Function
    plt.plot(x_vis.cpu().numpy(), y_hard.detach().cpu().numpy(),
             color='tab:blue', linewidth=2, label='Quantizer Output (Hard)')
    
    plt.plot(x_vis.cpu().numpy(), y_soft.detach().cpu().numpy(),
             color='tab:orange', linestyle='--', alpha=0.8, label=f'Soft Approx (T={T_val})')
    
    # 3. Plot Thresholds
    for i, th in enumerate(quant.thresholds):
        plt.axvline(th.detach().cpu(), color='red', linestyle=':', alpha=0.3,
                    label='Threshold' if i == 0 else "")
    
    # 4. Decorate
    plt.title(f"K-Means Initialized {num_bits}-bit Quantizer\n(Analytical Per-Step Alphas)")
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Save
    fname = f"quant_{num_bits}bit_T{int(T_val)}.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {save_path}")

# ------------------- Run -------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create complex multimodal distribution
    torch.manual_seed(42)
    data = torch.cat([
        torch.randn(10000, device=device) * 0.2 + 1.0,   # Mode at 1
        torch.randn(10000, device=device) * 0.5 + 2.0,   # Mode at 2
        torch.randn(5000, device=device) * 0.1 - 1,      # Mode at -1
    ])
    
    # 2. Run tests
    for bits in [2, 3, 4]:
        visualize_quantizer(data, num_bits=bits, T_val=20.0)
    
    print("#"*50)
    data = torch.cat([
        torch.randn(40000, device=device) * 5.9 + 50,    # Heavily clustered near 50
        torch.randn(40000, device=device) * 0.7 + 1.0,    # Secondary cluster near 1.0
        torch.randn(10000, device=device) * 1.5 - 8.0,
    ])
    data = data.reshape(10, 100, -1)  # Example reshaping, adjust as needed
    
    plt.hist(data.detach().flatten().cpu(), bins=300)
    plt.show()
    # --- Gradient Flow Verification (2-bit example) ---
    print("\n--- Running Gradient Verification (2-bit) ---")
    NUM_BITS = 6
    
    # Instantiate the 2-bit Quantizer
    quantizer = KMeansTrainableQuantizer(num_bits=NUM_BITS, train_params=True).to(device)
    
    # Initialize parameters using the data
    quantizer.initialize_from_data(data)
    
    # Set up input tensor for the gradient check
    input_x = torch.linspace(data.min().item(), data.max().item(), 1000, device=device)
    input_x.requires_grad_(True)
    
    # Use a high T to simulate hard quantization forward, but the soft gradient still works
    T_val = 100.0
    T = torch.tensor(T_val, device=device)
    
    # Perform Forward Pass on data (or input_x)
    quantized_output = quantizer(data, T)
    print(quantized_output.shape)

    plt.hist(quantized_output.detach().flatten().cpu(), bins=300)
    plt.show()
    
    # Calculate a dummy loss (e.g., sum of all outputs)
    dummy_loss = quantized_output.sum()
    
    # Zero out existing gradients
    quantizer.zero_grad()
    
    # Perform Backward Pass
    dummy_loss.backward()
    
    # --- Print Gradients ---
    print(f"\n✅ Gradient Check for {NUM_BITS}-bit Quantizer (T={T_val:.1f}):")
    
    # Print gradients for b
    print(f"   Base b Gradient: {quantizer.b.grad.item():.6f}")
    
    # Print gradients for alphas
    print("\n   Alpha Gradients:")
    for i, alpha in enumerate(quantizer.alphas):
        grad = alpha.grad.item() if alpha.grad is not None else 0.0
        print(f"   - Alpha {i} ({alpha.item():.4f}): {grad:.6f}")
    
    # Print gradients for thresholds
    print("\n   Threshold Gradients:")
    for i, th in enumerate(quantizer.thresholds):
        grad = th.grad.item() if th.grad is not None else 0.0
        print(f"   - Threshold {i+1} ({th.item():.4f}): {grad:.6f}")
    
    # Final check
    has_b_grad = quantizer.b.grad.abs().sum().item() > 1e-6
    has_alpha_grad = all(alpha.grad.abs().sum().item() > 1e-6 for alpha in quantizer.alphas)
    has_threshold_grad = all(th.grad.abs().sum().item() > 1e-6 for th in quantizer.thresholds)
    if has_b_grad and has_alpha_grad and has_threshold_grad:
        print("\n🎉 ALL trainable parameters received non-zero gradients. The STE is working correctly!")
    else:
        print("\n⚠️ WARNING: Some parameters have zero gradients. Check T value or input distribution.")
