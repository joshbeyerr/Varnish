import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import os
from pathlib import Path
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
import json

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - some advanced features disabled")

@dataclass
class AdvancedConfig:
    """Advanced configuration for high-resource protection."""
    # Basic parameters
    epsilon: float = 16.0 / 255.0  # L∞ budget
    num_steps: int = 1000  # Much higher iteration count
    step_size: float = 0.5 / 255.0  # Smaller, more precise steps
    
    # Multi-scale attack
    scales: List[int] = None  # [256, 512, 1024] for multi-resolution
    scale_weights: List[float] = None  # Weights for each scale
    
    # Ensemble parameters
    num_ensemble_models: int = 5  # Virtual ensemble size
    ensemble_diversity: float = 0.3  # Diversity in ensemble
    
    # Advanced optimization
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop
    lr_schedule: str = "cosine"  # cosine, plateau, exponential
    gradient_clipping: float = 1.0
    momentum_decay: float = 0.9
    
    # Loss components
    perceptual_weight: float = 1.0
    frequency_weight: float = 0.5  # Reduced from 2.0
    semantic_weight: float = 0.5
    texture_weight: float = 1.0    # Reduced from 1.5
    adversarial_weight: float = 2.0  # Reduced from 3.0
    
    # Advanced techniques
    use_momentum: bool = True
    use_variance_reduction: bool = True
    use_gradient_penalty: bool = True
    use_spectral_normalization: bool = True
    use_adaptive_epsilon: bool = True
    
    # Multi-objective optimization
    pareto_optimization: bool = True
    objective_weights: Dict[str, float] = None
    
    # Resource usage
    batch_size: int = 4  # Process multiple variants simultaneously  
    num_workers: int = 8  # Parallel processing
    mixed_precision: bool = True
    memory_efficient: bool = False  # Trade speed for memory
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [256, 512, 1024]
        if self.scale_weights is None:
            self.scale_weights = [0.2, 0.5, 0.3]
        if self.objective_weights is None:
            self.objective_weights = {
                "imperceptibility": 0.3,
                "robustness": 0.4, 
                "transferability": 0.3
            }

class ProtectionLevel(Enum):
    LIGHT = "light"          # Fast, basic protection
    MEDIUM = "medium"        # Balanced protection
    HEAVY = "heavy"          # Strong protection, more resources
    EXTREME = "extreme"      # Maximum protection, highest resources
    ADAPTIVE = "adaptive"    # Adapts based on image content

class AdvancedMistProtector:
    """
    Advanced MIST protector with state-of-the-art adversarial techniques.
    Uses significantly more computational resources for stronger protection.
    """
    
    def __init__(self, 
                 config: AdvancedConfig = None,
                 device: str = "auto",
                 protection_level: ProtectionLevel = ProtectionLevel.HEAVY):
        """Initialize advanced protector."""
        self.device = self._setup_device(device)
        self.config = config or AdvancedConfig()
        self.protection_level = protection_level
        
        # Adjust config based on protection level
        self._adjust_config_for_protection_level()
        
        # Initialize advanced components
        self._init_perceptual_networks()
        self._init_frequency_analyzers()
        self._init_semantic_extractors()
        self._init_optimizers()
        
        print(f"Advanced MIST initialized on {self.device}")
        print(f"Protection level: {protection_level.value}")
        print(f"Max iterations: {self.config.num_steps}")
        print(f"Ensemble size: {self.config.num_ensemble_models}")
        
    def _setup_device(self, device: str) -> str:
        """Setup device with memory optimization."""
        if device == "auto":
            if torch.cuda.is_available():
                # Check VRAM and adjust settings
                total_memory = torch.cuda.get_device_properties(0).total_memory
                print(f"CUDA detected: {total_memory / 1e9:.1f}GB VRAM")
                return "cuda"
            else:
                print("CUDA not available, using CPU")
                return "cpu"
        return device
    
    def _adjust_config_for_protection_level(self):
        """Adjust configuration based on protection level."""
        if self.protection_level == ProtectionLevel.LIGHT:
            self.config.num_steps = 200
            self.config.num_ensemble_models = 2
            self.config.scales = [512]
            
        elif self.protection_level == ProtectionLevel.MEDIUM:
            self.config.num_steps = 500
            self.config.num_ensemble_models = 3
            self.config.scales = [512, 1024]
            
        elif self.protection_level == ProtectionLevel.HEAVY:
            self.config.num_steps = 1000
            self.config.num_ensemble_models = 5
            self.config.scales = [256, 512, 1024]
            
        elif self.protection_level == ProtectionLevel.EXTREME:
            self.config.num_steps = 800  # Reduced from 2000
            self.config.num_ensemble_models = 5  # Reduced from 8
            self.config.scales = [512, 1024]  # Reduced scales
            self.config.epsilon = 12.0 / 255.0
            self.config.step_size = 1.0 / 255.0  # Increased step size  # Tighter budget for better quality
            
    def _init_perceptual_networks(self):
        """Initialize perceptual loss networks."""
        print("Loading perceptual networks...")
        
        # Simplified perceptual network (you could use VGG, ResNet, etc.)
        class PerceptualNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # Multi-scale feature extraction
                self.features1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.features2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.features3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                )
            
            def forward(self, x):
                f1 = self.features1(x)
                f2 = self.features2(f1)
                f3 = self.features3(f2)
                return [f1, f2, f3]
        
        self.perceptual_net = PerceptualNetwork().to(self.device)
        
        # Freeze perceptual network
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
            
    def _init_frequency_analyzers(self):
        """Initialize frequency domain analysis tools."""
        self.frequency_bands = {
            'low': (0.0, 0.1),
            'mid': (0.1, 0.3), 
            'high': (0.3, 0.5),
            'ultra_high': (0.5, 1.0)
        }
        
    def _init_semantic_extractors(self):
        """Initialize semantic feature extractors."""
        # Placeholder for semantic extractors
        # In full implementation, could use CLIP, DINO, etc.
        self.semantic_extractors = {}
        
    def _init_optimizers(self):
        """Initialize optimizer factories."""
        self.optimizer_factories = {
            'adam': lambda params, lr: Adam(params, lr=lr, betas=(0.9, 0.999)),
            'adamw': lambda params, lr: AdamW(params, lr=lr, weight_decay=0.01),
            'sgd': lambda params, lr: SGD(params, lr=lr, momentum=0.9, nesterov=True),
            'rmsprop': lambda params, lr: torch.optim.RMSprop(params, lr=lr, momentum=0.9)
        }
    
    def create_advanced_pattern(self, 
                              shape: Tuple[int, ...], 
                              pattern_type: str,
                              complexity_level: int = 5) -> torch.Tensor:
        """
        Create advanced adversarial patterns with higher complexity.
        
        Args:
            shape: Tensor shape (B, C, H, W)
            pattern_type: Type of pattern
            complexity_level: Complexity level (1-10)
        """
        _, c, h, w = shape
        device = self.device
        
        if pattern_type == "neural_texture":
            # Generate neural texture using learned patterns
            # Create complex multi-scale texture
            pattern = torch.zeros((h, w), device=device)
            
            for scale in range(1, complexity_level + 1):
                freq = 2 ** scale
                x = torch.linspace(0, freq * np.pi, w, device=device)
                y = torch.linspace(0, freq * np.pi, h, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                # Complex wave combination
                wave = (torch.sin(X + 0.3 * torch.sin(3*Y)) * 
                       torch.cos(Y + 0.3 * torch.cos(2*X)) * 
                       torch.exp(-0.1 * (X**2 + Y**2) / (freq**2)))
                
                pattern += wave / (2 ** scale)
            
            # Add fractal-like details
            for i in range(complexity_level):
                noise_scale = 2 ** i
                noise = torch.randn((h // noise_scale, w // noise_scale), device=device)
                noise_up = F.interpolate(noise.unsqueeze(0).unsqueeze(0), 
                                       size=(h, w), mode='bicubic', align_corners=False)
                pattern += 0.1 * noise_up.squeeze() / noise_scale
                
            return pattern.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
            
        elif pattern_type == "adversarial_fourier":
            # Create adversarial patterns in frequency domain
            # Generate base noise
            noise = torch.randn((h, w), device=device)
            fft_noise = torch.fft.fft2(noise)
            
            # Create sophisticated frequency mask
            freq_x = torch.fft.fftfreq(w, device=device)
            freq_y = torch.fft.fftfreq(h, device=device)
            FX, FY = torch.meshgrid(freq_x, freq_y, indexing='ij')
            freq_mag = torch.sqrt(FX**2 + FY**2)
            
            # Multi-band frequency manipulation
            mask = torch.zeros_like(freq_mag)
            for i in range(complexity_level):
                band_center = (i + 1) * 0.1
                band_width = 0.05
                band_mask = torch.exp(-((freq_mag - band_center) / band_width)**2)
                mask += band_mask * ((-1) ** i)  # Alternating amplification/suppression
            
            # Apply sophisticated filtering
            filtered_fft = fft_noise * (1 + 0.5 * mask)
            pattern = torch.fft.ifft2(filtered_fft).real
            
            return pattern.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
            
        elif pattern_type == "chaos_theory":
            # Use chaos theory for unpredictable patterns
            x = torch.linspace(-2, 2, w, device=device)
            y = torch.linspace(-2, 2, h, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Strange attractor-inspired pattern
            pattern = torch.zeros_like(X)
            
            # Multiple chaotic systems
            for i in range(complexity_level):
                a, b, c = 1.4 + 0.1 * i, 0.3 + 0.05 * i, 2.0 + 0.1 * i
                
                # Henon-like map in 2D
                x_chaos = a - X**2 + b * Y
                y_chaos = X
                
                chaos_pattern = torch.sin(c * x_chaos) * torch.cos(c * y_chaos)
                chaos_pattern = torch.tanh(chaos_pattern)  # Bound the values
                
                pattern += chaos_pattern / (i + 1)
            
            # Add deterministic noise based on position
            det_noise = torch.sin(7 * X) * torch.cos(11 * Y) * torch.sin(13 * X * Y)
            pattern += 0.3 * det_noise
            
            return pattern.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
            
        elif pattern_type == "multi_objective":
            # Pattern optimized for multiple objectives simultaneously
            patterns = []
            
            # Combine multiple simpler patterns
            simple_patterns = ["neural_texture", "adversarial_fourier", "chaos_theory"]
            weights = torch.softmax(torch.randn(len(simple_patterns)), dim=0)
            
            combined_pattern = torch.zeros((h, w), device=device)
            for i, simple_type in enumerate(simple_patterns):
                simple_pattern = self.create_advanced_pattern(
                    shape, simple_type, complexity_level // 2
                ).squeeze()
                combined_pattern += weights[i] * simple_pattern
            
            return combined_pattern.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
            
        else:
            # Fallback to highly complex random pattern
            pattern = torch.zeros((h, w), device=device)
            for i in range(complexity_level * 2):
                scale = 2 ** (i // 2)
                freq = np.random.uniform(1, 5)
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                
                x = torch.linspace(0, freq * np.pi, w, device=device)
                y = torch.linspace(0, freq * np.pi, h, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                wave = torch.sin(X + phase_x) * torch.cos(Y + phase_y)
                pattern += wave / scale
                
            return pattern.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
    
    def compute_advanced_loss(self, 
                            original: torch.Tensor,
                            adversarial: torch.Tensor,
                            target_pattern: torch.Tensor,
                            iteration: int) -> Dict[str, torch.Tensor]:
        """Compute sophisticated multi-component loss."""
        losses = {}
        
        # 1. Perceptual loss (multiple scales)
        perceptual_loss = 0
        orig_features = self.perceptual_net(original)
        adv_features = self.perceptual_net(adversarial)
        
        for orig_feat, adv_feat in zip(orig_features, adv_features):
            perceptual_loss += F.mse_loss(orig_feat, adv_feat)
        losses['perceptual'] = perceptual_loss
        
        # 2. Multi-scale frequency loss
        frequency_loss = 0
        for scale in [1, 2, 4]:
            if scale > 1:
                orig_scaled = F.avg_pool2d(original, scale)
                adv_scaled = F.avg_pool2d(adversarial, scale)
            else:
                orig_scaled = original
                adv_scaled = adversarial
                
            # FFT analysis
            orig_fft = torch.fft.fft2(orig_scaled.mean(dim=1, keepdim=True))
            adv_fft = torch.fft.fft2(adv_scaled.mean(dim=1, keepdim=True))
            
            # Focus on different frequency bands
            for band_name, (low, high) in self.frequency_bands.items():
                freq_mask = self._create_frequency_mask(orig_fft.shape[-2:], low, high)
                
                orig_band = orig_fft * freq_mask
                adv_band = adv_fft * freq_mask
                
                band_loss = F.mse_loss(torch.abs(adv_band), torch.abs(orig_band))
                frequency_loss += band_loss / scale
                
        losses['frequency'] = frequency_loss
        
        # 3. Texture loss (local patterns)
        texture_loss = 0
        for kernel_size in [3, 5, 7]:
            # Compute local variance as texture measure
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size**2)
            
            for img in [original, adversarial]:
                img_gray = img.mean(dim=1, keepdim=True)
                local_mean = F.conv2d(img_gray, kernel, padding=kernel_size//2)
                local_var = F.conv2d(img_gray**2, kernel, padding=kernel_size//2) - local_mean**2
                
                if img is original:
                    orig_texture = local_var
                else:
                    adv_texture = local_var
                    
            texture_loss += F.mse_loss(adv_texture, orig_texture)
            
        losses['texture'] = texture_loss
        
        # 4. Adversarial steering loss
        steering_loss = F.mse_loss(adversarial, target_pattern)
        losses['adversarial'] = steering_loss
        
        # 5. Adaptive regularization (changes during optimization)
        adaptation_factor = 1.0 - (iteration / self.config.num_steps)
        regularization = self.config.epsilon**2 * adaptation_factor
        losses['regularization'] = torch.tensor(regularization, device=self.device)
        
        # 6. Gradient penalty (for stability)
        if self.config.use_gradient_penalty:
            grad_penalty = self._compute_gradient_penalty(original, adversarial)
            losses['gradient_penalty'] = grad_penalty
        
        return losses
    
    def _create_frequency_mask(self, shape, low_freq, high_freq):
        """Create frequency domain mask for specific band."""
        h, w = shape
        freq_x = torch.fft.fftfreq(w, device=self.device)
        freq_y = torch.fft.fftfreq(h, device=self.device)
        FX, FY = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_mag = torch.sqrt(FX**2 + FY**2)
        
        mask = ((freq_mag >= low_freq) & (freq_mag < high_freq)).float()
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _compute_gradient_penalty(self, original, adversarial, lambda_gp=0.1):
        """Compute gradient penalty for training stability."""
        alpha = torch.rand(original.shape[0], 1, 1, 1, device=self.device)
        interpolated = alpha * original + (1 - alpha) * adversarial
        interpolated.requires_grad_(True)
        
        # This is a simplified version - in full implementation would use discriminator
        d_interpolated = interpolated.mean()  # Placeholder
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=[1,2,3]) - 1) ** 2).mean()
        return lambda_gp * gradient_penalty  # Much smaller lambda_gp
    
    def generate_advanced_adversarial_noise(self, 
                                          clean_image: torch.Tensor,
                                          pattern_type: str = "multi_objective") -> torch.Tensor:
        """Generate adversarial noise using advanced optimization."""
        print(f"Advanced adversarial generation: {self.config.num_steps} iterations")
        start_time = time.time()
        
        # Multi-scale processing
        multi_scale_deltas = []
        
        for scale_idx, scale in enumerate(self.config.scales):
            print(f"Processing scale {scale}px ({scale_idx+1}/{len(self.config.scales)})")
            
            # Resize image to current scale
            if clean_image.shape[-1] != scale:
                scaled_image = F.interpolate(clean_image, size=(scale, scale), 
                                           mode='bilinear', align_corners=False)
            else:
                scaled_image = clean_image
            
            # Initialize perturbation with momentum
            delta = torch.zeros_like(scaled_image, requires_grad=True)
            
            if self.config.use_momentum:
                momentum = torch.zeros_like(scaled_image)
                
            # Create advanced target pattern
            target_pattern = self.create_advanced_pattern(
                scaled_image.shape, pattern_type, complexity_level=8
            )
            
            # Setup advanced optimizer
            optimizer_factory = self.optimizer_factories[self.config.optimizer_type]
            optimizer = optimizer_factory([delta], self.config.step_size)
            
            # Setup learning rate scheduler
            if self.config.lr_schedule == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.config.num_steps)
            elif self.config.lr_schedule == "plateau":
                scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.8)
                
            # Optimization loop with advanced techniques
            best_loss = float('inf')
            best_delta = None
            patience_counter = 0
            
            for step in range(self.config.num_steps):
                optimizer.zero_grad()
                
                # Apply perturbation
                adversarial_image = scaled_image + delta
                
                # Compute advanced multi-component loss
                losses = self.compute_advanced_loss(
                    scaled_image, adversarial_image, target_pattern, step
                )
                
                # Combine losses with balanced weighting
                total_loss = 0
                loss_weights = {
                    'perceptual': self.config.perceptual_weight,
                    'frequency': self.config.frequency_weight, 
                    'texture': self.config.texture_weight,
                    'adversarial': self.config.adversarial_weight,
                    'gradient_penalty': 0.1,  # Fixed small weight
                    'regularization': 0.01    # Very small weight
                }
                
                for loss_name, loss_value in losses.items():
                    if loss_name in loss_weights:
                        weight = loss_weights[loss_name]
                        total_loss += weight * loss_value
                
                # We want to maximize disruption, so minimize negative loss
                total_loss = -total_loss
                
                total_loss.backward()
                
                # Gradient clipping for stability
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_([delta], self.config.gradient_clipping)
                
                # Advanced momentum update
                if self.config.use_momentum:
                    momentum = self.config.momentum_decay * momentum + delta.grad
                    delta.grad = momentum
                
                optimizer.step()
                
                # Project to epsilon ball with adaptive constraints
                with torch.no_grad():
                    if self.config.use_adaptive_epsilon:
                        # Adapt epsilon based on image content
                        local_epsilon = self.config.epsilon * (1 + 0.2 * torch.rand_like(delta))
                        delta.clamp_(-local_epsilon, local_epsilon)
                    else:
                        delta.clamp_(-self.config.epsilon, self.config.epsilon)
                    
                    # Ensure output is valid
                    adversarial_image = scaled_image + delta
                    adversarial_image.clamp_(0, 1)
                    delta.copy_(adversarial_image - scaled_image)
                
                # Learning rate scheduling
                if self.config.lr_schedule == "cosine":
                    scheduler.step()
                elif self.config.lr_schedule == "plateau":
                    scheduler.step(total_loss)
                
                # Early stopping with more reasonable patience
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_delta = delta.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter > 50 and step > 200:  # More aggressive early stopping
                    print(f"Early stopping at step {step}")
                    break
                
                # Progress reporting - less frequent for speed
                if step % 50 == 0:  # Every 50 steps instead of 100
                    elapsed = time.time() - start_time
                    print(f"Scale {scale}, Step {step}/{self.config.num_steps}")
                    print(f"Total loss: {total_loss.item():.6f}")
                    print(f"  frequency: {losses['frequency'].item():.6f}")
                    print(f"  adversarial: {losses['adversarial'].item():.6f}")
                    if 'gradient_penalty' in losses:
                        print(f"  gradient_penalty: {losses['gradient_penalty'].item():.6f}")
                    print(f"Elapsed: {elapsed:.1f}s, Patience: {patience_counter}/50")
            
            # Use best delta found
            if best_delta is not None:
                delta = best_delta
                
            # Resize delta back to original size if needed
            if delta.shape[-1] != clean_image.shape[-1]:
                delta = F.interpolate(delta, size=clean_image.shape[-2:], 
                                    mode='bilinear', align_corners=False)
            
            multi_scale_deltas.append(delta * self.config.scale_weights[scale_idx])
        
        # Combine multi-scale perturbations
        final_delta = sum(multi_scale_deltas)
        
        total_time = time.time() - start_time
        print(f"Advanced generation completed in {total_time:.1f}s")
        
        return final_delta
    
    def protect_image_advanced(self, 
                              image: Image.Image,
                              pattern_type: str = "multi_objective",
                              save_intermediates: bool = False) -> Dict[str, Any]:
        """
        Apply advanced protection with detailed analysis.
        
        Returns:
            Dictionary containing protected image and analysis data
        """
        print(f"Advanced protection: {pattern_type}")
        start_time = time.time()
        
        original_size = image.size
        
        # Preprocessing with multiple scales
        processed_image = image.resize((max(self.config.scales), max(self.config.scales)), Image.LANCZOS)
        clean_tensor = self._image_to_tensor(processed_image)
        
        # Generate advanced adversarial noise
        with torch.enable_grad():
            delta = self.generate_advanced_adversarial_noise(clean_tensor, pattern_type)
        
        # Apply protection
        protected_tensor = clean_tensor + delta
        protected_tensor.clamp_(0, 1)
        
        # Convert back to image
        protected_image = self._tensor_to_image(protected_tensor, original_size)
        
        # Advanced analysis
        analysis = self._analyze_protection(image, protected_image, delta)
        
        total_time = time.time() - start_time
        analysis['processing_time'] = total_time
        
        result = {
            'protected_image': protected_image,
            'analysis': analysis,
            'config': self.config,
            'pattern_type': pattern_type
        }
        
        if save_intermediates:
            result['noise_visualization'] = self._visualize_noise(delta)
            result['frequency_analysis'] = self._analyze_frequency_content(clean_tensor, protected_tensor)
        
        print(f"Advanced protection completed in {total_time:.1f}s")
        return result
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor with proper preprocessing."""
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _tensor_to_image(self, tensor: torch.Tensor, target_size: Tuple[int, int]) -> Image.Image:
        """Convert tensor back to PIL image."""
        # Detach from computation graph and move to CPU
        img_array = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        if image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)
        
        return image
    
    def _analyze_protection(self, original: Image.Image, protected: Image.Image, delta: torch.Tensor) -> Dict:
        """Comprehensive protection analysis."""
        # Convert to tensors for analysis
        orig_tensor = self._image_to_tensor(original.resize(protected.size, Image.LANCZOS))
        prot_tensor = self._image_to_tensor(protected)
        
        # Basic metrics
        mse = F.mse_loss(orig_tensor, prot_tensor).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        linf_norm = delta.abs().max().item()
        
        # Advanced metrics
        ssim_score = self._compute_ssim(orig_tensor, prot_tensor)
        lpips_score = self._compute_perceptual_distance(orig_tensor, prot_tensor)
        
        # Frequency analysis
        freq_diversity = self._compute_frequency_diversity(delta)
        
        # Robustness estimation
        robustness_score = self._estimate_robustness(delta)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_score,
            'lpips': lpips_score,
            'linf_norm': linf_norm,
            'linf_norm_255': linf_norm * 255,
            'frequency_diversity': freq_diversity,
            'robustness_score': robustness_score,
            'noise_statistics': {
                'std': delta.std().item(),
                'mean': delta.mean().item(),
                'min': delta.min().item(),
                'max': delta.max().item()
            }
        }
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Structural Similarity Index."""
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def _compute_perceptual_distance(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute perceptual distance using feature networks."""
        with torch.no_grad():
            features1 = self.perceptual_net(img1)
            features2 = self.perceptual_net(img2)
            
            distance = 0
            for f1, f2 in zip(features1, features2):
                distance += F.mse_loss(f1, f2).item()
                
        return distance
    
    def _compute_frequency_diversity(self, delta: torch.Tensor) -> Dict[str, float]:
        """Analyze frequency diversity of the noise."""
        # Convert to frequency domain
        delta_gray = delta.mean(dim=1, keepdim=True)
        fft_delta = torch.fft.fft2(delta_gray)
        magnitude = torch.abs(fft_delta)
        
        # Analyze different frequency bands
        diversity = {}
        for band_name, (low, high) in self.frequency_bands.items():
            mask = self._create_frequency_mask(delta_gray.shape[-2:], low, high)
            band_energy = (magnitude * mask).sum().item()
            diversity[f'{band_name}_energy'] = band_energy
            
        # Compute spectral flatness (measure of noise-likeness)
        log_magnitude = torch.log(magnitude + 1e-8)
        geometric_mean = torch.exp(log_magnitude.mean())
        arithmetic_mean = magnitude.mean()
        spectral_flatness = (geometric_mean / arithmetic_mean).item()
        diversity['spectral_flatness'] = spectral_flatness
        
        return diversity
    
    def _estimate_robustness(self, delta: torch.Tensor) -> Dict[str, float]:
        """Estimate robustness against common defenses."""
        robustness = {}
        
        # Robustness against Gaussian blur
        # Apply blur channel-wise using groups parameter
        kernel = self._get_gaussian_kernel()
        blurred_delta = F.conv2d(delta, kernel, padding=2, groups=3)  # groups=3 for 3 channels
        blur_preservation = F.mse_loss(delta, blurred_delta).item()
        robustness['blur_resistance'] = 1.0 / (1.0 + blur_preservation)
        
        # Robustness against JPEG compression (approximated)
        compressed_delta = self._simulate_jpeg_compression(delta)
        jpeg_preservation = F.mse_loss(delta, compressed_delta).item()
        robustness['jpeg_resistance'] = 1.0 / (1.0 + jpeg_preservation)
        
        # Robustness against rescaling
        downscaled = F.interpolate(delta, scale_factor=0.5, mode='bilinear', align_corners=False)
        upscaled = F.interpolate(downscaled, size=delta.shape[-2:], mode='bilinear', align_corners=False)
        scale_preservation = F.mse_loss(delta, upscaled).item()
        robustness['scale_resistance'] = 1.0 / (1.0 + scale_preservation)
        
        return robustness
    
    def _get_gaussian_kernel(self, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Create Gaussian blur kernel for 3-channel input."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= (kernel_size - 1) / 2.0
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
        
        # Create kernel for grouped convolution: [out_channels, in_channels/groups, kH, kW]
        # For groups=3, each group has 1 input and 1 output channel
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
        kernel = kernel.repeat(3, 1, 1, 1)  # [3, 1, kH, kW] for 3 groups
        
        return kernel
    
    def _simulate_jpeg_compression(self, image: torch.Tensor, quality: int = 85) -> torch.Tensor:
        """Simulate JPEG compression effects."""
        # Simplified JPEG simulation using frequency domain manipulation
        # Real implementation would use actual JPEG encoding/decoding
        
        # Convert to frequency domain
        image_gray = image.mean(dim=1, keepdim=True)
        fft_image = torch.fft.fft2(image_gray)
        
        # Apply frequency-based compression (remove high frequencies)
        h, w = fft_image.shape[-2:]
        freq_x = torch.fft.fftfreq(w, device=self.device)
        freq_y = torch.fft.fftfreq(h, device=self.device)
        FX, FY = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_mag = torch.sqrt(FX**2 + FY**2)
        
        # Quality-based cutoff
        cutoff = 0.3 * (quality / 100.0)
        compression_mask = (freq_mag < cutoff).float()
        
        compressed_fft = fft_image * compression_mask
        compressed_image = torch.fft.ifft2(compressed_fft).real
        
        return compressed_image.repeat(1, 3, 1, 1)
    
    def _visualize_noise(self, delta: torch.Tensor) -> Image.Image:
        """Create visualization of the adversarial noise."""
        # Normalize noise for visualization
        noise_vis = delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Normalize to [0, 1] for visualization
        noise_min = noise_vis.min()
        noise_max = noise_vis.max()
        if noise_max > noise_min:
            noise_vis = (noise_vis - noise_min) / (noise_max - noise_min)
        
        # Convert to uint8
        noise_vis = (noise_vis * 255).astype(np.uint8)
        
        return Image.fromarray(noise_vis)
    
    def _analyze_frequency_content(self, original: torch.Tensor, protected: torch.Tensor) -> Dict:
        """Analyze frequency content changes."""
        analysis = {}
        
        for name, tensor in [('original', original), ('protected', protected)]:
            # Convert to grayscale for frequency analysis
            gray = tensor.mean(dim=1, keepdim=True)
            fft = torch.fft.fft2(gray)
            magnitude = torch.abs(fft)
            
            # Analyze each frequency band
            band_analysis = {}
            for band_name, (low, high) in self.frequency_bands.items():
                mask = self._create_frequency_mask(gray.shape[-2:], low, high)
                band_energy = (magnitude * mask).sum().item()
                band_analysis[band_name] = band_energy
                
            analysis[name] = band_analysis
        
        # Compute frequency changes
        frequency_changes = {}
        for band_name in self.frequency_bands.keys():
            original_energy = analysis['original'][band_name]
            protected_energy = analysis['protected'][band_name]
            
            if original_energy > 0:
                change_ratio = protected_energy / original_energy
            else:
                change_ratio = float('inf') if protected_energy > 0 else 1.0
                
            frequency_changes[f'{band_name}_change_ratio'] = change_ratio
        
        analysis['frequency_changes'] = frequency_changes
        return analysis
    
    def batch_protect_advanced(self, 
                              images: List[Image.Image],
                              pattern_types: List[str] = None,
                              parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Protect multiple images with advanced techniques in parallel.
        
        Args:
            images: List of PIL Images to protect
            pattern_types: List of pattern types (or None for auto-selection)
            parallel: Whether to use parallel processing
            
        Returns:
            List of protection results with analysis
        """
        if pattern_types is None:
            pattern_types = ["multi_objective"] * len(images)
        elif len(pattern_types) == 1:
            pattern_types = pattern_types * len(images)
        elif len(pattern_types) != len(images):
            raise ValueError("Number of pattern types must match number of images")
        
        print(f"Batch protecting {len(images)} images")
        print(f"Parallel processing: {parallel}")
        
        if parallel and len(images) > 1:
            # Parallel processing using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(self.config.num_workers, len(images))) as executor:
                futures = []
                for i, (image, pattern) in enumerate(zip(images, pattern_types)):
                    future = executor.submit(self.protect_image_advanced, image, pattern)
                    futures.append(future)
                
                results = []
                for i, future in enumerate(futures):
                    print(f"Completing image {i+1}/{len(images)}")
                    result = future.result()
                    results.append(result)
                    
        else:
            # Sequential processing
            results = []
            for i, (image, pattern) in enumerate(zip(images, pattern_types)):
                print(f"Processing image {i+1}/{len(images)}")
                result = self.protect_image_advanced(image, pattern)
                results.append(result)
        
        return results
    
    def adaptive_protect(self, 
                        image: Image.Image,
                        content_analysis: bool = True) -> Dict[str, Any]:
        """
        Adaptively protect image based on content analysis.
        
        Args:
            image: PIL Image to protect
            content_analysis: Whether to analyze content for pattern selection
            
        Returns:
            Protection result with adaptive analysis
        """
        print("Adaptive protection with content analysis")
        
        if content_analysis:
            # Analyze image content to select optimal protection
            content_type = self._analyze_image_content(image)
            pattern_type = self._select_optimal_pattern(content_type)
            
            # Adapt protection level based on content
            original_level = self.protection_level
            self.protection_level = self._select_protection_level(content_type)
            self._adjust_config_for_protection_level()
            
            print(f"Detected content type: {content_type}")
            print(f"Selected pattern: {pattern_type}")
            print(f"Protection level: {self.protection_level.value}")
        else:
            pattern_type = "multi_objective"
        
        # Apply advanced protection
        result = self.protect_image_advanced(image, pattern_type, save_intermediates=True)
        
        # Add adaptive analysis
        result['adaptive_analysis'] = {
            'content_type': content_type if content_analysis else 'unknown',
            'selected_pattern': pattern_type,
            'protection_level': self.protection_level.value
        }
        
        # Restore original protection level
        if content_analysis:
            self.protection_level = original_level
            self._adjust_config_for_protection_level()
        
        return result
    
    def _analyze_image_content(self, image: Image.Image) -> str:
        """Analyze image content to determine optimal protection strategy."""
        # Convert to tensor for analysis
        img_tensor = self._image_to_tensor(image)
        
        # Simple content analysis based on image statistics
        # In a full implementation, this could use trained classifiers
        
        # Analyze color distribution
        color_std = img_tensor.std(dim=[2, 3]).mean().item()
        
        # Analyze edge content
        # Simple edge detection using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gray = img_tensor.mean(dim=1, keepdim=True)
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = edge_magnitude.mean().item()
        
        # Analyze frequency content
        fft = torch.fft.fft2(gray)
        freq_magnitude = torch.abs(fft)
        high_freq_energy = freq_magnitude[..., freq_magnitude.shape[-1]//4:].mean().item()
        
        # Simple heuristic classification
        if edge_density > 0.3:
            content_type = "line_art"
        elif color_std < 0.15:
            content_type = "low_contrast"
        elif high_freq_energy > 0.1:
            content_type = "detailed"
        elif color_std > 0.25:
            content_type = "colorful"
        else:
            content_type = "photographic"
            
        return content_type
    
    def _select_optimal_pattern(self, content_type: str) -> str:
        """Select optimal protection pattern based on content type."""
        pattern_mapping = {
            "line_art": "edge_chaos",
            "low_contrast": "gradient_explosion", 
            "detailed": "neural_texture",
            "colorful": "adversarial_fourier",
            "photographic": "multi_objective"
        }
        
        return pattern_mapping.get(content_type, "multi_objective")
    
    def _select_protection_level(self, content_type: str) -> ProtectionLevel:
        """Select protection level based on content type."""
        level_mapping = {
            "line_art": ProtectionLevel.EXTREME,  # Line art needs strong protection
            "low_contrast": ProtectionLevel.HEAVY,
            "detailed": ProtectionLevel.HEAVY, 
            "colorful": ProtectionLevel.MEDIUM,
            "photographic": ProtectionLevel.HEAVY
        }
        
        return level_mapping.get(content_type, ProtectionLevel.HEAVY)
    
    def save_protection_report(self, 
                              results: List[Dict[str, Any]], 
                              output_path: str):
        """Save comprehensive protection report."""
        report = {
            'timestamp': time.time(),
            'config': {
                'protection_level': self.protection_level.value,
                'num_steps': self.config.num_steps,
                'epsilon': self.config.epsilon,
                'scales': self.config.scales,
                'ensemble_size': self.config.num_ensemble_models
            },
            'results': []
        }
        
        for i, result in enumerate(results):
            result_summary = {
                'image_index': i,
                'pattern_type': result['pattern_type'],
                'processing_time': result['analysis']['processing_time'],
                'quality_metrics': {
                    'psnr': result['analysis']['psnr'],
                    'ssim': result['analysis']['ssim'],
                    'lpips': result['analysis']['lpips']
                },
                'protection_metrics': {
                    'linf_norm_255': result['analysis']['linf_norm_255'],
                    'robustness_score': result['analysis']['robustness_score'],
                    'frequency_diversity': result['analysis']['frequency_diversity']
                }
            }
            report['results'].append(result_summary)
        
        # Save report as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Protection report saved to {output_path}")


def main():
    """Demonstration of advanced MIST protector."""
    print("=== Advanced MIST Protector ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize with different protection levels
    configs = {
        'extreme': AdvancedConfig(
            num_steps=2000,
            epsilon=12.0/255.0,
            scales=[256, 512, 1024, 2048],
            num_ensemble_models=8
        )
    }
    
    # Example usage
    protector = AdvancedMistProtector(
        config=configs.get('extreme'),
        protection_level=ProtectionLevel.EXTREME
    )
    
    print(f"\nAdvanced protector initialized!")
    print(f"Available pattern types:")
    patterns = ["neural_texture", "adversarial_fourier", "chaos_theory", "multi_objective"]
    for pattern in patterns:
        print(f"  - {pattern}")
    
    print(f"\nExample usage:")
    
    # Single image protection
    image = Image.open("test3.jpg")
    result = protector.protect_image_advanced(image, "adversarial_fourier")
    result['protected_image'].save("protected_artwork.jpg")

    # protector.save_protection_report(result, "protection_report.json")
    
    # Batch protection
    # images = [Image.open(f"image_{i}.png") for i in range(5)]
    # results = protector.batch_protect_advanced(images, parallel=True)
    
    # # Adaptive protection
    # adaptive_result = protector.adaptive_protect(image, content_analysis=True)
    
    # # Save comprehensive report
    # protector.save_protection_report(results, "protection_report.json")
    
    
    # print("\nReady for advanced image protection!")


if __name__ == "__main__":
    main()