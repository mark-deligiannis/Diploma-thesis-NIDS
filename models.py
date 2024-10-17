import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from os.path import join
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import floor

###
CYAN = '\033[36m'
COLOR_RESET = '\033[0m'
###

class ModelBase(ABC):
    def _init_weights(self, Layer):
        if Layer.__class__.__name__ == 'Linear':
            nn.init.normal_(Layer.weight, mean=0, std=0.02)
            if Layer.bias is not None:
                nn.init.constant_(Layer.bias, 0)

    def init_weights(self):
        for component in self.components:
            component.apply(self._init_weights)

    @abstractmethod
    # e.g. self.generator = self.components[0]
    def _set_component_handles(self):
        pass

    def __init__(self, name, components, device='cpu', inference_requires_autograd=False):
        super().__init__()
        self.name = name
        self.device = device
        self.components = components
        self.inference_requires_autograd = inference_requires_autograd
        self._inference_params_set = False
        self._set_component_handles()
        self.init_weights()
    
    @abstractmethod
    def set_inference_params(self, **kwargs):
        pass

    def _train(self):
        for component in self.components:
            component.train()
            if self.inference_requires_autograd: component.requires_grad_(True)

    def _eval(self):
        for component in self.components:
            component.eval()
            if self.inference_requires_autograd: component.requires_grad_(False)
    
    @abstractmethod
    def _optimizers_init(self, optim, optimizer_params):
        pass

    @abstractmethod
    def _test_val(self, real_samples):
        pass

    @abstractmethod
    def _train_step(self, i, epoch, real_samples, **kwargs):
        pass

    def train(self, dataloader, n_epochs, counterdataloader=None, optimizer_params={"lr": 0.0001, "betas": (0.5, 0.999)}, optim=torch.optim.Adam, sample_interval=500, save_at={}, save_path="./", **kwargs):
        len_data = len(dataloader)
        if counterdataloader is not None:
            # The counterdataloader often is not long enough
            # Create a wrapper dataloader that circles back to the beginning
            def counterdataloader_repeat():
                while True: yield from counterdataloader

        max_epoch_digits = len(str(n_epochs))
        max_i = len(str(len_data))
        self._optimizers_init(optim, optimizer_params)
        self._train()

        for epoch in range(1,n_epochs+1):
            data = dataloader if counterdataloader is None else zip(dataloader, counterdataloader_repeat())
            for i, real_samples in enumerate(data):
                if isinstance(real_samples,tuple):
                    real_samples = ( real_samples[0].to(self.device), real_samples[1].to(self.device) )
                else:
                    real_samples = real_samples.to(self.device)
                self._train_step(i, epoch, real_samples, **kwargs)
                if i % sample_interval == 0:
                    self._eval()
                    if self.inference_requires_autograd:
                        metrics = self._test_val(real_samples)
                    else:
                        with torch.inference_mode():
                            metrics = self._test_val(real_samples)
                    self._train()
                    print(f"[Epoch {epoch:{max_epoch_digits}d}/{n_epochs}] [Batch {i:{max_i}d}/{len_data}] ", end="")
                    for metric, value in metrics.items():
                        print(f"[{metric}: {value:.6f}] ", end="")
                    print()
            if epoch in save_at:
                print(f"Epoch {epoch} complete! Saving...",end=" ")
                timestamp = self.save(save_path)
                print(f"Saved successfully! Timestamp = {CYAN}{timestamp}{COLOR_RESET}")

    @abstractmethod
    def _anomaly_score(self, features):
        pass

    def _infer_no_inf(self, features, include_anomal_scores=False):
        anomaly_scores = self._anomaly_score(features)
        decisions = anomaly_scores > self.anomaly_threshold
        if include_anomal_scores: return anomaly_scores, decisions
        return decisions

    def infer(self, features):
        # In the validation phase inference params are used. Make sure
        # they are set before training (if not, exception is raised)
        if not self._inference_params_set:
            raise Exception("Inference parameters not set.")
        
        dev = features.device
        self._eval()
        if self.inference_requires_autograd:
            result = self._infer_no_inf(features).to(dev)
        else:
            with torch.inference_mode():
                result = self._infer_no_inf(features).to(dev)
        self._train()
        return result

    def test(self, dataloader, metrics, anomaly_fraction, metrics_need_AS_indx={}):
        self._eval()
        with torch.inference_mode():
            anomaly_scores, targets = [], []
            for features, labels in tqdm(dataloader):
                anomaly_score = self._anomaly_score(features.to(self.device)).cpu()
                anomaly_scores.append(anomaly_score)
                targets.append(labels)
            anomaly_scores, targets = torch.cat(tuple(anomaly_scores), dim=0), torch.cat(tuple(targets), dim=0)
            anomaly_scores, sort_indices = torch.sort(anomaly_scores)
            targets = targets[sort_indices]
            decisions = torch.zeros_like(targets)
            change_point = -int(anomaly_fraction*len(decisions))
            decisions[change_point:] = 1
            # Save useful values to model for later use
            self.anomaly_threshold = 0.5 * ( anomaly_scores[change_point] + anomaly_scores[change_point-1] ).item()
            self.result = {metric.__class__.__name__:( metric(anomaly_scores,targets) if i in metrics_need_AS_indx else metric(decisions,targets) ) for i, metric in enumerate(metrics)}
            self.anomaly_scores, self.decisions, self.targets = anomaly_scores, decisions, targets
        self._train()
        return self.result
        
    def to(self, device):
        self.device = device
        for component in self.components:
            component.to(device)

    def save(self, folder="./"):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = self.name+"-"+now+".model"
        torch.save(self.components, join(folder, filename))
        return now

    def load(self, time, folder="./"):
        self.components = torch.load(join(folder, self.name+"-"+time+".model"))
        self._set_component_handles()

    def visualize_anomaly_scores(self, num_bins,upper_lim=None):
        if upper_lim is None: upper_lim = 2 * self.anomaly_threshold
        A, B = torch.clamp(self.anomaly_scores[self.targets == 0],min=None,max=upper_lim), torch.clamp(self.anomaly_scores[self.targets == 1],min=None,max=upper_lim)
        min_value = min(min(A), min(B)).item()
        max_value = max(max(A), max(B)).item()
        width = (max_value - min_value) / num_bins
        bins = np.arange(min_value, max_value + width, width)
        counts_A, _ = np.histogram(A, bins=bins)
        counts_B, _ = np.histogram(B, bins=bins)
        fig, ax = plt.subplots()
        ax.bar(bins[:-1] + width/2, counts_A, width=width, color='mediumblue', alpha=0.5, label='normal')
        ax.bar(bins[:-1] + width/2, counts_B, width=width, color='red',        alpha=0.5, label='anomalous')
        ax.axvline(x=self.anomaly_threshold, color='black', linestyle='--', label=f'Threshold = {self.anomaly_threshold:0.2e}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Value Frequencies (clamped at max={upper_lim:0.2e})')
        ax.legend()
        plt.show()

    def plot_confusion_matrix(self):
        conf_matrix = torch.flip(self.result["BinaryConfusionMatrix"],dims=(0,))
        categories = ['Normal', 'Anomalous']

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(conf_matrix/conf_matrix.sum(axis=1).reshape(2,1), annot=conf_matrix, fmt='d', cmap='Blues', cbar=False, square=True,
                    linewidths=2, linecolor='black', annot_kws={"size": 16}, ax=ax)

        # Title and labels
        ax.set_title('Confusion Matrix', fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)

        # Set x and y ticks with proper labels
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_yticklabels(categories, fontsize=12)

        # Ensure 'Normal' is on the top-left and 'Anomalous' is bottom-right
        ax.xaxis.set_ticklabels(categories, fontsize=12)
        ax.yaxis.set_ticklabels(categories[::-1], fontsize=12)  # Reverse y-axis labels for correct layout

        # Add value labels inside cells (already done by annot=True)
        # Adjust the layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

class AutoEncoder(ModelBase):
    def _set_component_handles(self):
        self.encoder, self.decoder = self.components

    def __init__(self, n_features, latent_dimension, theta, device, use_counterexamples, leaky_relu_slope=0.2):
        self.n_features = n_features
        self.latent_dim = latent_dimension
        self.theta = theta
        self.use_counterexamples = use_counterexamples
        super().__init__("AutoEncoder" + ("" if use_counterexamples else "_no") + "_counterexamples",
            [
                # Encoder
                nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, latent_dimension)
                ).to(device),
                # Decoder
                nn.Sequential(
                    nn.Linear(latent_dimension, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, n_features),
                    nn.Sigmoid()
                ).to(device)
            ],
            device
        )

    def _optimizers_init(self, optim, optimizer_params):
        self.optimizer = optim([*self.encoder.parameters(),*self.decoder.parameters()], **optimizer_params)
    
    def set_inference_params(self, anomaly_threshold):
        self.anomaly_threshold = anomaly_threshold
        self._inference_params_set = True

    def loss(self, normal, *anomal):
        L = torch.mean(( normal - self.decoder(self.encoder(normal)) )**2)
        if self.use_counterexamples:
            anomal = anomal[0]
            L += torch.mean(self.theta / torch.mean(( anomal - self.decoder(self.encoder(anomal)) )**2,axis=1))
        return L

    def _anomaly_score(self, features):
        x   = features
        gex = self.decoder(self.encoder(x))
        return ((gex - x)**2).mean(axis=1)

    def _test_val(self, real_samples):
        if self.use_counterexamples:
            normal, anomal = real_samples
            normal_as = self._anomaly_score(normal)
            anomal_as = self._anomaly_score(anomal)
            return {
                "AS normal mean": normal_as.mean(),
                "AS normal std":  normal_as.std(),
                "AS anomalous mean": anomal_as.mean(),
                "AS anomalous std": anomal_as.std()
            }
        normal_as = self._anomaly_score(real_samples)
        return {
            "AS normal mean": normal_as.mean(),
            "AS normal std":  normal_as.std()
        }

    def _train_step(self, i, epoch, real_samples):
        if self.use_counterexamples:
            eg_loss = self.loss(*real_samples)
        else:
            eg_loss = self.loss(real_samples)
        self.optimizer.zero_grad()
        eg_loss.backward()
        self.optimizer.step()

class VariationalAutoEncoder(ModelBase):
    def _set_component_handles(self):
        self.encoder, self.decoder = self.components

    def __init__(self, n_features, latent_dimension, theta, device, use_counterexamples, beta, gamma=1000.0, max_capacity=25, Capacity_max_iter=1e5, loss_type="H", leaky_relu_slope=0.2):
        self.n_features          = n_features
        self.latent_dim          = latent_dimension
        self.theta               = theta
        self.use_counterexamples = use_counterexamples
        self.beta                = beta
        self.gamma               = gamma
        self.C_max               = max_capacity
        self.C_stop_iter         = Capacity_max_iter
        self.loss_type           = loss_type
        self.num_iter            = 0

        ModelBase.__init__(self,
            "VariationalAutoEncoder",
            [
                # Encoder
                nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, 2*latent_dimension)
                ).to(device),
                # Decoder
                nn.Sequential(
                    nn.Linear(latent_dimension, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, n_features),
                    nn.Sigmoid()
                ).to(device),
            ],
            device
        )

    def _optimizers_init(self, optim, optimizer_params):
        self.optimizer = optim([*self.encoder.parameters(),*self.decoder.parameters()], **optimizer_params)
    
    def set_inference_params(self, anomaly_threshold):
        self.anomaly_threshold = anomaly_threshold
        self._inference_params_set = True

    def vae_loss(self, real_samples):
        if self.use_counterexamples:
            normal, anomal = real_samples
            e_normal, e_anomal = self.encoder(normal), self.encoder(anomal)
            mu_normal, log_var_normal = e_normal[:, :self.latent_dim], e_normal[:, self.latent_dim:]
            norm_reconstr = self.decoder(self.vae_reparameterize(e_normal))
            anom_reconstr = self.decoder(self.vae_reparameterize(e_anomal))
            norm_recons_loss = ((norm_reconstr - normal)**2).mean(axis=1)
            anom_recons_loss = ((anom_reconstr - anomal)**2).mean(axis=1)
            nrl, arl, kld = norm_recons_loss.mean(), (1/anom_recons_loss).mean(), 0.5 * torch.mean(torch.exp(log_var_normal) + mu_normal**2 - 1 - log_var_normal)
            if self.loss_type == "H":
                loss = nrl + self.theta * arl + self.beta * kld
            elif self.loss_type == "B":
                C = max(min(self.C_max/self.C_stop_iter*self.num_iter,self.C_max),0)
                loss = nrl + self.theta * arl + self.gamma * (kld - C).abs()
            else:
                raise ValueError('Undefined loss type.')
            return loss, nrl, arl, kld

        normal = real_samples
        e_normal = self.encoder(normal)
        mu_normal, log_var_normal = e_normal[:, :self.latent_dim], e_normal[:, self.latent_dim:]
        norm_reconstr = self.decoder(self.vae_reparameterize(e_normal))
        norm_recons_loss = ((norm_reconstr - normal)**2).mean(axis=1)
        nrl, kld = norm_recons_loss.mean(), 0.5 * torch.mean(torch.exp(log_var_normal) + mu_normal**2 - 1 - log_var_normal)
        if self.loss_type == "H":
            loss = nrl + self.beta * kld
        elif self.loss_type == "B":
            C = max(min(self.C_max/self.C_stop_iter*self.num_iter,self.C_max),0)
            loss = nrl + self.gamma * (kld - C).abs()
        else:
            raise ValueError('Undefined loss type.')
        return loss, nrl, kld

    def _anomaly_score(self, features):
        x   = features
        gex = self.decoder(self.vae_reparameterize(self.encoder(x)))
        return ((gex - x)**2).mean(axis=1)

    def _test_val(self, real_samples):
        if self.use_counterexamples:
            loss, nrl, arl, kld = self.vae_loss(real_samples)
            return {
                "Total loss": loss.item(), "NRL": nrl.item(), "ARL": arl.item(), "KLD": kld.item()
            }
        loss, nrl, kld = self.vae_loss(real_samples)
        return {
            "Total loss": loss.item(), "NRL": nrl.item(), "KLD": kld.item()
        }

    def vae_reparameterize(self,encoded):
        mu,log_var = encoded[:,:self.latent_dim],encoded[:,self.latent_dim:]
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _train_step(self, i, epoch, real_samples):
        self.num_iter += 1
        vae_loss = self.vae_loss(real_samples)[0]
        self.optimizer.zero_grad()
        vae_loss.backward()
        self.optimizer.step()

# An adaptation of the GANomaly model for NIDS
# https://arxiv.org/pdf/1805.06725
class GANomaly_variant(ModelBase):
    class Discriminator(nn.Module):
        def __init__(self, n_features, leaky_relu_slope):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(leaky_relu_slope),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x, return_intermediate=False):
            x = self.layer1(x)
            if return_intermediate:
                return x
            return self.layer2(x)

    def _set_component_handles(self):
        self.Ge, self.Gd, self.E, self.D = self.components

    def __init__(self, n_features, latent_dimension, leaky_relu_slope, device, w_adv, w_con, w_enc):
        self.n_features = n_features
        self.latent_dim = latent_dimension
        super().__init__(
            "GANomaly_variant",
            [
                # Generator encoder
                nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, latent_dimension)
                ).to(device),
                # Generator decoder
                nn.Sequential(
                    nn.Linear(latent_dimension, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, n_features),
                    nn.Sigmoid()
                ).to(device),
                # Encoder
                nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, latent_dimension)
                ).to(device),
                # Discriminator
                self.Discriminator(
                    n_features, leaky_relu_slope
                ).to(device),
            ],
            device
        )
        self.w_adv, self.w_con, self.w_enc = w_adv, w_con, w_enc
        self.l_bce = nn.BCELoss()

    def set_inference_params(self, anomaly_threshold):
        self.anomaly_threshold = anomaly_threshold
        self._inference_params_set = True

    def _optimizers_init(self, optim, optimizer_params):
        self.optimG = optim([*self.Ge.parameters(),*self.Gd.parameters(),*self.E.parameters()], **optimizer_params)
        self.optimD = optim(self.D.parameters(), **optimizer_params)

    def adversarial_loss(self):
        return ((self.D(self.real,True)-self.D(self.fake,True))**2).mean()

    def contextual_loss(self):
        return (self.real-self.fake).abs().mean()

    def encoder_loss(self):
        return ((self.z_real-self.z_fake)**2).mean()

    def loss_g(self):
        adv_l = self.adversarial_loss()
        con_l = self.contextual_loss()
        enc_l = self.encoder_loss()
        return self.w_adv * adv_l + self.w_con * con_l + self.w_enc * enc_l
    
    def loss_d(self):
        real_d, fake_d = self.D(self.real), self.D(self.fake)
        return ( self.l_bce(real_d, torch.ones_like(real_d)) + self.l_bce(fake_d, torch.zeros_like(fake_d)) ) * 0.5

    def _anomaly_score(self, x):
        z1 = self.Ge(x)
        z2 = self.E(self.Gd(z1))
        return (z1 - z2).abs().mean(axis=1)
        # return torch.norm(z1 - z2, p=1, dim=1)

    # # alternative
    # def _anomaly_score(self, x):
    #     x_hat = self.Gd(self.Ge(x))
    #     return torch.norm(x_hat - x, p=1, dim=1)

    def _test_val(self, real_samples):
        with torch.inference_mode():
            self.real = real_samples
            self.z_real = self.Ge(self.real)
            self.fake = self.Gd(self.z_real)
            self.z_fake = self.E(self.fake)
            ano_score = self._anomaly_score(self.real)
            return {"Generator Loss": self.loss_g(), "Discriminator Loss": self.loss_d(), "AS mean": ano_score.mean(), "AS std": ano_score.std()}

    def reinit_d(self):
        print('Discriminator got too good. Resetting parameters...')
        self.D.apply(self._init_weights)

    def _train_step(self, i, epoch, real_samples, n_critic):
        # Forward pass
        self.real = real_samples
        self.z_real = self.Ge(self.real)
        self.fake = self.Gd(self.z_real)
        self.z_fake = self.E(self.fake)
        # Backward pass
        # NetG
        if (i+1) % n_critic == 0:
            self.D.requires_grad_(False)
            g_loss = self.loss_g()
            self.optimG.zero_grad()
            g_loss.backward()
            self.optimG.step()
            self.D.requires_grad_(True)

        # NetD
        self.fake = self.fake.detach()
        d_loss = self.loss_d()
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        if d_loss.item() < 1e-5: self.reinit_d()

# Modification of the above model to include training with counterexamples
class GANomaly_variant_counterex(GANomaly_variant):
    def __init__(self, n_features, latent_dimension, leaky_relu_slope, device, w_adv, w_con, w_enc, theta):
        super().__init__(n_features, latent_dimension, leaky_relu_slope, device, w_adv, w_con, w_enc)
        self.theta = theta

    def adversarial_loss(self):
        return ((self.D(self.real_normal,True)-self.D(self.fake_normal,True))**2).mean()

    def contextual_loss(self):
        return (self.real_normal-self.fake_normal).abs().mean() + \
               (self.theta / (self.real_anomalous-self.fake_anomalous).abs().mean(axis=1)).mean()

    def encoder_loss(self):
        return ((self.z_real_normal-self.z_fake_normal)**2).mean()

    def loss_d(self):
        real_d, fake_d = self.D(self.real_normal), self.D(self.fake_normal)
        return ( self.l_bce(real_d, torch.ones_like(real_d)) + self.l_bce(fake_d, torch.zeros_like(fake_d)) ) * 0.5

    def _test_val(self, real_samples):
        with torch.inference_mode():
            self.real_normal, self.real_anomalous = real_samples
            self.z_real_normal = self.Ge(self.real_normal)
            self.fake_normal = self.Gd(self.z_real_normal)
            self.z_fake_normal = self.E(self.fake_normal)
            ano_score_normal = self._anomaly_score(self.real_normal)

            self.z_real_anomalous = self.Ge(self.real_anomalous)
            self.fake_anomalous = self.Gd(self.z_real_anomalous)
            ano_score_anomalous =  self._anomaly_score(self.real_anomalous)
            return {
                "Generator Loss": self.loss_g(), "Discriminator Loss": self.loss_d(),
                "(NORMAL) AS mean": ano_score_normal.mean(), "(NORMAL) AS std": ano_score_normal.std(),
                "(ANOMALOUS) AS mean": ano_score_anomalous.mean(), "(ANOMALOUS) AS std": ano_score_anomalous.std()
            }

    def _train_step(self, i, epoch, real_samples, n_critic):
        # Forward pass
        self.real_normal, self.real_anomalous = real_samples
        self.z_real_normal = self.Ge(self.real_normal)
        self.fake_normal = self.Gd(self.z_real_normal)
        self.z_fake_normal = self.E(self.fake_normal)

        # Model is put into eval mode so that any BatchNorm-type component doesn't fit the anomalous data
        self._eval()
        self.z_real_anomalous = self.Ge(self.real_anomalous)
        self.fake_anomalous = self.Gd(self.z_real_anomalous)
        self._train()

        # Backward pass
        # NetG
        if (i+1) % n_critic == 0:
            self.D.requires_grad_(False)
            g_loss = self.loss_g()
            self.optimG.zero_grad()
            g_loss.backward()
            self.optimG.step()
            self.D.requires_grad_(True)

        # NetD
        self.fake_normal = self.fake_normal.detach()
        d_loss = self.loss_d()
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        if d_loss.item() < 1e-5: self.reinit_d()

# My implementation of the model proposed in the paper:
# https://www.sciencedirect.com/science/article/pii/S1084804523000413
class BiWGAN_GP(ModelBase):
    class Discriminator(nn.Module):
        def __init__(self, n_features, latent_dimension, leaky_relu_slope):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(n_features + latent_dimension, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(leaky_relu_slope),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x, return_intermediate=False):
            x = self.layer1(x)
            if return_intermediate:
                return x
            return self.layer2(x)

    def _set_component_handles(self):
        self.encoder, self.generator, self.discriminator, self.classifier = self.components

    def __init__(self, n_features, latent_dimension, leaky_relu_slope, device, lambda_gp, sigma):
        self.n_features = n_features
        self.latent_dim = latent_dimension
        self.lambda_gp = lambda_gp
        self.sigma     = sigma

        super().__init__(
            "BiWGAN_GP",
            [
                # Encoder
                nn.Sequential(
                    nn.Linear(n_features, 64),
                    # Batch normalization is not recommended in the original WGAN-GP paper
                    # Instead, layernorm is used
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, latent_dimension)
                ).to(device),
                # Generator
                nn.Sequential(
                    nn.Linear(latent_dimension, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, n_features),
                    nn.Sigmoid()
                ).to(device),
                # Discriminator
                self.Discriminator(
                    n_features, latent_dimension, leaky_relu_slope
                ).to(device),
                # Classifier
                nn.Sequential(
                    nn.Linear(latent_dimension, 64),
                    nn.LayerNorm(64),
                    nn.LeakyReLU(leaky_relu_slope),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                ).to(device)
            ],
            device
        )

    def set_inference_params(self, anomaly_threshold):
        self.anomaly_threshold = anomaly_threshold
        self._inference_params_set = True

    def _optimizers_init(self, optim, optimizer_params):
        self.optimEG = optim([*self.encoder.parameters(), *self.generator.parameters()], **optimizer_params)
        self.optimD  = optim(self.discriminator.parameters(), **optimizer_params)
        self.optimC  = optim(self.classifier.parameters(), **optimizer_params)

    def _gradient_penalty(self, real, fake, neural_net):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real.size(0), 1), device=self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        NN_interpolates = neural_net(interpolates).squeeze()
        fake = torch.ones(real.shape[0], device=self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=NN_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True, # I suspect that this can go
        )[0]
        gradients = gradients.view(gradients.size(0), -1) # This can probably also go
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _loss_base(self, real, fake, is_discriminator):
        neural_net = self.discriminator if is_discriminator else self.classifier
        real_validity = neural_net(real)
        fake_validity = neural_net(fake)
        gradient_penalty = self._gradient_penalty(real, fake, neural_net)
        loss = torch.mean(fake_validity) - torch.mean(real_validity) + self.lambda_gp * gradient_penalty
        return loss

    def adversarial_loss(self, real, fake):
        return self._loss_base(real, fake, True)
    
    def coding_loss(self, real, fake):
        return self._loss_base(real, fake, False)
    
    def cycle_consistency_loss(self, x):
        reconstr = self.generator(self.encoder(x))
        return torch.mean((x - reconstr).abs())

    def ge_loss(self,real,fake,real_z,fake_z,real_x):
        return + torch.mean(self.discriminator(real)) - torch.mean(self.discriminator(fake)) \
               + torch.mean(self.classifier(real_z))  - torch.mean(self.classifier(fake_z)) \
               + self.sigma * self.cycle_consistency_loss(real_x)

    def _anomaly_score(self, features):
        x   = features
        ex  = self.encoder(x)
        gex = self.generator(ex)
        fd_x_ex, fd_gex_ex = self.discriminator(torch.cat((x,ex),  axis=1),return_intermediate=True), \
                             self.discriminator(torch.cat((gex,ex),axis=1),return_intermediate=True)
        return torch.mean((fd_x_ex - fd_gex_ex).abs(), axis=1)

    def _test_val(self, real_samples):
        ano_score = self._anomaly_score(real_samples)
        return { "AS mean": ano_score.mean(), "AS std": ano_score.std() }

    def _train_step(self, i, epoch, real_samples, n_critic):
        # ----------------------------------
        #  Train Discriminator & Classifier
        # ----------------------------------
        # Sample noise as generator input
        z = torch.randn((real_samples.shape[0], self.latent_dim), device=self.device)
        # Generate a batch of samples
        fake_x_undetached, real_z_undetached = self.generator(z), self.encoder(real_samples)
        fake_x, real_z = fake_x_undetached.detach(), real_z_undetached.detach()
        real = torch.cat((real_samples, real_z), axis=1)
        fake = torch.cat((fake_x, z), axis=1)
        # Discriminator
        d_loss = self.adversarial_loss(real, fake)
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        # Classifier
        c_loss = self.coding_loss(real_z, z)
        self.optimC.zero_grad()
        c_loss.backward()
        self.optimC.step()
        # Train the generator every n_critic steps
        if (i+1) % n_critic == 0:
            # ---------------------------
            #  Train Generator & Encoder
            # ---------------------------
            # Loss measures generator's ability to fool the discriminator
            # Train on fake samples
            real = torch.cat((real_samples, real_z_undetached), axis=1)
            fake = torch.cat((fake_x_undetached, z), axis=1)
            # Turn off gradients for D and C (only G and E are updated)
            self.discriminator.requires_grad_(False)
            self.classifier.requires_grad_(False)
            # Calculate compound loss
            ge_loss = self.ge_loss(real, fake, real_z_undetached, z, real_samples)
            # Backpropagate and update parameters
            self.optimEG.zero_grad()
            ge_loss.backward()
            self.optimEG.step()
            # Turn gradients back on
            self.discriminator.requires_grad_(True)
            self.classifier.requires_grad_(True)

# Modification of the above model to include training with counterexamples
class BiWGAN_GP_counterex(BiWGAN_GP):
    def __init__(self, n_features, latent_dimension, leaky_relu_slope, device, lambda_gp, sigma, theta):
        super().__init__(n_features, latent_dimension, leaky_relu_slope, device, lambda_gp, sigma)
        # Add theta hyperparameter
        self.theta = theta
    
    def cycle_consistency_loss(self, normal, anomal):
        reconstr_n = self.generator(self.encoder(normal))
        reconstr_a = self.generator(self.encoder(anomal))
        return + torch.mean((normal - reconstr_n).abs()) \
               + torch.mean(self.theta / torch.mean((anomal - reconstr_a).abs(),axis=1))

    def ge_loss(self,real,fake,real_z,fake_z,real_x,anomalous):
        return + torch.mean(self.discriminator(real)) - torch.mean(self.discriminator(fake)) \
               + torch.mean(self.classifier(real_z))  - torch.mean(self.classifier(fake_z)) \
               + self.sigma * self.cycle_consistency_loss(real_x,anomalous)

    def _test_val(self, real_samples):
        normal, anomal = real_samples
        ano_score_normal = self._anomaly_score(normal)
        ano_score_anomal = self._anomaly_score(anomal)
        return {
            "(+) AS mean": ano_score_normal.mean(), "(+) AS std": ano_score_normal.std(),
            "(-) AS mean": ano_score_anomal.mean(), "(-) AS std": ano_score_anomal.std(),
        }

    def _train_step(self, i, epoch, real_samples, n_critic):
        # ----------------------------------
        #  Train Discriminator & Classifier
        # ----------------------------------
        normal, anomalous = real_samples
        # Sample noise as generator input
        z = torch.randn((normal.shape[0], self.latent_dim), device=self.device)
        # Generate a batch of samples
        fake_x_undetached, real_z_undetached = self.generator(z), self.encoder(normal)
        fake_x, real_z = fake_x_undetached.detach(), real_z_undetached.detach()
        real = torch.cat((normal, real_z), axis=1)
        fake = torch.cat((fake_x, z), axis=1)
        # Discriminator
        d_loss = self.adversarial_loss(real, fake)
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        # Classifier
        c_loss = self.coding_loss(real_z, z)
        self.optimC.zero_grad()
        c_loss.backward()
        self.optimC.step()
        # Train the generator every n_critic steps
        if (i+1) % n_critic == 0:
            # ---------------------------
            #  Train Generator & Encoder
            # ---------------------------
            # Loss measures generator's ability to fool the discriminator
            # Train on fake samples
            real = torch.cat((normal, real_z_undetached), axis=1)
            fake = torch.cat((fake_x_undetached, z), axis=1)
            # Turn off gradients for D and C (only G and E are updated)
            self.discriminator.requires_grad_(False)
            self.classifier.requires_grad_(False)
            # Calculate compound loss
            ge_loss = self.ge_loss(real, fake, real_z_undetached, z, normal, anomalous)
            # Backpropagate and update parameters
            self.optimEG.zero_grad()
            ge_loss.backward()
            self.optimEG.step()
            # Turn gradients back on
            self.discriminator.requires_grad_(True)
            self.classifier.requires_grad_(True)

class ConvAutoencoder(nn.Module):
    def _init_weights(self, Layer):
        if Layer.__class__.__name__ == 'Linear':
            nn.init.normal_(Layer.weight, mean=0, std=0.02)
            if Layer.bias is not None:
                nn.init.constant_(Layer.bias, 0)

    def calc_correlations(self,ds,i,batch_size):
        matrix = ds.features[i-self.corr_window+1:i+batch_size].to(self.device).T
        ds_len = len(ds)
        stack_size = min(batch_size, ds_len-i)
        ret = torch.stack([self.correlation(matrix[:,i:i+self.corr_window]) for i in range(stack_size)])
        return torch.where(torch.isnan(ret), torch.ones_like(ret), ret)

    def __init__(self, n_features, n_z, n_z_channels=32, corr_window=5, correlation=torch.corrcoef, device='cuda', theta=0):
        super().__init__()
        
        self.n_feat = n_features
        self.n_z = n_z
        self.n_z_channels = n_z_channels
        self.corr_window = corr_window
        self.correlation = correlation
        self.device = device
        self.theta = theta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, n_z_channels, kernel_size=4, stride=4, padding=1),
            nn.Tanh()
        )

        d1 = self.calc_dim(n_features,4,2,1)
        d2 = self.calc_dim(d1,4,2,1)
        d3 = self.calc_dim(d2,4,2,1)
        d4 = self.calc_dim(d3,4,4,1)
        self.conv_out_dim = d4
        # Calculate the size of the feature map after convolution to know how to flatten for FC layers
        self.conv_output_size = self.conv_out_dim ** 2 * n_z_channels

        # Latent space
        self.fc1 = nn.Linear(self.conv_output_size, n_z)  # Fully connected layer to latent space
        self.fc2 = nn.Linear(n_z, self.conv_output_size)  # Fully connected layer from latent space to expanded feature map

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_z_channels, 16, kernel_size=4, stride=4, padding=0),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def calc_dim(self, Hin, kernel_size, stride, padding):
        return floor((Hin + 2*padding - kernel_size)/stride + 1)
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to pass into the fully connected layer
        z = self.fc1(x)
        return z

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), self.n_z_channels, self.conv_out_dim, self.conv_out_dim)  # Reshape back to feature map size for decoding
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def forward(self, x):
        z = self.encode(x.unsqueeze(1))
        x_reconstructed = self.decode(z)
        return x_reconstructed.squeeze(1)

    def train_model(self, dataset, num_epochs, learning_rate=5e-4, batch_size=64, sample_interval=500):
        self.to(self.device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            running_loss_n = 0.0
            running_loss_a = 0.0
            running_loss_t = 0.0
            counter = 0
            for i in range(self.corr_window-1, len(dataset), batch_size):
                in_data = self.calc_correlations(dataset, i, batch_size)
                labels = dataset.labels[i:i+batch_size]
                norm_idx, anom_idx = labels==0, labels!=0
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                out_data = self(in_data)
                in_normal, in_anomal, out_normal, out_anomal = in_data[norm_idx], in_data[anom_idx], out_data[norm_idx], out_data[anom_idx]
                # Check if there are normal samples
                if in_normal.numel() != 0:
                    loss_normal = ((in_normal - out_normal)**2).mean()
                else:
                    loss_normal = torch.tensor(0, device=self.device)
                # Check if there are anomalous samples
                if in_anomal.numel() != 0:
                    assert len(out_anomal.shape) == 3
                    rec_loss = ((in_anomal - out_anomal)**2).mean(axis=(1,2))
                    loss_anomal = ( self.theta / rec_loss ).mean()
                else:
                    loss_anomal = torch.tensor(0, device=self.device)
                loss = loss_normal + loss_anomal
                running_loss_n += loss_normal.item()
                running_loss_a += loss_anomal.item()
                running_loss_t += loss.item()
                counter += 1
                # Backward pass + optimization
                loss.backward()
                optimizer.step()
            len_epoch_str = len(str(num_epochs))
            print(f"Epoch [{epoch + 1:0{len_epoch_str}d}/{num_epochs}]: Normal Loss: {running_loss_n/counter:.6f} | Anomaly Loss: {running_loss_a/counter:.6f} | Total Loss: {running_loss_t/counter:.6f}")

    def test_model(self, dataset, metrics, batch_size, anomaly_fraction, metrics_need_AS_indx={}):
        self.eval()
        self.to(self.device)

        get_anomaly_score = lambda x,y: ((x-y)**2).mean(axis=(1,2))
        anomaly_scores, targets = [], []
        with torch.inference_mode():
            for i in tqdm(range(self.corr_window-1, len(dataset), batch_size)):
                labels = dataset.labels[i:i+batch_size].to(self.device)
                inputs = self.calc_correlations(dataset, i, batch_size)
                outputs = self(inputs)
                anomaly_score = get_anomaly_score(inputs,outputs)
                anomaly_scores.append(anomaly_score)
                targets.append(labels)

            anomaly_scores, targets = torch.cat(tuple(anomaly_scores), dim=0), torch.cat(tuple(targets), dim=0)
            anomaly_scores, sort_indices = torch.sort(anomaly_scores)
            targets = targets[sort_indices]
            anomaly_scores, targets = anomaly_scores.cpu(), targets.cpu()
            decisions = torch.zeros_like(targets)
            change_point = -int(anomaly_fraction*len(decisions))
            decisions[change_point:] = 1
            # Save useful values to model for later use
            self.anomaly_threshold = 0.5 * ( anomaly_scores[change_point] + anomaly_scores[change_point-1] ).item()
            self.result = {
                metric.__class__.__name__:( metric(anomaly_scores,targets) if i in metrics_need_AS_indx \
                else metric(decisions,targets) ) \
                for i, metric in enumerate(metrics)
            }
            self.anomaly_scores, self.decisions, self.targets = anomaly_scores, decisions, targets
        return self.result

    def save(self, folder="./"):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = "ConvAutoencoder-"+now+".model"
        torch.save(self.state_dict(), join(folder, filename))
        return now

    def load(self, time, folder="./"):
        self.load_state_dict(torch.load(join(folder, "ConvAutoencoder-"+time+".model")))
        self._set_component_handles()

    def visualize_anomaly_scores(self, num_bins,upper_lim=None):
        if upper_lim is None: upper_lim = 2 * self.anomaly_threshold
        A, B = torch.clamp(self.anomaly_scores[self.targets == 0],min=None,max=upper_lim), torch.clamp(self.anomaly_scores[self.targets == 1],min=None,max=upper_lim)
        min_value = min(min(A), min(B)).item()
        max_value = max(max(A), max(B)).item()
        width = (max_value - min_value) / num_bins
        bins = np.arange(min_value, max_value + width, width)
        counts_A, _ = np.histogram(A, bins=bins)
        counts_B, _ = np.histogram(B, bins=bins)
        fig, ax = plt.subplots()
        ax.bar(bins[:-1] + width/2, counts_A, width=width, color='mediumblue', alpha=0.5, label='normal')
        ax.bar(bins[:-1] + width/2, counts_B, width=width, color='red',        alpha=0.5, label='anomalous')
        ax.axvline(x=self.anomaly_threshold, color='black', linestyle='--', label=f'Threshold = {self.anomaly_threshold:0.2e}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Value Frequencies (clamped at max={upper_lim:0.2e})')
        ax.legend()
        plt.show()

    def plot_confusion_matrix(self):
        conf_matrix = torch.flip(self.result["BinaryConfusionMatrix"],dims=(0,))
        categories = ['Normal', 'Anomalous']

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(conf_matrix/conf_matrix.sum(axis=1).reshape(2,1), annot=conf_matrix, fmt='d', cmap='Blues', cbar=False, square=True,
                    linewidths=2, linecolor='black', annot_kws={"size": 16}, ax=ax)

        # Title and labels
        ax.set_title('Confusion Matrix', fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)

        # Set x and y ticks with proper labels
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_yticklabels(categories, fontsize=12)

        # Ensure 'Normal' is on the top-left and 'Anomalous' is bottom-right
        ax.xaxis.set_ticklabels(categories, fontsize=12)
        ax.yaxis.set_ticklabels(categories[::-1], fontsize=12)  # Reverse y-axis labels for correct layout

        # Add value labels inside cells (already done by annot=True)
        # Adjust the layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()
