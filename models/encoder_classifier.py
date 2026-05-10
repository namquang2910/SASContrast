from models.net.CNNEncoder import CNNEncoder
from models.model import Model
import torch
import torch.nn as nn
import os

class EncoderClassifierModel(Model):
    """
    Downstream model using only the CNN stem encoder.
    Modes: supervised, random_init, train_linear
    """

    VALID_MODES = {"supervised", "random_init", "train_linear", "fine_tune", "train_linear_proj"}

    def __init__(
        self,
        base_encoder: nn.Module,
        num_class:     int = 1,
        model_path:    str = None,
        training_mode: str = "supervised",
        use_linear: bool = False,
        device=None,
    ):
        super().__init__(device=device)

        self.device        = device
        self.loss_fn       = None
        self.encoder       = base_encoder
        self.training_mode = training_mode
        self.use_linear    = use_linear

        assert training_mode in self.VALID_MODES, \
            f"Invalid mode '{training_mode}'. Must be one of {self.VALID_MODES}"

        # classifier head 
        in_dim = base_encoder.output_dim if self.use_linear == False else base_encoder.cnn_output_dim
        self.classifier = nn.Linear(in_dim, num_class)

        #  weight loading 
        if training_mode in ["train_linear", "fine_tune"]:
            self._load_encoder_only(model_path)
        if training_mode == "train_linear_proj":
            self._load_model(model_path)
            
        #  freezing logic 
        if training_mode in {"train_linear", "random_init"}:
            # freeze both cnn_layers and linear_layer
            for p in self.encoder.cnn_layers.parameters():
                p.requires_grad = False
            print(f"[StemFinetuneModel] {training_mode}: "
                  "cnn_layers + linear_layer frozen, classifier trainable")
            
        elif training_mode == "train_linear_proj":
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[StemFinetuneModel] train_linear_proj: encoder frozen, classifier trainable")
            
        elif training_mode == "supervised":
            print("[StemFinetuneModel] supervised: "
                  "all layers trainable, training from scratch")

        self.check_frozen(self.encoder)

    def _load_model(self, model_path):
        """
        Load a full model checkpoint saved by self.save().
        Expects keys matching self.state_dict() exactly — no stripping needed.
        """
        if not os.path.isfile(model_path):
            print(f"=> no checkpoint found at '{model_path}'")
            exit(1)

        print(f"=> loading checkpoint '{model_path}'")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        self.load_state_dict(checkpoint)   # load into the whole model, not just encoder
        print(f"=> loaded full model from '{model_path}'")

    def _load_encoder_only(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at '{path}'")
        ckpt   = torch.load(path, map_location="cpu", weights_only=False)
        sd     = ckpt.get("state_dict", ckpt)
        enc_sd = {
            k.replace("module.", "", 1)[len("encoder."):]: v
            for k, v in sd.items()
            if k.replace("module.", "", 1).startswith("encoder.")
        }
        msg = self.encoder.load_state_dict(enc_sd, strict=False)
        if msg.missing_keys:
            print(f"[MoEFinetuneModel] Missing keys: {msg.missing_keys}")
        print(f"[MoEFinetuneModel] Loaded encoder from '{path}'")

    def get_parameters(self):
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        cls_params = list(self.classifier.parameters())
        return cls_params, enc_params


    def forward(self, data):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x = data['x'].to(self.device, non_blocking=True).float()
        y = data['y'].to(self.device, non_blocking=True).float()

        x = x.unsqueeze(1).float() 
        if self.use_linear:
            _, h_final     = self.encoder(x, return_embedding=True)        # (B, last_dim=256) — after ReLU
        else:
            h_final, _     = self.encoder(x, return_embedding=True)       # (B, cnn_output_dim=128) — before linear layer
        y_hat = self.classifier(h_final)     # (B, num_class)    — raw logits
        
        y = self._prepare_targets(y)
        loss  = self.loss_fn(y_hat, y)

        return {
            "total_loss": loss,
            "y_hat":      y_hat,
        }