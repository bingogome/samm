from copy import deepcopy
from functools import partial
from threading import Lock

from .base import BackendError
from .medsam import MedSamBackend


CLIP_MODEL = "openai/clip-vit-base-patch16"


class MedSamTextBackend(MedSamBackend):
    def __init__(self):
        super().__init__()
        self.tokenizers = {}

    def prepare(self, weight, checkpoint_path, device):
        self.use_source()
        try:
            torch = self.required_module("torch")
            self.required_module("segment_anything")
            modeling = self.required_module("segment_anything.modeling")
            transformers = self.required_module("transformers")
            base = self.base_model(weight, modeling, torch)
            prompt_encoder = self.text_prompt_encoder(modeling.PromptEncoder, torch, transformers.CLIPTextModel)
            model = self.text_model(torch)(base.image_encoder, deepcopy(base.mask_decoder), prompt_encoder)
            self.load_text_weights(model, checkpoint_path, torch)
            model.to(device=device)
            model.eval()
            tokenizer = transformers.CLIPTokenizer.from_pretrained(CLIP_MODEL)
        finally:
            self.clear_source_modules()
        self.clear_weight_embeddings(weight.id)
        self.models[weight.id] = model
        self.tokenizers[weight.id] = tokenizer
        self.locks[weight.id] = Lock()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        super().offload()
        self.tokenizers.clear()

    def predict_mask(self, weight, features, shape, text, torch):
        model = self.models[weight.id]
        height, width = shape
        tokens = self.text_tokens(weight, text)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=None, masks=None, tokens=tokens)
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=features,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            logits = torch.sigmoid(low_res_logits)
            logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return (logits.squeeze() > 0.5).to(torch.uint8).cpu().numpy().tobytes()

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        torch = self.required_module("torch")
        features = self.image_embedding(weight, image_bytes, shape, torch)
        with self.locks[weight.id]:
            return self.predict_mask(weight, features, shape[:2], text, torch)

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        torch = self.required_module("torch")
        embedding = self.embeddings.get(embedding_id)
        if not embedding or embedding["weight_id"] != weight.id:
            raise BackendError("embedding not found")
        with embedding["lock"]:
            output = self.predict_mask(weight, embedding["features"], embedding["shape"], text, torch)
        return output, embedding["shape"]

    def validate_prompt(self, points, box, mask, text=None):
        if points or box or mask:
            raise BackendError("MedSAM Text supports text prompts only")
        if not text:
            raise BackendError("MedSAM Text requires a text prompt")

    def text_tokens(self, weight, text):
        tokens = self.tokenizers[weight.id](
            text.strip(),
            max_length=self.tokenizers[weight.id].model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens.to(self.device_for(weight))

    def text_prompt_encoder(self, PromptEncoder, torch, CLIPTextModel):
        class TextPromptEncoder(PromptEncoder):
            def __init__(self):
                super().__init__(
                    embed_dim=256,
                    image_embedding_size=(64, 64),
                    input_image_size=(1024, 1024),
                    mask_in_chans=1,
                    activation=torch.nn.GELU,
                )
                self.text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL)
                self.text_encoder.requires_grad_(False)
                self.text_encoder_head = torch.nn.Linear(512, 256)

            def forward(self, points, boxes, masks, tokens):
                sparse_embeddings = torch.empty((self._get_batch_size(points, boxes, masks, tokens), 0, self.embed_dim), device=self._get_device())
                if points is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, self._embed_points(*points, pad=(boxes is None))], dim=1)
                if boxes is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, self._embed_boxes(boxes)], dim=1)
                if tokens is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, self.text_encoder_head(self.text_encoder(tokens)[0])], dim=1)
                if masks is not None:
                    dense_embeddings = self._embed_masks(masks)
                else:
                    dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                        sparse_embeddings.shape[0], -1, self.image_embedding_size[0], self.image_embedding_size[1]
                    )
                return sparse_embeddings, dense_embeddings

            def _get_batch_size(self, points, boxes, masks, tokens):
                if points is not None:
                    return points[0].shape[0]
                if boxes is not None:
                    return boxes.shape[0]
                if masks is not None:
                    return masks.shape[0]
                if tokens is not None:
                    return tokens.shape[0]
                return 1

        return TextPromptEncoder()

    def base_model(self, weight, modeling, torch):
        if weight.model_type != "vit_b":
            raise BackendError(f"MedSAM Text does not support {weight.model_type}")
        image_size = 1024
        prompt_dim = 256
        return modeling.Sam(
            image_encoder=modeling.ImageEncoderViT(
                depth=12,
                embed_dim=768,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=prompt_dim,
            ),
            prompt_encoder=modeling.PromptEncoder(
                embed_dim=prompt_dim,
                image_embedding_size=(64, 64),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=modeling.MaskDecoder(
                num_multimask_outputs=3,
                transformer=modeling.TwoWayTransformer(depth=2, embedding_dim=prompt_dim, mlp_dim=2048, num_heads=8),
                transformer_dim=prompt_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    def text_model(self, torch):
        class MedSAMText(torch.nn.Module):
            def __init__(self, image_encoder, mask_decoder, prompt_encoder):
                super().__init__()
                self.image_encoder = image_encoder
                self.mask_decoder = mask_decoder
                self.prompt_encoder = prompt_encoder

        return MedSAMText

    def load_text_weights(self, model, checkpoint_path, torch):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        weights = checkpoint["model"] if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict) else checkpoint
        state = model.state_dict()
        required = [key for key in state if not key.startswith("prompt_encoder.text_encoder.")]
        missing = [key for key in required if key not in weights]
        if missing:
            raise BackendError(f"MedSAM Text checkpoint missing key: {missing[0]}")
        for key in required:
            state[key].copy_(weights[key])
