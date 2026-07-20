class BackendError(RuntimeError):
    pass


class Backend:
    def prepare(self, weight, checkpoint_path, device):
        raise NotImplementedError

    def offload(self):
        raise NotImplementedError

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        raise NotImplementedError

    def embed(self, weight, image_bytes, shape):
        raise NotImplementedError

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        raise NotImplementedError

    def propagate_video(
        self,
        weight,
        frames,
        prompt,
        direction="both",
        cancel_event=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    ):
        raise NotImplementedError

    def save_embeddings(self, weight, items, path):
        raise NotImplementedError

    def load_embeddings(self, weight, path):
        raise NotImplementedError
