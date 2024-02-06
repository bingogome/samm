import os
from collections import namedtuple


from segment_anything import sam_model_registry


# A example for training SAM,
# user can specify the model name and the data root
# the class will load the model and train it with the given data
# Support training for MedSAM and Medical SAM Adapter


class SammTraining:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.join = os.path.join

        if model_name == "MedSAM":

            self.train_medsam()

        if model_name == "MedicalSAMAdapter":

            self.train_medical_sam_adapter()

        pass

    def pre_processing(self):
        pass

    def load_pretrained_model(self):
        pass

    def train_medical_sam_adapter(self):
        from thirdparty.MedicalSAMAdapter import train as adapter_train

        adapter_train.run()

    def train_medsam(self):
        from thirdparty.medsam import train_one_gpu as medsam_train

        medsam_train.run()

    def save_model(self):
        pass


if __name__ == "__main__":
    SammTraining("MedicalSAMAdapter")
