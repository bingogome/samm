from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from samm_lib.logic import SegmentAnyMedicalModelLogic
from samm_lib.test import SegmentAnyMedicalModelTest
from samm_lib.widget import SegmentAnyMedicalModelWidget


class SegmentAnyMedicalModel(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = _("Segment Any Medical Model")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["SAMM contributors"]
        self.parent.helpText = _("Interactive medical image segmentation with SAM-family models.")
        self.parent.acknowledgementText = _("Developed from the SAMM project.")
