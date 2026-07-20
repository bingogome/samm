import qt


class ToolTipManager:
    def __init__(self, owner):
        self.owner = owner
        self.ui = owner.ui

    def configure(self):
        tooltips = {
            "inputVolumeSelector": "Choose the volume to segment.",
            "segmentationSelector": "Choose the output segmentation node.",
            "segmentComboBox": "Choose the segment that receives prediction results.",
            "sliceViewComboBox": "Choose the active slice view for data, prompts, embeddings, and auto predict.",
            "predictionSliceViewComboBox": "Choose the slice view used for 2D prediction.",
            "embeddingSliceViewComboBox": "Choose the slice view to embed.",
            "maskSegmentationSelector": "Choose the segmentation that provides an input mask prompt.",
            "maskSegmentComboBox": "Choose the segment used as the input mask prompt.",
            "modelFamilyComboBox": "Choose the SAM model family.",
            "weightComboBox": "Choose the checkpoint weight to prepare.",
            "modelAvailabilityLabel": "Shows whether the selected checkpoint exists locally.",
            "statusLabel": "Shows the latest server, model, or prediction status.",
            "showServerLogButton": "Show the managed server log path and its most recent output.",
            "embeddingProgressBar": "Shows embedding progress for the selected embedding job.",
            "embeddingStatusLabel": "Shows cache status for the selected volume and weight.",
            "embeddedSlicesListWidget": "Lists cached embeddings for the current volume and weight.",
            "boxVolumeProgressBar": "Shows per-slice progress for the current volume prediction job.",
            "boxVolumeStatusLabel": "Shows the current volume prediction status.",
            "cancelBoxVolumeVideoButton": "Cancel the current slices-as-video prediction.",
            "predictAutoVideoButton": "Propagate the current slice prompts through the selected view as a video.",
            "cancelAutoVideoButton": "Cancel the current auto slices-as-video prediction.",
            "finetuneVolumeSelector": "Choose the volume to add to the finetuning dataset.",
            "finetuneSegmentationSelector": "Choose the segmentation to pair with the finetuning volume.",
            "finetuneDatasetStatusLabel": "Shows source volume and train/validation split counts for the selected finetuning dataset.",
            "finetuneJobStatusLabel": "Shows the active finetuning server job status and log path.",
            "finetuneJobProgressBar": "Shows progress for the active finetuning job.",
        }
        for name, text in tooltips.items():
            self.set(getattr(self.ui, name), text)

    def update(self):
        w = self.owner
        if not w._parameterNode:
            for widget in self.dynamic_widgets():
                self.set(widget, "Unavailable until the module parameter node is ready.")
            return
        connected = w.logic.client is not None
        started = w.logic.server_running()
        box_running = w.logic.box_volume_prediction_running()
        embedding_running = w.logic.embedding_job_running()
        weight = w._weights.get(w.currentWeightId())
        capabilities = weight.get("capabilities", {}) if weight else {}
        text_blocks_interactive = w.textBlocksInteractivePrompts(weight)
        supports_points = capabilities.get("points", False)
        supports_box = capabilities.get("box", False)
        supports_mask = capabilities.get("mask", False)
        supports_text = capabilities.get("text", False)
        supports_volume_box = capabilities.get("box_3d", False)
        supports_video_propagation = capabilities.get("video_propagation", False)
        supports_auto_predict = capabilities.get("auto_predict_2d", False)
        supports_embeddings = capabilities.get("embeddings", False)
        has_volume = bool(w._parameterNode.inputVolume)
        has_segmentation = bool(w._parameterNode.segmentation)
        has_segment = bool(w._parameterNode.selectedSegmentId)
        has_data = has_volume and has_segmentation and has_segment
        has_mask_segmentation = bool(w._parameterNode.maskSegmentation)
        has_mask_segment = bool(w._parameterNode.maskSegmentId)
        has_volume_box_prompt = w.logic.has_volume_box_prompt(w._parameterNode)
        has_slice_prompt = w.hasEnabledSlicePrompt(w.promptSendOptions(capabilities))
        send_options = w.videoPromptSendOptions(capabilities)
        has_video_points = bool(send_options["use_points"] and w.logic.has_point_prompt(w._parameterNode))
        has_video_box = bool(send_options["use_box"] and w.logic.has_box_prompt(w._parameterNode))
        has_video_mask = bool(send_options["use_mask"] and w.logic.has_mask_prompt(w._parameterNode))
        has_video_text = bool(send_options["use_text"] and w.logic.has_text_prompt(w._parameterNode))
        has_video_prompt = has_video_points or has_video_box or has_video_mask or has_video_text
        video_prompt_compatible = not (has_video_mask and (has_video_points or has_video_box or has_video_text))
        send_mask_prompt = w._parameterNode.sendMaskPrompt
        send_text_prompt = w._parameterNode.sendTextPrompt
        prepared_weight_id = w._prepared["weight_id"] if w._prepared else None
        already_prepared = w.currentWeightId() == prepared_weight_id
        can_embed_reason = self.first_missing(
            (connected, "Start the server first."),
            (w._prepared, "Prepare a model first."),
            (has_volume, "Select an input volume first."),
            (supports_embeddings, "Selected weight does not support cached embeddings."),
            (not embedding_running, "Wait for the embedding job to finish."),
            (not box_running, "Wait for the volume prediction job to finish."),
        )
        predict_reason = self.first_missing(
            (connected, "Start the server first."),
            (w._prepared, "Prepare a model first."),
            (has_data, "Select an input volume, segmentation, and output segment first."),
            (has_slice_prompt, "Add or enable a supported 2D prompt first."),
            (not box_running, "Wait for the volume prediction job to finish."),
        )
        box_volume_reason = self.first_missing(
            (connected, "Start the server first."),
            (w._prepared, "Prepare a model first."),
            (has_data, "Select an input volume, segmentation, and output segment first."),
            (supports_volume_box, "Selected weight does not support 3D box prompts."),
            (has_volume_box_prompt, "Place a 3D box ROI first."),
            (not box_running, "Wait for the current volume prediction job to finish."),
            (not embedding_running, "Wait for the embedding job to finish."),
        )
        box_volume_video_reason = self.first_missing(
            (connected, "Start the server first."),
            (w._prepared, "Prepare a model first."),
            (has_data, "Select an input volume, segmentation, and output segment first."),
            (supports_volume_box, "Selected weight does not support 3D box prompts."),
            (supports_video_propagation, "Selected weight does not support slices-as-video prediction."),
            (has_volume_box_prompt, "Place a 3D box ROI first."),
            (not box_running, "Wait for the current volume prediction job to finish."),
            (not embedding_running, "Wait for the embedding job to finish."),
        )
        auto_video_reason = self.first_missing(
            (connected, "Start the server first."),
            (w._prepared, "Prepare a model first."),
            (has_data, "Select an input volume, segmentation, and output segment first."),
            (supports_video_propagation, "Selected weight does not support slices-as-video prediction."),
            (has_video_prompt, "Add or enable a supported point, 2D box, input mask, or text prompt first."),
            (video_prompt_compatible, "For slices-as-video, use an input mask by itself or disable it to use points, box, or text."),
            (not box_running, "Wait for the current volume prediction job to finish."),
            (not embedding_running, "Wait for the embedding job to finish."),
        )
        auto_reason = self.auto_predict_disabled_reason(capabilities, supports_auto_predict, has_slice_prompt, has_data)
        prompt_context_reason = self.prompt_context_reason(connected, started, weight)
        prepare_reason = self.first_missing(
            (connected, "Start the server first."),
            (weight, "Choose a model weight first."),
            (weight and weight["available"], f"Missing checkpoint: {weight['checkpoint']}" if weight else "Selected checkpoint is missing."),
            (not box_running, "Wait for the volume prediction job to finish."),
            (not already_prepared, "Selected weight is already prepared."),
        )
        self.set(self.ui.startServerButton, self.start_server_tooltip(connected, started))
        self.set(self.ui.stopServerButton, "Stop the SAMM server process." if self.ui.stopServerButton.enabled else "Server is not running.")
        self.set(
            self.ui.showServerLogButton,
            "Show the managed server log path and its most recent output."
            if self.ui.showServerLogButton.enabled
            else "No managed server log is available yet.",
        )
        self.set(self.ui.addSegmentationButton, "Create a SAMM segmentation for predictions.")
        self.set(self.ui.removeSegmentationButton, "Remove the selected segmentation from the scene." if has_segmentation else "No segmentation is selected.")
        self.set(self.ui.saveSegmentationButton, "Save the selected segmentation to disk." if has_segmentation else "Select or create a segmentation first.")
        self.set(self.ui.loadSegmentationButton, "Load a segmentation from disk.")
        self.set(self.ui.addSegmentButton, "Add an output segment to the selected segmentation." if has_segmentation else "Select or create a segmentation first.")
        self.set(self.ui.removeSegmentButton, "Remove the selected output segment." if has_segment else "No output segment is selected.")
        self.set(self.ui.finetuneDatasetLineEdit, "Type a new dataset name or select an existing one below.")
        self.set(self.ui.newFinetuneDatasetButton, "Create an empty source dataset from the name field.")
        self.set(self.ui.finetuneDatasetComboBox, self.dataset_picker_tooltip())
        self.set(self.ui.refreshFinetuneDatasetsButton, self.refresh_datasets_tooltip())
        self.set(self.ui.finetuneVolumeSelector, "Volume to add to the selected finetuning dataset.")
        self.set(self.ui.finetuneSegmentationSelector, "Segmentation labels to add with the selected volume.")
        self.set(self.ui.addSegmentedVolumeButton, self.add_segmented_volume_tooltip(has_volume, has_segmentation))
        self.set(self.ui.finetuneValCountSpinBox, "Number of source volumes to reserve for validation when building the dataset.")
        self.set(self.ui.finetuneAxis0CheckBox, self.axis_tooltip(0))
        self.set(self.ui.finetuneAxis1CheckBox, self.axis_tooltip(1))
        self.set(self.ui.finetuneAxis2CheckBox, self.axis_tooltip(2))
        self.set(self.ui.finetuneUseWindowCheckBox, self.window_toggle_tooltip())
        self.set(self.ui.finetuneWindowMinSpinBox, "Minimum intensity for explicit finetuning build window.")
        self.set(self.ui.finetuneWindowMaxSpinBox, "Maximum intensity for explicit finetuning build window.")
        self.set(self.ui.buildFinetuneDatasetButton, self.build_dataset_tooltip())
        self.set(self.ui.finetuneRunLineEdit, "Finetuning run folder name under finetuning_runs/. Leave empty to use dataset_v1.")
        self.set(self.ui.showFinetuneReportButton, self.report_tooltip())
        self.set(self.ui.finetuneBaseModelComboBox, "Base model used to initialize MedSAM2 finetuning.")
        self.set(self.ui.finetuneEpochsSpinBox, "Number of training epochs.")
        self.set(self.ui.finetuneBatchSizeSpinBox, "Training batch size.")
        self.set(self.ui.finetuneNumFramesSpinBox, "Number of frames sampled per training example.")
        self.set(self.ui.finetuneNumWorkersSpinBox, "Training dataloader workers.")
        self.set(self.ui.startFinetuneTrainingButton, self.train_tooltip())
        self.set(self.ui.finetuneEvalMaxSegmentedVolumesSpinBox, "Maximum validation segmented volumes for eval; 0 means all segmented volumes.")
        self.set(self.ui.startFinetuneEvalButton, self.eval_tooltip())
        self.set(self.ui.finetuneJobProgressBar, self.finetune_progress_tooltip())
        self.set(self.ui.cancelFinetuneJobButton, self.cancel_job_tooltip())
        self.set_prompt_tooltips(prompt_context_reason, has_data, supports_points, supports_box, supports_mask, supports_text, supports_volume_box, text_blocks_interactive, box_running)
        self.set(self.ui.maskSegmentationSelector, self.mask_selector_tooltip(prompt_context_reason, supports_mask, send_mask_prompt, has_mask_segmentation, text_blocks_interactive, box_running))
        self.set(self.ui.maskSegmentComboBox, self.mask_segment_tooltip(prompt_context_reason, supports_mask, send_mask_prompt, has_mask_segmentation, has_mask_segment, text_blocks_interactive, box_running))
        self.set(self.ui.maskSourceLabel, self.mask_selector_tooltip(prompt_context_reason, supports_mask, send_mask_prompt, has_mask_segmentation, text_blocks_interactive, box_running))
        self.set(self.ui.textPromptLineEdit, self.text_prompt_tooltip(prompt_context_reason, supports_text, send_text_prompt, box_running))
        self.set(self.ui.predictSliceButton, "Run prediction on the selected slice." if not predict_reason else predict_reason)
        self.set(self.ui.debugPredictionLogCheckBox, self.debug_log_tooltip(box_running))
        self.set(self.ui.autoPredictCheckBox, self.auto_predict_tooltip(auto_reason))
        self.set(
            self.ui.predictAutoVideoButton,
            "Use the current slice prompts as a seed and propagate through the full selected view in both directions."
            if not auto_video_reason
            else auto_video_reason,
        )
        self.set(
            self.ui.cancelAutoVideoButton,
            "Cancel auto slices-as-video prediction. Completed masks remain in the output segment."
            if w.logic.auto_video_prediction_running()
            else "No auto slices-as-video prediction is running.",
        )
        self.set(self.ui.volumeBoxPromptButton, self.volume_box_prompt_tooltip(prompt_context_reason, has_data, supports_volume_box, box_running))
        self.set(self.ui.clear3dPromptsButton, self.clear_3d_tooltip(box_running))
        self.set(self.ui.predictBoxVolumeButton, "Predict every slice intersecting the 3D box ROI." if not box_volume_reason else box_volume_reason)
        self.set(
            self.ui.predictBoxVolumeVideoButton,
            "Treat the ROI slice range as a video, seed its middle slice with the box, and propagate in both directions."
            if not box_volume_video_reason
            else box_volume_video_reason,
        )
        self.set(
            self.ui.cancelBoxVolumeVideoButton,
            "Cancel the current slices-as-video prediction. Completed masks remain in the output segment."
            if w.logic.video_box_volume_prediction_running()
            else "No slices-as-video prediction is running.",
        )
        self.set(self.ui.embedAllButton, f"Embed all slices in the {w._parameterNode.sliceView} view." if not can_embed_reason else can_embed_reason)
        self.set(self.ui.embedAllAxesButton, "Embed Red, Green, and Yellow views." if not can_embed_reason else can_embed_reason)
        self.set(self.ui.saveEmbeddingsButton, self.save_embeddings_tooltip(can_embed_reason))
        self.set(self.ui.loadEmbeddingsButton, "Load saved embeddings for the selected volume and weight." if not can_embed_reason else can_embed_reason)
        self.set(self.ui.prepareModelButton, f"Prepare {weight['label']} for prediction." if not prepare_reason else prepare_reason)
        self.set(
            self.ui.offloadModelButton,
            "Wait for the volume prediction job to finish."
            if box_running
            else ("Release the prepared model worker." if w._prepared else "No model is prepared."),
        )

    def set(self, widget, text):
        widget.toolTip = text
        widget.setAttribute(qt.Qt.WA_AlwaysShowToolTips, True)

    def dynamic_widgets(self):
        return (
            self.ui.startServerButton,
            self.ui.stopServerButton,
            self.ui.showServerLogButton,
            self.ui.addSegmentationButton,
            self.ui.removeSegmentationButton,
            self.ui.saveSegmentationButton,
            self.ui.loadSegmentationButton,
            self.ui.addSegmentButton,
            self.ui.removeSegmentButton,
            self.ui.finetuneDatasetLineEdit,
            self.ui.newFinetuneDatasetButton,
            self.ui.finetuneDatasetComboBox,
            self.ui.refreshFinetuneDatasetsButton,
            self.ui.finetuneVolumeSelector,
            self.ui.finetuneSegmentationSelector,
            self.ui.addSegmentedVolumeButton,
            self.ui.finetuneValCountSpinBox,
            self.ui.finetuneAxis0CheckBox,
            self.ui.finetuneAxis1CheckBox,
            self.ui.finetuneAxis2CheckBox,
            self.ui.finetuneUseWindowCheckBox,
            self.ui.finetuneWindowMinSpinBox,
            self.ui.finetuneWindowMaxSpinBox,
            self.ui.buildFinetuneDatasetButton,
            self.ui.finetuneRunLineEdit,
            self.ui.showFinetuneReportButton,
            self.ui.finetuneBaseModelComboBox,
            self.ui.finetuneEpochsSpinBox,
            self.ui.finetuneBatchSizeSpinBox,
            self.ui.finetuneNumFramesSpinBox,
            self.ui.finetuneNumWorkersSpinBox,
            self.ui.startFinetuneTrainingButton,
            self.ui.finetuneEvalMaxSegmentedVolumesSpinBox,
            self.ui.startFinetuneEvalButton,
            self.ui.finetuneJobProgressBar,
            self.ui.cancelFinetuneJobButton,
            self.ui.positivePromptButton,
            self.ui.negativePromptButton,
            self.ui.boxPromptButton,
            self.ui.volumeBoxPromptButton,
            self.ui.clear2dPromptsButton,
            self.ui.clear3dPromptsButton,
            self.ui.sendPointPromptsCheckBox,
            self.ui.sendBoxPromptCheckBox,
            self.ui.sendMaskPromptCheckBox,
            self.ui.sendTextPromptCheckBox,
            self.ui.textPromptLabel,
            self.ui.textPromptLineEdit,
            self.ui.clear2dOptionsLabel,
            self.ui.clearPointPromptsCheckBox,
            self.ui.clearBoxPromptCheckBox,
            self.ui.clearMaskPromptCheckBox,
            self.ui.maskSourceLabel,
            self.ui.maskSegmentationSelector,
            self.ui.maskSegmentComboBox,
            self.ui.predictSliceButton,
            self.ui.predictBoxVolumeButton,
            self.ui.predictBoxVolumeVideoButton,
            self.ui.cancelBoxVolumeVideoButton,
            self.ui.debugPredictionLogCheckBox,
            self.ui.autoPredictCheckBox,
            self.ui.predictAutoVideoButton,
            self.ui.cancelAutoVideoButton,
            self.ui.embedAllButton,
            self.ui.embedAllAxesButton,
            self.ui.saveEmbeddingsButton,
            self.ui.loadEmbeddingsButton,
            self.ui.prepareModelButton,
            self.ui.offloadModelButton,
        )

    def set_prompt_tooltips(self, context_reason, has_data, supports_points, supports_box, supports_mask, supports_text, supports_volume_box, text_blocks_interactive, box_running):
        points_reason = self.prompt_disabled_reason(context_reason, has_data, supports_points, text_blocks_interactive, box_running, "point")
        box_reason = self.prompt_disabled_reason(context_reason, has_data, supports_box, False, box_running, "2D box")
        volume_reason = self.prompt_disabled_reason(context_reason, has_data, supports_volume_box, False, box_running, "3D box")
        send_points_reason = self.prompt_toggle_disabled_reason(context_reason, supports_points, text_blocks_interactive, box_running, "point")
        send_box_reason = self.prompt_toggle_disabled_reason(context_reason, supports_box, False, box_running, "2D box")
        send_mask_reason = self.prompt_toggle_disabled_reason(context_reason, supports_mask, text_blocks_interactive, box_running, "input mask")
        send_text_reason = self.prompt_toggle_disabled_reason(context_reason, supports_text, False, box_running, "text")
        self.set(self.ui.positivePromptButton, "Place a positive point on the selected slice." if not points_reason else points_reason)
        self.set(self.ui.negativePromptButton, "Place a negative point on the selected slice." if not points_reason else points_reason)
        self.set(self.ui.boxPromptButton, "Place or replace a 2D box on the selected slice." if not box_reason else box_reason)
        self.set(self.ui.clearPointPromptsCheckBox, self.clear_toggle_tooltip("point prompts", self.ui.clearPointPromptsCheckBox.checked, box_running))
        self.set(self.ui.clearBoxPromptCheckBox, self.clear_toggle_tooltip("the 2D box", self.ui.clearBoxPromptCheckBox.checked, box_running))
        self.set(self.ui.clearMaskPromptCheckBox, self.clear_mask_tooltip(context_reason, supports_mask, text_blocks_interactive, box_running))
        self.set(self.ui.clear2dOptionsLabel, self.clear_2d_tooltip(box_running))
        self.set(self.ui.clear2dPromptsButton, self.clear_2d_tooltip(box_running))
        self.set(self.ui.sendPointPromptsCheckBox, self.send_toggle_tooltip("point prompts", self.ui.sendPointPromptsCheckBox.checked, send_points_reason))
        self.set(self.ui.sendBoxPromptCheckBox, self.send_toggle_tooltip("the 2D box", self.ui.sendBoxPromptCheckBox.checked, send_box_reason))
        self.set(self.ui.sendMaskPromptCheckBox, self.send_toggle_tooltip("the input mask", self.ui.sendMaskPromptCheckBox.checked, send_mask_reason))
        self.set(self.ui.sendTextPromptCheckBox, self.send_toggle_tooltip("text prompt", self.ui.sendTextPromptCheckBox.checked, send_text_reason))
        self.set(self.ui.textPromptLabel, self.text_prompt_tooltip(context_reason, supports_text, self.owner._parameterNode.sendTextPrompt, box_running))
        self.set(self.ui.volumeBoxPromptButton, "Place or replace a 3D box ROI." if not volume_reason else volume_reason)

    def prompt_context_reason(self, connected, started, weight):
        if connected:
            return None if weight else "Choose a model weight first."
        return "Wait for the server connection." if started else "Start the server first."

    def prompt_disabled_reason(self, context_reason, has_data, supported, blocked_by_text, box_running, label):
        return context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (has_data, "Select an input volume, segmentation, and output segment first."),
            (supported, f"Selected weight does not support {label} prompts."),
            (not blocked_by_text, "SAM3 text prompts can be combined with boxes, not points or masks."),
        )

    def prompt_toggle_disabled_reason(self, context_reason, supported, blocked_by_text, box_running, label):
        return context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (supported, f"Selected weight does not support {label} prompts."),
            (not blocked_by_text, "SAM3 text prompts can be combined with boxes, not points or masks."),
        )

    def send_toggle_tooltip(self, label, checked, reason):
        if reason:
            return reason
        return f"Sending {label} is enabled; uncheck to ignore it." if checked else f"Sending {label} is disabled; check to include it."

    def clear_toggle_tooltip(self, label, checked, box_running):
        if box_running:
            return "Wait for the volume prediction job to finish."
        return f"Clear 2D will remove {label}." if checked else f"Clear 2D will keep {label}."

    def clear_mask_tooltip(self, context_reason, supports_mask, text_blocks_interactive, box_running):
        reason = context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (supports_mask, "Selected weight does not support input mask prompts."),
            (not text_blocks_interactive, "SAM3 text prompts cannot be combined with input mask prompts."),
        )
        return reason or self.clear_toggle_tooltip("the input mask selection", self.ui.clearMaskPromptCheckBox.checked, box_running)

    def clear_2d_tooltip(self, box_running):
        if box_running:
            return "Wait for the volume prediction job to finish."
        return "Clear the selected 2D prompt types." if self.ui.clear2dPromptsButton.enabled else "No selected 2D prompt type is present to clear."

    def clear_3d_tooltip(self, box_running):
        if box_running:
            return "Wait for the current volume prediction job to finish."
        return "Clear the current 3D box ROI." if self.ui.clear3dPromptsButton.enabled else "No 3D box ROI is present."

    def mask_selector_tooltip(self, context_reason, supports_mask, send_mask_prompt, has_mask_segmentation, text_blocks_interactive, box_running):
        return context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (supports_mask, "Selected weight does not support input mask prompts."),
            (not text_blocks_interactive, "SAM3 text prompts cannot be combined with input mask prompts."),
            (send_mask_prompt, "Enable Send input mask to use a mask prompt."),
        ) or ("Choose the input mask segmentation." if not has_mask_segmentation else "Input mask segmentation selected.")

    def mask_segment_tooltip(self, context_reason, supports_mask, send_mask_prompt, has_mask_segmentation, has_mask_segment, text_blocks_interactive, box_running):
        return context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (supports_mask, "Selected weight does not support input mask prompts."),
            (not text_blocks_interactive, "SAM3 text prompts cannot be combined with input mask prompts."),
            (send_mask_prompt, "Enable Send input mask to choose a mask segment."),
            (has_mask_segmentation, "Choose an input mask segmentation first."),
        ) or ("Choose the segment used as the input mask prompt." if not has_mask_segment else "Input mask segment selected.")

    def text_prompt_tooltip(self, context_reason, supports_text, send_text_prompt, box_running):
        return context_reason or self.first_missing(
            (not box_running, "Wait for the volume prediction job to finish."),
            (supports_text, "Selected weight does not support text prompts."),
            (send_text_prompt, "Enable Send text to use a text prompt."),
        ) or "Enter a text prompt. SAM3 can combine text with boxes; FastSAM and MedSAM Text use text as their only prompt mode."

    def volume_box_prompt_tooltip(self, context_reason, has_data, supports_volume_box, box_running):
        reason = self.prompt_disabled_reason(context_reason, has_data, supports_volume_box, False, box_running, "3D box")
        return "Place or replace a 3D box ROI." if not reason else reason

    def auto_predict_disabled_reason(self, capabilities, supports_auto_predict, has_slice_prompt, has_data):
        w = self.owner
        if w.logic.client is None:
            return "Start the server first."
        if not w._prepared:
            return "Prepare a model first."
        if not supports_auto_predict:
            return "Selected weight does not support auto predict."
        if not has_data:
            return "Select an input volume, segmentation, and output segment first."
        if not has_slice_prompt:
            return "Add or enable a supported 2D prompt first."
        if capabilities.get("embeddings", False):
            embedded, total = w.logic.view_embedding_count(w._parameterNode)
            if embedded != total:
                return f"Embed {w._parameterNode.sliceView} view first ({embedded} of {total})."
        if w.logic.box_volume_prediction_running() or w.logic.embedding_job_running():
            return "Wait for the running job to finish."
        return None

    def auto_predict_tooltip(self, reason):
        if reason:
            return reason
        return "Auto predict is on; predictions update while scrolling." if self.ui.autoPredictCheckBox.checked else "Enable automatic 2D prediction while scrolling."

    def debug_log_tooltip(self, box_running):
        if box_running:
            return "Wait for the volume prediction job to finish."
        if self.ui.debugPredictionLogCheckBox.checked:
            return "Debug logging is enabled for predictions."
        return "Save input image, prompts, response shape, and overlays for each prediction."

    def save_embeddings_tooltip(self, can_embed_reason):
        if can_embed_reason:
            return can_embed_reason
        return "Save cached embeddings for the current volume and weight." if self.owner.logic.current_embeddings(self.owner._parameterNode) else "Embed slices before saving embeddings."

    def add_segmented_volume_tooltip(self, has_volume, has_segmentation):
        reason = self.first_missing(
            (has_volume, "Select an input volume first."),
            (has_segmentation, "Select or create a segmentation first."),
        )
        return reason or "Add the selected volume and all non-empty segments to the selected finetuning dataset."

    def axis_tooltip(self, axis):
        return f"Include axis {axis} slices when building the finetuning dataset."

    def window_toggle_tooltip(self):
        return "Use the explicit intensity window when building the dataset." if self.ui.finetuneUseWindowCheckBox.checked else "Use percentile windowing when building the dataset."

    def build_dataset_tooltip(self):
        if self.owner.logic.client is None:
            return "Start the server first."
        status = self.owner.logic.finetune_dataset_status(self.owner._parameterNode)
        axes = self.owner.selectedFinetuneAxes()
        val_count = self.ui.finetuneValCountSpinBox.value
        reason = self.first_missing(
            (status["source_segmented_volumes"] > 0, f"No source segmented volumes found in {status['source']}."),
            (bool(axes), "Select at least one axis."),
            (val_count == 0 or val_count < status["source_segmented_volumes"], "Validation count must leave at least one training segmented volume."),
        )
        return reason or "Build train_npz and val_npz from source segmented volumes."

    def report_tooltip(self):
        if self.owner.logic.client is None:
            return "Start the server first."
        return "Show a compact report for the selected finetuning run."

    def dataset_picker_tooltip(self):
        return "Select an existing source or built finetuning dataset."

    def refresh_datasets_tooltip(self):
        return "Refresh available finetuning datasets."

    def train_tooltip(self):
        if self.owner.logic.client is None:
            return "Start the server first."
        if self.owner.finetuneJobRunning():
            return "Wait for the current finetuning job to finish."
        status = self.owner.logic.finetune_dataset_status(self.owner._parameterNode)
        if status["train_segmented_volumes"] < 1:
            return "Build the finetuning dataset first."
        return "Launch MedSAM2 training on the SAMM server."

    def eval_tooltip(self):
        if self.owner.logic.client is None:
            return "Start the server first."
        if self.owner.finetuneJobRunning():
            return "Wait for the current finetuning job to finish."
        return "Launch MedSAM2 eval on the SAMM server using box, point, and box-point prompts."

    def cancel_job_tooltip(self):
        if self.owner.logic.client is None:
            return "Start the server first."
        return "Cancel the current finetuning job." if self.owner.finetuneJobRunning() else "No finetuning job is running."

    def finetune_progress_tooltip(self):
        if not self.owner._finetuneJob:
            return "No finetuning job is running."
        job = self.owner._finetuneJob
        return f"{job['kind']} {job['status']}: {job['progress']['label']}"

    def start_server_tooltip(self, connected, started):
        if self.owner._startingServer:
            return "SAMM server is starting."
        if connected:
            return "SAMM service is already connected."
        if started:
            return "SAMM server process is running; waiting for service connection."
        return "Start the local SAMM server."

    def first_missing(self, *checks):
        for ok, message in checks:
            if not ok:
                return message
        return None
