#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentAnyMedicalModel)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  samm_lib/__init__.py
  samm_lib/client.py
  samm_lib/finetune_export.py
  samm_lib/finetune_models.py
  samm_lib/logic.py
  samm_lib/parameter_node.py
  samm_lib/prediction_debug.py
  samm_lib/prompt_sync.py
  samm_lib/protocol.py
  samm_lib/segmentation_writer.py
  samm_lib/slice_payload.py
  samm_lib/test.py
  samm_lib/tooltips.py
  samm_lib/widget.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/requirements.txt
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
