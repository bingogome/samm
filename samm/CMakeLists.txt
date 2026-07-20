cmake_minimum_required(VERSION 3.28.0...3.28.0 FATAL_ERROR)

project(samm)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/bingogome/samm")
set(EXTENSION_CONTRIBUTORS "SAMM contributors")
set(EXTENSION_DESCRIPTION "Interactive medical image segmentation with SAM-family models")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/bingogome/samm/main/samm/samm.png")
set(EXTENSION_SCREENSHOTURLS "NA")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentAnyMedicalModel)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
