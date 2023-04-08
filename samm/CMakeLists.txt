cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(samm)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/samm")
set(EXTENSION_CATEGORY "samm")
set(EXTENSION_CONTRIBUTORS "Yihao Liu (Johns Hopkins University), Jiaming Zhang (Johns Hopkins University), Zhangcong She (Johns Hopkins University)")
set(EXTENSION_DESCRIPTION "This is an 3D Slicer extension to use Meta SAM for medical image segmentation.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/samm.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/samm/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SammBase)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
