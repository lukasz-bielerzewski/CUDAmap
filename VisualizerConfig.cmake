# - Config file for the Walkers package
# It defines the following variables
#  WALKERS_INCLUDE_DIRS - include directories for Walkers
#  WALKERS_LIBRARIES    - libraries to link against

# Compute paths
get_filename_component(WALKERS_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(WALKERS_INCLUDE_DIRS ";/home/rezzec/Project/include")

# These are IMPORTED targets created by FooBarTargets.cmake
set(WALKERS_LIBRARIES "/home/rezzec/Project/build/lib")
