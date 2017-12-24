find_package(LiveSegmentation ${LiveSegmentation_FIND_VERSION} QUIET
             CONFIG
             PATHS ${CMAKE_INSTALL_PREFIX}
             NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LiveSegmentation CONFIG_MODE)
