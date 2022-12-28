# CMake generated Testfile for 
# Source directory: /home/mikedefranco/repos/iGibson/igibson/render/cryptopp
# Build directory: /home/mikedefranco/repos/iGibson/build/temp.linux-x86_64-cpython-38/cryptopp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(build_cryptest "/usr/bin/cmake" "--build" "/home/mikedefranco/repos/iGibson/build/temp.linux-x86_64-cpython-38" "--target" "cryptest")
set_tests_properties(build_cryptest PROPERTIES  _BACKTRACE_TRIPLES "/home/mikedefranco/repos/iGibson/igibson/render/cryptopp/CMakeLists.txt;1139;add_test;/home/mikedefranco/repos/iGibson/igibson/render/cryptopp/CMakeLists.txt;0;")
add_test(cryptest "/home/mikedefranco/repos/iGibson/igibson/render/mesh_renderer/build/cryptest.exe" "v")
set_tests_properties(cryptest PROPERTIES  DEPENDS "build_cryptest" _BACKTRACE_TRIPLES "/home/mikedefranco/repos/iGibson/igibson/render/cryptopp/CMakeLists.txt;1140;add_test;/home/mikedefranco/repos/iGibson/igibson/render/cryptopp/CMakeLists.txt;0;")
