get_directory_property(headers
                        DIRECTORY ${CMAKE_SOURCE_DIR}/tests
                        DEFINITION headers)

set(headers ${headers}
${CMAKE_CURRENT_SOURCE_DIR}/test_basic_qp2.h PARENT_SCOPE)

get_directory_property(codegen_headers
                        DIRECTORY ${CMAKE_SOURCE_DIR}/tests
                        DEFINITION codegen_headers)

set(codegen_headers ${codegen_headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/data.h PARENT_SCOPE)
